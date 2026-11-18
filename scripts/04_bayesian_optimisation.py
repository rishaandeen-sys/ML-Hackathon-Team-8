"""Stage 4: Bayesian hyperparameter optimisation for the selected model.

Search strategy
    Optuna's TPE sampler.  TPE is a Bayesian method: it models p(x | y) for
    good and bad trials and samples where the ratio of those densities is
    highest, so each trial is chosen using everything learned from the
    previous ones rather than from a fixed grid.

Objective
    Observation-level Top-1 accuracy under group-aware cross-validation on the
    80 training trips.  Top-1 is the quantity the rest of the system depends
    on -- it is what the HMM consumes as its emission model, and a candidate
    that is ranked first is the one that reaches the route.  Binary macro F1
    is recorded alongside it but is not the search target, because a model can
    improve its macro F1 by being better at rejecting obviously wrong
    candidates without ever improving the choice that actually matters.

Leakage control
    Folds are grouped by trip, so no trip is ever split across folds.  The
    student-testing partition is untouched during the search and is only used
    afterwards to report tuned-versus-untuned performance.  The evaluator
    partition is never read by this script.

Run:  python scripts/04_bayesian_optimisation.py [--model NAME] [--trials N]
Out:  results/bayesopt_trials.csv
      results/bayesopt_summary.csv
      results/bayesopt_best_params.json
      results/figures/bayesopt_*.png
      artifacts/models/tuned_<model>.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURE_NAMES
from pathsnap.dataset import labelled
from pathsnap.metrics import classification_metrics, route_continuity, top1_accuracy
from pathsnap.network import build_network
from pathsnap.trips import load_trip

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts" / "models"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
SEARCH_FOLDS = 3     # folds inside the search, for compute budget
FINAL_FOLDS = 5      # folds for the tuned-vs-untuned comparison (matches stage 3)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


# --------------------------------------------------------------- search spaces
def suggest_lightgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    }


def suggest_xgboost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def suggest_hist_gb(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 150, 800, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 5.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
    }


def suggest_random_forest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 8, 34),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40, log=True),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


SPACES = {
    "lightgbm": suggest_lightgbm,
    "xgboost": suggest_xgboost,
    "hist_gradient_boosting": suggest_hist_gb,
    "random_forest": suggest_random_forest,
    "extra_trees": suggest_random_forest,
}


def make_estimator(model_name: str, params: dict):
    """Rebuild the stage-3 pipeline with the trial's hyperparameters."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(is_unbalance=True, random_state=SEED, n_jobs=-1,
                             verbose=-1, **params)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(eval_metric="logloss", tree_method="hist",
                            random_state=SEED, n_jobs=-1, **params)
    elif model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        est = HistGradientBoostingClassifier(early_stopping=True,
                                             validation_fraction=0.1,
                                             random_state=SEED, **params)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=SEED, **params)
    elif model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        est = ExtraTreesClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=SEED, **params)
    else:
        raise ValueError(f"no search space defined for {model_name}")

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])


# ------------------------------------------------------------------ evaluation
def cv_evaluate(pipeline, X, y, groups, frame, n_folds: int):
    """Group-aware CV returning top-1 accuracy, macro F1 and log loss."""
    gkf = GroupKFold(n_splits=n_folds)
    top1s, f1s, lls = [], [], []
    for tr, va in gkf.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[va])[:, 1]
        view = frame.iloc[va][["trip_id", "obs_idx", "label"]].copy()
        view["score"] = prob
        top1s.append(top1_accuracy(view))
        m = classification_metrics(y[va], prob)
        f1s.append(m["macro_f1"]); lls.append(m["log_loss"])
    return (float(np.mean(top1s)), float(np.std(top1s)),
            float(np.mean(f1s)), float(np.std(f1s)), float(np.nanmean(lls)))


def evaluate_on_test(pipeline, X, y, groups, Xte, yte, test_df, nets):
    """Fit on all training trips and score the student-testing partition."""
    model = clone(pipeline)
    t0 = time.time(); model.fit(X, y); train_s = time.time() - t0
    t0 = time.time(); prob = model.predict_proba(Xte)[:, 1]; infer_s = time.time() - t0

    m = classification_metrics(yte, prob)
    view = test_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    view["score"] = prob
    top1 = top1_accuracy(view)
    best = view.loc[view.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    cont = float(np.mean([route_continuity(g.sort_values("obs_idx")["arc_id"].to_numpy(),
                                           nets[t]) for t, g in best.groupby("trip_id")]))
    return model, view, {
        "test_top1_accuracy": top1, "test_macro_f1": m["macro_f1"],
        "test_log_loss": m["log_loss"], "test_precision_macro": m["precision_macro"],
        "test_recall_macro": m["recall_macro"], "test_route_continuity": cont,
        "train_seconds": train_s, "inference_seconds": infer_s,
        "inference_us_per_row": infer_s / len(Xte) * 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model to tune; defaults to the stage-3 winner")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=5400, help="search timeout, seconds")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    if args.model:
        model_name = args.model
    else:
        tunable = comparison[comparison.model.isin(SPACES)]
        if tunable.empty:
            raise SystemExit("no tunable model found in model_comparison.csv")
        model_name = tunable.sort_values("test_top1_accuracy", ascending=False).iloc[0].model
    print(f"Tuning: {model_name}")

    train_df = labelled(pd.read_parquet(PROCESSED / "train_candidates.parquet")).reset_index(drop=True)
    test_df = labelled(pd.read_parquet(PROCESSED / "student_test_candidates.parquet")).reset_index(drop=True)
    X = train_df[FEATURE_NAMES].to_numpy(np.float64)
    y = train_df["label"].to_numpy(np.int8)
    groups = train_df["trip_id"].to_numpy()
    Xte = test_df[FEATURE_NAMES].to_numpy(np.float64)
    yte = test_df["label"].to_numpy(np.int8)

    nets = {t: build_network(load_trip(ROOT / "data" / "raw" / "student_test", t))
            for t in sorted(test_df.trip_id.unique())}

    suggest = SPACES[model_name]
    search_gkf = list(GroupKFold(n_splits=SEARCH_FOLDS).split(X, y, groups))
    trial_log = []

    def objective(trial):
        params = suggest(trial)
        pipeline = make_estimator(model_name, params)
        t0 = time.time()
        top1s, f1s = [], []
        for tr, va in search_gkf:
            model = clone(pipeline)
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[va])[:, 1]
            view = train_df.iloc[va][["trip_id", "obs_idx", "label"]].copy()
            view["score"] = prob
            top1s.append(top1_accuracy(view))
            f1s.append(classification_metrics(y[va], prob)["macro_f1"])
        elapsed = time.time() - t0
        score = float(np.mean(top1s))
        trial_log.append({"trial": trial.number, "cv_top1_accuracy": score,
                          "cv_top1_std": float(np.std(top1s)),
                          "cv_macro_f1": float(np.mean(f1s)),
                          "seconds": elapsed, **params})
        print(f"  trial {trial.number:3d}: top-1 {score:.4f}  "
              f"macro-F1 {np.mean(f1s):.4f}  ({elapsed:.0f}s)", flush=True)
        return score

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"pathsnap_{model_name}")
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_seconds = time.time() - t0

    best_params = study.best_params
    print(f"\nBest CV top-1: {study.best_value:.4f}")
    print("Best params:", json.dumps(best_params, indent=2))

    trials_df = pd.DataFrame(trial_log)
    trials_df.to_csv(RESULTS / "bayesopt_trials.csv", index=False)

    # ---- tuned vs untuned, identical protocol for both
    untuned_row = comparison[comparison.model == model_name].iloc[0]
    untuned_pipeline = joblib.load(ARTIFACTS / f"{model_name}.joblib")
    tuned_pipeline = make_estimator(model_name, best_params)

    print(f"\nFinal {FINAL_FOLDS}-fold comparison (untuned)...")
    u_top1, u_top1s, u_f1, u_f1s, u_ll = cv_evaluate(untuned_pipeline, X, y, groups,
                                                     train_df, FINAL_FOLDS)
    print(f"Final {FINAL_FOLDS}-fold comparison (tuned)...")
    t_top1, t_top1s, t_f1, t_f1s, t_ll = cv_evaluate(tuned_pipeline, X, y, groups,
                                                     train_df, FINAL_FOLDS)

    tuned_model, tuned_view, tuned_test = evaluate_on_test(
        tuned_pipeline, X, y, groups, Xte, yte, test_df, nets)
    joblib.dump(tuned_model, ARTIFACTS / f"tuned_{model_name}.joblib")
    tuned_view.to_parquet(RESULTS / f"test_predictions_tuned_{model_name}.parquet",
                          index=False)

    summary = pd.DataFrame([
        {"variant": "untuned", "model": model_name,
         "cv_top1_accuracy": u_top1, "cv_top1_std": u_top1s,
         "cv_macro_f1": u_f1, "cv_macro_f1_std": u_f1s, "cv_log_loss": u_ll,
         "test_top1_accuracy": untuned_row.test_top1_accuracy,
         "test_macro_f1": untuned_row.test_macro_f1,
         "test_log_loss": untuned_row.test_log_loss,
         "test_route_continuity": untuned_row.test_route_continuity,
         "train_seconds": untuned_row.train_seconds,
         "inference_us_per_row": untuned_row.inference_us_per_row},
        {"variant": "tuned", "model": model_name,
         "cv_top1_accuracy": t_top1, "cv_top1_std": t_top1s,
         "cv_macro_f1": t_f1, "cv_macro_f1_std": t_f1s, "cv_log_loss": t_ll,
         **{k: v for k, v in tuned_test.items()
            if k in ("test_top1_accuracy", "test_macro_f1", "test_log_loss",
                     "test_route_continuity", "train_seconds", "inference_us_per_row")}},
    ])
    summary.to_csv(RESULTS / "bayesopt_summary.csv", index=False)

    with open(RESULTS / "bayesopt_best_params.json", "w") as f:
        json.dump({"model": model_name, "best_params": best_params,
                   "best_cv_top1": study.best_value,
                   "n_trials": len(trials_df),
                   "search_folds": SEARCH_FOLDS,
                   "final_folds": FINAL_FOLDS,
                   "sampler": "TPE (Optuna)",
                   "objective": "group-CV observation-level top-1 accuracy",
                   "search_seconds": search_seconds,
                   "seed": SEED}, f, indent=2)

    make_plots(trials_df, summary, model_name)

    print("\n=== tuned vs untuned ===")
    print(summary.to_string(index=False))


def make_plots(trials: pd.DataFrame, summary: pd.DataFrame, model_name: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(trials.trial, trials.cv_top1_accuracy, "o-", ms=4, lw=1,
               color="#4C72B0", label="trial")
    ax[0].plot(trials.trial, trials.cv_top1_accuracy.cummax(), lw=2,
               color="#C44E52", label="best so far")
    ax[0].set(xlabel="trial", ylabel="CV top-1 accuracy",
              title=f"TPE search history ({model_name})")
    ax[0].legend()

    ax[1].scatter(trials.seconds, trials.cv_top1_accuracy, s=28, color="#55A868")
    ax[1].set(xlabel="trial duration (s)", ylabel="CV top-1 accuracy",
              title="Objective against trial cost")
    fig.tight_layout(); fig.savefig(FIGS / "bayesopt_history.png"); plt.close(fig)

    # how each searched hyperparameter relates to the objective
    param_cols = [c for c in trials.columns
                  if c not in ("trial", "cv_top1_accuracy", "cv_top1_std",
                               "cv_macro_f1", "seconds")]
    n = len(param_cols)
    if n:
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow))
        for axx, col in zip(np.ravel(axes), param_cols):
            axx.scatter(trials[col], trials.cv_top1_accuracy, s=20, color="#4C72B0")
            axx.set_xlabel(col, fontsize=8)
            axx.tick_params(labelsize=7)
        for axx in np.ravel(axes)[n:]:
            axx.axis("off")
        fig.suptitle("Objective against each searched hyperparameter",
                     fontweight="bold")
        fig.tight_layout(); fig.savefig(FIGS / "bayesopt_params.png"); plt.close(fig)

    metrics = ["cv_top1_accuracy", "cv_macro_f1", "test_top1_accuracy",
               "test_macro_f1", "test_route_continuity"]
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(metrics)); w = 0.38
    for i, variant in enumerate(("untuned", "tuned")):
        row = summary[summary.variant == variant].iloc[0]
        ax.bar(idx + (i - 0.5) * w, [row[m] for m in metrics], w, label=variant)
    ax.set(xticks=idx, xticklabels=[m.replace("_", "\n") for m in metrics],
           ylabel="score", title=f"Tuned vs untuned ({model_name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "bayesopt_tuned_vs_untuned.png"); plt.close(fig)
    print(f"  figures -> {FIGS}")


if __name__ == "__main__":
    main()
