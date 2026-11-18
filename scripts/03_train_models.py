"""Stage 3: train and compare candidate-classification models.

Ten models from seven algorithm families are fitted on the 80 training trips
with group-aware cross-validation (groups = trips, so no trip is ever split
across folds) and then evaluated on the ten student-testing trips.

Run:  python scripts/03_train_models.py [--quick]
Out:  results/model_comparison.csv
      results/model_cv_folds.csv
      results/test_predictions_<model>.parquet
      results/figures/model_*.png
      artifacts/models/<model>.joblib
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
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap.candidates import FEATURE_NAMES
from pathsnap.dataset import labelled
from pathsnap.metrics import classification_metrics, route_continuity, top1_accuracy
from pathsnap.network import build_network
from pathsnap.trips import load_trip

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts" / "models"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
N_FOLDS = 5

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def numeric_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Median imputation (the first/last fix has no neighbour features),
    optional standardisation for the distance- and gradient-based models."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def model_zoo(quick: bool = False) -> dict:
    """Ten models spanning linear, probabilistic, instance-based, tree,
    ensemble, boosting and neural-network families."""
    n_est = 60 if quick else 200
    zoo = {
        "logistic_regression": {
            "family": "Linear / probabilistic",
            "config": "lbfgs, C=1.0, class_weight=balanced, standardised",
            "pipeline": numeric_pipeline(
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1), scale=True),
            "subsample": None,
        },
        "gaussian_nb": {
            "family": "Probabilistic",
            "config": "var_smoothing=1e-9, standardised",
            "pipeline": numeric_pipeline(GaussianNB(), scale=True),
            "subsample": None,
        },
        "knn_distance": {
            "family": "Instance-based",
            "config": "k=15, distance-weighted, standardised, 150k-row subsample",
            "pipeline": numeric_pipeline(
                KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
                scale=True),
            "subsample": 150_000,
        },
        "decision_tree": {
            "family": "Tree-based",
            "config": "gini, max_depth=14, min_samples_leaf=50, balanced",
            "pipeline": numeric_pipeline(
                DecisionTreeClassifier(max_depth=14, min_samples_leaf=50,
                                       class_weight="balanced", random_state=SEED)),
            "subsample": None,
        },
        "random_forest": {
            "family": "Ensemble (bagging)",
            "config": f"{n_est} trees, max_depth=20, min_samples_leaf=5, balanced_subsample",
            "pipeline": numeric_pipeline(
                RandomForestClassifier(n_estimators=n_est, max_depth=20,
                                       min_samples_leaf=5, n_jobs=-1,
                                       class_weight="balanced_subsample",
                                       random_state=SEED)),
            "subsample": None,
        },
        "extra_trees": {
            "family": "Ensemble (randomised trees)",
            "config": f"{n_est} trees, max_depth=20, min_samples_leaf=5",
            "pipeline": numeric_pipeline(
                ExtraTreesClassifier(n_estimators=n_est, max_depth=20,
                                     min_samples_leaf=5, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     random_state=SEED)),
            "subsample": None,
        },
        "hist_gradient_boosting": {
            "family": "Boosting",
            "config": "lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping",
            "pipeline": numeric_pipeline(
                HistGradientBoostingClassifier(learning_rate=0.1,
                                               max_iter=100 if quick else 300,
                                               max_leaf_nodes=31,
                                               early_stopping=True,
                                               validation_fraction=0.1,
                                               random_state=SEED)),
            "subsample": None,
        },
        "mlp": {
            "family": "Neural network",
            "config": "hidden=(64, 32), relu, adam, early stopping, standardised",
            "pipeline": numeric_pipeline(
                MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              solver="adam", early_stopping=True,
                              max_iter=40 if quick else 120, random_state=SEED),
                scale=True),
            "subsample": 400_000,
        },
    }

    try:
        from lightgbm import LGBMClassifier
        zoo["lightgbm"] = {
            "family": "Boosting (GBDT)",
            "config": f"{n_est} trees, lr=0.05, num_leaves=63, is_unbalance=True",
            "pipeline": numeric_pipeline(
                LGBMClassifier(n_estimators=n_est, learning_rate=0.05,
                               num_leaves=63, is_unbalance=True,
                               random_state=SEED, n_jobs=-1, verbose=-1)),
            "subsample": None,
        }
    except ImportError:
        print("  (lightgbm unavailable, skipping)")

    try:
        from xgboost import XGBClassifier
        zoo["xgboost"] = {
            "family": "Boosting (GBDT)",
            "config": f"{n_est} trees, lr=0.05, max_depth=7, scale_pos_weight tuned",
            "pipeline": numeric_pipeline(
                XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=7,
                              subsample=0.9, colsample_bytree=0.9,
                              eval_metric="logloss", tree_method="hist",
                              random_state=SEED, n_jobs=-1)),
            "subsample": None,
        }
    except ImportError:
        print("  (xgboost unavailable, skipping)")

    return zoo


def subsample_rows(X, y, groups, n, seed=SEED):
    if n is None or n >= len(X):
        return X, y, groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    idx.sort()
    return X[idx], y[idx], groups[idx]


def load_partitions():
    train = pd.read_parquet(PROCESSED / "train_candidates.parquet")
    test = pd.read_parquet(PROCESSED / "student_test_candidates.parquet")
    return labelled(train).reset_index(drop=True), labelled(test).reset_index(drop=True)


def build_test_networks(test_df):
    """Networks for the student-testing trips, reused for continuity scoring."""
    nets = {}
    for tid in sorted(test_df.trip_id.unique()):
        trip = load_trip(ROOT / "data" / "raw" / "student_test", tid)
        nets[tid] = build_network(trip)
    return nets


def continuity_for_predictions(pred_df, nets) -> float:
    """Mean route continuity of the top-1 arc sequence across test trips."""
    vals = []
    for tid, grp in pred_df.groupby("trip_id"):
        seq = grp.sort_values("obs_idx")["arc_id"].to_numpy()
        vals.append(route_continuity(seq, nets[tid]))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller forests/epochs for a fast sanity run")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Loading candidate tables...")
    train_df, test_df = load_partitions()
    print(f"  train: {len(train_df):,} rows / {train_df.trip_id.nunique()} trips")
    print(f"  test : {len(test_df):,} rows / {test_df.trip_id.nunique()} trips")

    X = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = train_df["label"].to_numpy(dtype=np.int8)
    groups = train_df["trip_id"].to_numpy()
    Xte = test_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    yte = test_df["label"].to_numpy(dtype=np.int8)

    nets = build_test_networks(test_df)
    zoo = model_zoo(args.quick)
    if args.models:
        zoo = {k: v for k, v in zoo.items() if k in args.models}

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X, y, groups))
    rows, fold_rows = [], []

    for name, spec in zoo.items():
        print(f"\n=== {name} ({spec['family']}) ===", flush=True)
        fold_f1, fold_ll, fold_top1 = [], [], []

        for k, (tr, va) in enumerate(splits, 1):
            Xtr, ytr, gtr = subsample_rows(X[tr], y[tr], groups[tr], spec["subsample"])
            model = clone(spec["pipeline"])
            t0 = time.time()
            model.fit(Xtr, ytr)
            fit_s = time.time() - t0
            prob = model.predict_proba(X[va])[:, 1]
            m = classification_metrics(y[va], prob)

            fold_view = train_df.iloc[va][["trip_id", "obs_idx", "label"]].copy()
            fold_view["score"] = prob
            t1 = top1_accuracy(fold_view)

            fold_f1.append(m["macro_f1"]); fold_ll.append(m["log_loss"]); fold_top1.append(t1)
            fold_rows.append({"model": name, "fold": k, "macro_f1": m["macro_f1"],
                              "log_loss": m["log_loss"], "top1_accuracy": t1,
                              "precision_macro": m["precision_macro"],
                              "recall_macro": m["recall_macro"],
                              "fit_seconds": fit_s,
                              "n_train_rows": len(Xtr),
                              "n_val_trips": len(np.unique(groups[va]))})
            print(f"  fold {k}: macro-F1 {m['macro_f1']:.4f} | top-1 {t1:.4f} | "
                  f"log-loss {m['log_loss']:.4f} | {fit_s:.1f}s", flush=True)

        # refit on all training trips, then evaluate on student_test
        Xtr, ytr, _ = subsample_rows(X, y, groups, spec["subsample"])
        model = clone(spec["pipeline"])
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_s = time.time() - t0

        t0 = time.time()
        prob_te = model.predict_proba(Xte)[:, 1]
        infer_s = time.time() - t0

        te = classification_metrics(yte, prob_te)
        pred_view = test_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
        pred_view["score"] = prob_te
        top1 = top1_accuracy(pred_view)
        best = pred_view.loc[pred_view.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
        cont = continuity_for_predictions(best, nets)

        pred_view.to_parquet(RESULTS / f"test_predictions_{name}.parquet", index=False)
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")

        rows.append({
            "model": name,
            "algorithm_family": spec["family"],
            "key_configuration": spec["config"],
            "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "cv_macro_f1_std": float(np.std(fold_f1)),
            "cv_top1_accuracy_mean": float(np.mean(fold_top1)),
            "cv_log_loss_mean": float(np.nanmean(fold_ll)),
            "test_top1_accuracy": top1,
            "test_macro_f1": te["macro_f1"],
            "test_precision_macro": te["precision_macro"],
            "test_recall_macro": te["recall_macro"],
            "test_f1_positive": te["f1_positive"],
            "test_accuracy": te["accuracy"],
            "test_log_loss": te["log_loss"],
            "test_route_continuity": cont,
            "train_seconds": train_s,
            "inference_seconds": infer_s,
            "inference_us_per_row": infer_s / len(Xte) * 1e6,
            "n_train_rows_used": len(Xtr),
        })
        print(f"  -> test top-1 {top1:.4f} | macro-F1 {te['macro_f1']:.4f} | "
              f"log-loss {te['log_loss']:.4f} | continuity {cont:.4f}", flush=True)

    comp = pd.DataFrame(rows).sort_values("test_top1_accuracy", ascending=False)
    comp.to_csv(RESULTS / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "model_cv_folds.csv", index=False)

    with open(RESULTS / "experiment_config.json", "w") as f:
        json.dump({"seed": SEED, "n_folds": N_FOLDS, "quick": args.quick,
                   "features": FEATURE_NAMES,
                   "n_train_rows": int(len(X)), "n_test_rows": int(len(Xte))}, f, indent=2)

    make_comparison_plots(comp, pd.DataFrame(fold_rows))

    print("\n=== model comparison ===")
    print(comp[["model", "algorithm_family", "cv_macro_f1_mean", "test_top1_accuracy",
                "test_macro_f1", "test_log_loss", "test_route_continuity",
                "train_seconds"]].to_string(index=False))
    print(f"\nBest by test top-1: {comp.iloc[0]['model']}")


def make_comparison_plots(comp: pd.DataFrame, folds: pd.DataFrame) -> None:
    order = comp.sort_values("cv_macro_f1_mean")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(order.model, order.cv_macro_f1_mean,
            xerr=order.cv_macro_f1_std, color="#4C72B0", capsize=3)
    ax.set(xlabel="Cross-validated macro F1 (5 group folds)",
           title="Cross-validation macro F1 by model")
    fig.tight_layout(); fig.savefig(FIGS / "model_cv_macro_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = np.arange(len(comp)); w = 0.38
    s = comp.sort_values("test_top1_accuracy")
    ax.barh(idx - w / 2, s.test_top1_accuracy, w, label="Top-1 accuracy", color="#4C72B0")
    ax.barh(idx + w / 2, s.test_macro_f1, w, label="Macro F1", color="#DD8452")
    ax.set(yticks=idx, yticklabels=s.model, xlabel="score",
           title="Student-testing performance")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "model_test_performance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    s = comp.sort_values("test_log_loss", ascending=False)
    ax.barh(s.model, s.test_log_loss, color="#C44E52")
    ax.set(xlabel="log loss (lower is better)", title="Student-testing log loss")
    fig.tight_layout(); fig.savefig(FIGS / "model_log_loss.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(comp.train_seconds, comp.test_top1_accuracy, s=60, color="#55A868")
    for _, r in comp.iterrows():
        ax.annotate(r.model, (r.train_seconds, r.test_top1_accuracy),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="training time (s, log scale)",
           ylabel="student-testing top-1 accuracy",
           title="Accuracy against training cost")
    fig.tight_layout(); fig.savefig(FIGS / "model_accuracy_vs_cost.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    piv = folds.pivot(index="fold", columns="model", values="macro_f1")
    piv.plot(ax=ax, marker="o", lw=1.2)
    ax.set(xlabel="group fold", ylabel="macro F1",
           title="Per-fold stability across models")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "model_fold_stability.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    s = comp.sort_values("test_route_continuity")
    ax.barh(s.model, s.test_route_continuity, color="#8172B3")
    ax.set(xlabel="route continuity of the classifier-only sequence",
           title="Route continuity before HMM decoding")
    fig.tight_layout(); fig.savefig(FIGS / "model_route_continuity.png"); plt.close(fig)
    print(f"  figures -> {FIGS}")


if __name__ == "__main__":
    main()
