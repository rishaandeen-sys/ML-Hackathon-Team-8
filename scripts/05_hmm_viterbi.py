"""Stage 5: HMM + Viterbi route reconstruction and the comparison against
classifier-only predictions.

Two parts:

1. Decoder parameter selection.  A small grid over the transition scale beta
   and the unreachable-transition penalty, scored on held-out *training*
   trips.  The student-testing trips are not used to choose these values, so
   the numbers reported on them in part 2 stay honest.

2. Evaluation.  Classifier-only versus classifier + Viterbi on all ten
   student-testing trips, reporting top-1 accuracy, route continuity and
   sequence-level agreement per trip.

Run:  python scripts/05_hmm_viterbi.py [--model NAME] [--skip-tuning]
Out:  results/hmm_parameter_search.csv
      results/hmm_config.json
      results/hmm_comparison_per_trip.csv
      results/hmm_comparison_summary.csv
      results/figures/hmm_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.hmm import HMMConfig
from pathsnap.network import build_network
from pathsnap.pipeline import PathSnapMatcher, evaluate_against_reference
from pathsnap.trips import list_trips, load_trip

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts"
RAW = ROOT / "data" / "raw"
SEED = 42

#: training trips used to pick decoder parameters, chosen for variety of
#: length and network density rather than for score
TUNING_TRIPS = ["00000001", "00000005", "00000012", "00000023", "00000037",
                "00000048", "00000060", "00000075"]

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def load_best_model(model_name: str | None):
    """Prefer the Bayesian-tuned artifact, fall back to the stage-3 winner."""
    models_dir = ARTIFACTS / "models"
    if model_name is None:
        summary = RESULTS / "bayesopt_best_params.json"
        if summary.exists():
            model_name = json.loads(summary.read_text())["model"]
        else:
            comp = pd.read_csv(RESULTS / "model_comparison.csv")
            model_name = comp.sort_values("test_top1_accuracy", ascending=False).iloc[0].model
    tuned = models_dir / f"tuned_{model_name}.joblib"
    path = tuned if tuned.exists() else models_dir / f"{model_name}.joblib"
    print(f"Using model artifact: {path.name}")
    return joblib.load(path), model_name, path.name


def score_trips(matcher, partition: str, trip_ids: list[str], run_hmm=True,
                verbose=True) -> pd.DataFrame:
    rows = []
    for tid in trip_ids:
        trip = load_trip(RAW / partition, tid)
        net = build_network(trip)
        res = matcher.match(trip, net, run_hmm=run_hmm)
        ev = evaluate_against_reference(res, trip, net)
        rows.append({"trip_id": tid, "partition": partition,
                     "n_observations": trip.n_obs,
                     **{k: v for k, v in ev.items() if k != "has_reference"},
                     **res.diagnostics,
                     "viterbi_seconds": res.timings.get("viterbi_s", np.nan),
                     "total_seconds": res.timings.get("total_s", np.nan)})
        if verbose:
            print(f"  {tid}: clf {ev['classifier_top1_accuracy']:.4f} -> "
                  f"hmm {ev['hmm_top1_accuracy']:.4f} "
                  f"({ev['hmm_improvement_pp']:+.2f} pp), continuity "
                  f"{ev['classifier_route_continuity']:.3f} -> "
                  f"{ev['hmm_route_continuity']:.3f}", flush=True)
    return pd.DataFrame(rows)


def tune_decoder(model) -> tuple[HMMConfig, pd.DataFrame]:
    """Grid search over the decoder parameters on training trips only."""
    betas = [4.0, 8.0, 12.0, 20.0, 35.0]
    penalties = [6.0, 12.0, 25.0]
    rows = []
    print(f"Decoder search on {len(TUNING_TRIPS)} training trips "
          f"({len(betas) * len(penalties)} configurations)")

    cache = {}
    for tid in TUNING_TRIPS:
        trip = load_trip(RAW / "train", tid)
        cache[tid] = (trip, build_network(trip))

    for beta, pen in product(betas, penalties):
        cfg = HMMConfig(beta=beta, unreachable_penalty=pen)
        matcher = PathSnapMatcher(model, cfg)
        accs, conts, secs = [], [], []
        for tid, (trip, net) in cache.items():
            t0 = time.time()
            res = matcher.match(trip, net, run_hmm=True)
            ev = evaluate_against_reference(res, trip, net)
            accs.append(ev["hmm_top1_accuracy"])
            conts.append(ev["hmm_route_continuity"])
            secs.append(time.time() - t0)
        rows.append({"beta": beta, "unreachable_penalty": pen,
                     "mean_top1_accuracy": float(np.mean(accs)),
                     "mean_route_continuity": float(np.mean(conts)),
                     "mean_seconds_per_trip": float(np.mean(secs))})
        print(f"  beta={beta:5.1f} penalty={pen:5.1f} -> top-1 "
              f"{np.mean(accs):.4f}, continuity {np.mean(conts):.4f}", flush=True)

    grid = pd.DataFrame(rows).sort_values("mean_top1_accuracy", ascending=False)
    best = grid.iloc[0]
    return HMMConfig(beta=float(best.beta),
                     unreachable_penalty=float(best.unreachable_penalty)), grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model, model_name, artifact_name = load_best_model(args.model)

    if args.skip_tuning and (RESULTS / "hmm_config.json").exists():
        cfg = HMMConfig(**json.loads((RESULTS / "hmm_config.json").read_text())["config"])
        grid = pd.read_csv(RESULTS / "hmm_parameter_search.csv")
    else:
        cfg, grid = tune_decoder(model)
        grid.to_csv(RESULTS / "hmm_parameter_search.csv", index=False)

    print(f"\nSelected decoder config: beta={cfg.beta}, "
          f"unreachable_penalty={cfg.unreachable_penalty}")
    with open(RESULTS / "hmm_config.json", "w") as f:
        json.dump({"model": model_name, "model_artifact": artifact_name,
                   "config": cfg.to_dict(),
                   "tuning_trips": TUNING_TRIPS,
                   "note": "decoder parameters selected on training trips only"},
                  f, indent=2)

    matcher = PathSnapMatcher(model, cfg)
    print("\n=== student-testing trips: classifier-only vs classifier + HMM ===")
    per_trip = score_trips(matcher, "student_test", list_trips(RAW / "student_test"))
    per_trip.to_csv(RESULTS / "hmm_comparison_per_trip.csv", index=False)

    w = per_trip.n_observations
    summary = pd.DataFrame([
        {"stage": "classifier_only",
         "top1_accuracy_macro": per_trip.classifier_top1_accuracy.mean(),
         "top1_accuracy_weighted": float(np.average(per_trip.classifier_top1_accuracy, weights=w)),
         "route_continuity": per_trip.classifier_route_continuity.mean(),
         "arc_set_precision": per_trip.classifier_arc_set_precision.mean(),
         "arc_set_recall": per_trip.classifier_arc_set_recall.mean(),
         "arc_set_jaccard": per_trip.classifier_arc_set_jaccard.mean()},
        {"stage": "classifier_plus_hmm",
         "top1_accuracy_macro": per_trip.hmm_top1_accuracy.mean(),
         "top1_accuracy_weighted": float(np.average(per_trip.hmm_top1_accuracy, weights=w)),
         "route_continuity": per_trip.hmm_route_continuity.mean(),
         "arc_set_precision": per_trip.hmm_arc_set_precision.mean(),
         "arc_set_recall": per_trip.hmm_arc_set_recall.mean(),
         "arc_set_jaccard": per_trip.hmm_arc_set_jaccard.mean()},
    ])
    summary.to_csv(RESULTS / "hmm_comparison_summary.csv", index=False)

    make_plots(per_trip, grid)

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    helped = int((per_trip.hmm_improvement_pp > 0).sum())
    hurt = int((per_trip.hmm_improvement_pp < 0).sum())
    print(f"\nHMM decoding improved {helped} trips, degraded {hurt}, "
          f"unchanged {len(per_trip) - helped - hurt}")
    print(f"Mean Viterbi time: {per_trip.viterbi_seconds.mean():.2f}s per trip")


def make_plots(per_trip: pd.DataFrame, grid: pd.DataFrame) -> None:
    idx = np.arange(len(per_trip)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(idx - w / 2, per_trip.classifier_top1_accuracy, w,
              label="classifier only", color="#f29900")
    ax[0].bar(idx + w / 2, per_trip.hmm_top1_accuracy, w,
              label="classifier + HMM", color="#d93025")
    ax[0].set(xticks=idx, xticklabels=per_trip.trip_id, ylabel="top-1 accuracy",
              title="Top-1 accuracy per student-testing trip", ylim=(0, 1.02))
    ax[0].tick_params(axis="x", rotation=60, labelsize=7)
    ax[0].legend()

    ax[1].bar(idx - w / 2, per_trip.classifier_route_continuity, w,
              label="classifier only", color="#f29900")
    ax[1].bar(idx + w / 2, per_trip.hmm_route_continuity, w,
              label="classifier + HMM", color="#d93025")
    ax[1].set(xticks=idx, xticklabels=per_trip.trip_id, ylabel="route continuity",
              title="Route continuity per student-testing trip", ylim=(0, 1.02))
    ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "hmm_per_trip_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = np.where(per_trip.hmm_improvement_pp >= 0, "#17a558", "#C44E52")
    ax.bar(per_trip.trip_id, per_trip.hmm_improvement_pp, color=colours)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(ylabel="top-1 accuracy change (percentage points)",
           title="Effect of Viterbi decoding, per trip")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "hmm_improvement.png"); plt.close(fig)

    piv = grid.pivot(index="beta", columns="unreachable_penalty",
                     values="mean_top1_accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(piv.columns)), xticklabels=piv.columns,
           yticks=range(len(piv.index)), yticklabels=piv.index,
           xlabel="unreachable-transition penalty", ylabel="beta (metres)",
           title="Decoder parameter search (training trips)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                    color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean top-1 accuracy")
    fig.tight_layout(); fig.savefig(FIGS / "hmm_parameter_search.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(per_trip.classifier_route_continuity, per_trip.hmm_route_continuity,
               s=50, color="#4C72B0")
    lim = [min(per_trip.classifier_route_continuity.min(), 0.8), 1.005]
    ax.plot(lim, lim, "k--", lw=0.8)
    for _, r in per_trip.iterrows():
        ax.annotate(r.trip_id[-3:], (r.classifier_route_continuity,
                                     r.hmm_route_continuity),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlim=lim, ylim=lim, xlabel="continuity, classifier only",
           ylabel="continuity, classifier + HMM",
           title="Route continuity before and after decoding")
    fig.tight_layout(); fig.savefig(FIGS / "hmm_continuity_scatter.png"); plt.close(fig)
    print(f"  figures -> {FIGS}")


if __name__ == "__main__":
    main()
