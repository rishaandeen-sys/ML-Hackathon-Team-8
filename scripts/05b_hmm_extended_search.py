"""Stage 5b: extended decoder search past the stage-5 grid boundary.

The stage-5 grid selected beta = 4.0 and unreachable_penalty = 25.0, which
were respectively the *lowest* beta and the *highest* penalty it tested.
When the optimum lands on the edge of a grid, the grid is the binding
constraint rather than the data, so this stage searches outward:

    beta     down to 1.0   (stage 5 floor was 4.0)
    penalty  up to 80.0    (stage 5 ceiling was 25.0)

It also searches two parameters stage 5 held fixed:

    same_arc_bonus    stickiness for staying on the current arc, which
                      directly trades pointwise accuracy against the number
                      of spurious arc switches
    max_search_factor how far the shortest-path search may look relative to
                      the straight-line gap between fixes

Selection is on training trips only, exactly as in stage 5, so the
student-testing numbers stay honest.

Run:  python scripts/05b_hmm_extended_search.py [--stage sweep|refine]
Out:  results/hmm_extended_search.csv
      results/hmm_config.json   (overwritten when a better config is found)
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
from pathsnap.trips import load_trip

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RAW = ROOT / "data" / "raw"

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def load_setup():
    cfg_json = json.loads((RESULTS / "hmm_config.json").read_text())
    model_path = ROOT / "artifacts" / "models" / cfg_json["model_artifact"]
    model = joblib.load(model_path)
    trips = cfg_json["tuning_trips"]
    print(f"Model          : {model_path.name}")
    print(f"Baseline config: beta={cfg_json['config']['beta']}, "
          f"penalty={cfg_json['config']['unreachable_penalty']}")
    print(f"Tuning trips   : {', '.join(trips)}")
    return model, cfg_json, trips


def evaluate(model, cache, **overrides) -> dict:
    """Mean top-1 and continuity over the tuning trips for one config."""
    cfg = HMMConfig(**overrides)
    matcher = PathSnapMatcher(model, cfg)
    accs, conts = [], []
    t0 = time.time()
    for trip, net in cache.values():
        res = matcher.match(trip, net, run_hmm=True)
        ev = evaluate_against_reference(res, trip, net)
        accs.append(ev["hmm_top1_accuracy"])
        conts.append(ev["hmm_route_continuity"])
    return {"mean_top1_accuracy": float(np.mean(accs)),
            "mean_route_continuity": float(np.mean(conts)),
            "seconds": time.time() - t0, **overrides}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sweep", choices=["sweep", "refine"])
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    model, cfg_json, tuning_trips = load_setup()

    print("\nLoading tuning trips...")
    cache = {}
    for tid in tuning_trips:
        trip = load_trip(RAW / "train", tid)
        cache[tid] = (trip, build_network(trip))

    baseline = evaluate(model, cache, beta=4.0, unreachable_penalty=25.0)
    print(f"\nBaseline (stage 5): top-1 {baseline['mean_top1_accuracy']:.4f}, "
          f"continuity {baseline['mean_route_continuity']:.4f}\n")

    rows = [{**baseline, "phase": "baseline"}]

    # ---- phase 1: push past both grid edges
    betas = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    penalties = [25.0, 40.0, 60.0, 80.0]
    print(f"Phase 1: beta x penalty ({len(betas) * len(penalties)} configs)")
    for beta, pen in product(betas, penalties):
        r = evaluate(model, cache, beta=beta, unreachable_penalty=pen)
        rows.append({**r, "phase": "beta_penalty"})
        print(f"  beta={beta:5.1f} penalty={pen:5.1f} -> "
              f"top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    print(f"\nBest after phase 1: beta={best['beta']}, "
          f"penalty={best['unreachable_penalty']} -> "
          f"top-1 {best['mean_top1_accuracy']:.4f}")

    # ---- phase 2: parameters stage 5 held fixed, around the phase-1 winner
    print("\nPhase 2: same_arc_bonus and max_search_factor")
    for bonus in [0.0, 0.25, 0.5, 1.0, 2.0]:
        r = evaluate(model, cache, beta=best["beta"],
                     unreachable_penalty=best["unreachable_penalty"],
                     same_arc_bonus=bonus)
        rows.append({**r, "phase": "same_arc_bonus"})
        print(f"  same_arc_bonus={bonus:4.2f} -> "
              f"top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    for factor in [3.0, 4.0, 6.0, 8.0]:
        r = evaluate(model, cache, beta=best["beta"],
                     unreachable_penalty=best["unreachable_penalty"],
                     same_arc_bonus=best.get("same_arc_bonus", 0.0),
                     max_search_factor=factor)
        rows.append({**r, "phase": "max_search_factor"})
        print(f"  max_search_factor={factor:4.1f} -> "
              f"top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hmm_extended_search.csv", index=False)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    gain = (best["mean_top1_accuracy"] - baseline["mean_top1_accuracy"]) * 100
    print(f"\n{'=' * 62}")
    print(f"Best config : beta={best['beta']}, "
          f"penalty={best['unreachable_penalty']}, "
          f"same_arc_bonus={best.get('same_arc_bonus', 0.0)}, "
          f"max_search_factor={best.get('max_search_factor', 4.0)}")
    print(f"Top-1       : {baseline['mean_top1_accuracy']:.4f} -> "
          f"{best['mean_top1_accuracy']:.4f}  ({gain:+.2f} pp on tuning trips)")
    print(f"Continuity  : {baseline['mean_route_continuity']:.4f} -> "
          f"{best['mean_route_continuity']:.4f}")

    at_edge = []
    if best["beta"] == min(betas):
        at_edge.append(f"beta is at the search floor ({min(betas)})")
    if best["unreachable_penalty"] == max(penalties):
        at_edge.append(f"penalty is at the search ceiling ({max(penalties)})")
    if at_edge:
        print(f"NOTE: {'; '.join(at_edge)} - the optimum may lie further out.")

    if gain > 0.001:
        new_cfg = dict(cfg_json["config"])
        for k in ("beta", "unreachable_penalty", "same_arc_bonus",
                  "max_search_factor"):
            if k in best and best[k] is not None and not pd.isna(best[k]):
                new_cfg[k] = float(best[k])
        cfg_json["config"] = new_cfg
        cfg_json["note"] = ("decoder parameters selected on training trips only; "
                            "extended search in scripts/05b")
        (RESULTS / "hmm_config.json").write_text(json.dumps(cfg_json, indent=2))
        print(f"\nhmm_config.json updated -> re-run scripts/05_hmm_viterbi.py "
              f"--skip-tuning for test-set numbers")
    else:
        print("\nNo improvement over the stage-5 config; hmm_config.json unchanged.")

    plot(df)


def plot(df: pd.DataFrame) -> None:
    bp = df[df.phase == "beta_penalty"]
    if not bp.empty:
        piv = bp.pivot_table(index="beta", columns="unreachable_penalty",
                             values="mean_top1_accuracy")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(piv.columns)), xticklabels=piv.columns,
               yticks=range(len(piv.index)), yticklabels=piv.index,
               xlabel="unreachable-transition penalty", ylabel="beta (metres)",
               title="Extended decoder search (training trips)")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                            color="w", fontsize=7)
        fig.colorbar(im, ax=ax, label="mean top-1 accuracy")
        fig.tight_layout()
        fig.savefig(FIGS / "hmm_extended_search.png")
        plt.close(fig)
        print(f"  figure -> {FIGS / 'hmm_extended_search.png'}")


if __name__ == "__main__":
    main()
