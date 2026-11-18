"""Stage 5c: search the directional-consistency terms added to the decoder.

Stage 5b showed that tuning beta and the unreachable penalty is saturated -
top-1 varied only between 0.9519 and 0.9530 across the whole grid. A flat
response surface means the remaining error is not a parameter problem, so
this stage tests a structural change to the transition model instead.

Three new terms, all off by default so earlier results stay reproducible:

  uturn_penalty
      The reverse of the current arc has *identical geometry*, therefore an
      identical d_route. The distance term is mathematically incapable of
      telling the two directions of a two-way street apart, so without this
      the decoder may flip direction between consecutive fixes for free.

  turn_penalty_weight
      Penalises bearing change beyond turn_free_angle. A vehicle at speed
      does not take sharp turns between 1 Hz fixes.

  direction_consistency_weight
      Rewards candidates whose bearing agrees with the heading implied by
      the GPS movement over the step.

Selection is on training trips only, as in stages 5 and 5b.

Run:  python scripts/05c_hmm_directional_search.py
Out:  results/hmm_directional_search.csv
      results/hmm_config.json   (overwritten when a better config is found)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from pathsnap.hmm import HMMConfig
from pathsnap.network import build_network
from pathsnap.pipeline import PathSnapMatcher, evaluate_against_reference
from pathsnap.trips import load_trip

RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"


def main() -> None:
    cfg_json = json.loads((RESULTS / "hmm_config.json").read_text())
    base = dict(cfg_json["config"])
    model = joblib.load(ROOT / "artifacts" / "models" / cfg_json["model_artifact"])
    tuning_trips = cfg_json["tuning_trips"]

    print(f"Model     : {cfg_json['model_artifact']}")
    print(f"Base cfg  : beta={base['beta']}, penalty={base['unreachable_penalty']}, "
          f"same_arc_bonus={base.get('same_arc_bonus', 0.0)}")
    print(f"Trips     : {', '.join(tuning_trips)}\n")

    cache = {}
    for tid in tuning_trips:
        trip = load_trip(RAW / "train", tid)
        cache[tid] = (trip, build_network(trip))

    def run(**overrides):
        cfg = HMMConfig(**{**base, **overrides})
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

    rows = []
    baseline = run()
    rows.append({**baseline, "phase": "baseline"})
    print(f"Baseline: top-1 {baseline['mean_top1_accuracy']:.4f}, "
          f"continuity {baseline['mean_route_continuity']:.4f}\n")

    print("Phase 1: U-turn penalty")
    for v in [0.0, 1.0, 2.0, 4.0, 8.0, 15.0]:
        r = run(uturn_penalty=v)
        rows.append({**r, "phase": "uturn"})
        print(f"  uturn_penalty={v:5.1f} -> top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    best_u = best.get("uturn_penalty", 0.0)

    print(f"\nPhase 2: turn penalty (with uturn_penalty={best_u})")
    for v in [0.0, 0.5, 1.0, 2.0, 4.0]:
        r = run(uturn_penalty=best_u, turn_penalty_weight=v)
        rows.append({**r, "phase": "turn"})
        print(f"  turn_penalty_weight={v:4.1f} -> top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    best_t = best.get("turn_penalty_weight", 0.0)

    print(f"\nPhase 3: GPS direction consistency "
          f"(uturn={best_u}, turn={best_t})")
    for v in [0.0, 0.5, 1.0, 2.0, 4.0]:
        r = run(uturn_penalty=best_u, turn_penalty_weight=best_t,
                direction_consistency_weight=v)
        rows.append({**r, "phase": "direction"})
        print(f"  direction_weight={v:4.1f} -> top-1 {r['mean_top1_accuracy']:.4f}, "
              f"continuity {r['mean_route_continuity']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hmm_directional_search.csv", index=False)

    best = max(rows, key=lambda r: r["mean_top1_accuracy"])
    gain = (best["mean_top1_accuracy"] - baseline["mean_top1_accuracy"]) * 100
    print(f"\n{'=' * 64}")
    print(f"Best : uturn={best.get('uturn_penalty', 0.0)}, "
          f"turn={best.get('turn_penalty_weight', 0.0)}, "
          f"direction={best.get('direction_consistency_weight', 0.0)}")
    print(f"Top-1      : {baseline['mean_top1_accuracy']:.4f} -> "
          f"{best['mean_top1_accuracy']:.4f}  ({gain:+.2f} pp on tuning trips)")
    print(f"Continuity : {baseline['mean_route_continuity']:.4f} -> "
          f"{best['mean_route_continuity']:.4f}")

    if gain > 0.001:
        for k in ("uturn_penalty", "turn_penalty_weight",
                  "direction_consistency_weight"):
            base[k] = float(best.get(k, 0.0) or 0.0)
        cfg_json["config"] = base
        cfg_json["note"] = ("decoder parameters selected on training trips only; "
                            "directional terms from scripts/05c")
        (RESULTS / "hmm_config.json").write_text(json.dumps(cfg_json, indent=2))
        print("\nhmm_config.json updated with the directional terms.")
    else:
        print("\nNo improvement - directional terms left off, config unchanged.")
        print("That is itself a finding: with directed arcs already in the "
              "candidate set and heading already a classifier feature, the "
              "emission model may be resolving direction before the decoder "
              "ever sees it.")


if __name__ == "__main__":
    main()
