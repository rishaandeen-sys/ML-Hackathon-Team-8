"""Stage 1: build the (observation, candidate) tables for both partitions.

Run:  python scripts/01_build_dataset.py
Out:  data/processed/train_candidates.parquet
      data/processed/student_test_candidates.parquet
      results/candidate_generation_summary.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from pathsnap.candidates import DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M
from pathsnap.dataset import (LABEL_RESIDUAL_MAX_M, build_partition,
                              candidate_miss_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for partition in ("train", "student_test"):
        print(f"\n=== {partition} ===", flush=True)
        t0 = time.time()
        df = build_partition(Path(args.raw) / partition,
                             out_path=out_dir / f"{partition}_candidates.parquet",
                             radius_m=args.radius,
                             max_candidates=args.max_candidates,
                             with_labels=True)
        elapsed = time.time() - t0
        n_obs = df.groupby(["trip_id", "obs_idx"]).ngroups
        miss = candidate_miss_rate(df, only_valid=True)
        miss_all = candidate_miss_rate(df, only_valid=False)
        valid_obs = df[df.label_valid == 1].groupby(["trip_id", "obs_idx"]).ngroups
        rows.append({
            "partition": partition,
            "n_trips": df["trip_id"].nunique(),
            "n_observations": n_obs,
            "n_candidate_rows": len(df),
            "mean_candidates_per_obs": len(df) / n_obs,
            "n_observations_reliably_labelled": valid_obs,
            "frac_observations_reliably_labelled": valid_obs / n_obs,
            "label_residual_max_m": LABEL_RESIDUAL_MAX_M,
            "candidate_miss_rate": miss,
            "candidate_miss_rate_all_obs": miss_all,
            "positive_rate": float(df["label"].mean()),
            "median_label_residual_m": float(df["gt_residual_m"].median()),
            "build_seconds": elapsed,
            "radius_m": args.radius,
            "max_candidates": args.max_candidates,
        })
        print(f"  {partition}: {len(df):,} rows, miss rate {miss:.4%}, "
              f"{elapsed:.1f}s", flush=True)

    summary = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "results" / "candidate_generation_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
