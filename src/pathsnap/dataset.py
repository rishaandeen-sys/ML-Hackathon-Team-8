"""Turn raw trips into the (observation, candidate) table used for training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import (DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M,
                         FEATURE_NAMES, build_candidates)
from .labeling import align_observations_to_route
from .network import build_network
from .trips import iter_trips, list_trips, load_trip

#: An observation is only treated as reliably labelled when the reference arc
#: the alignment picked is within this distance of the GPS fix.  The training
#: residuals are sharply bimodal -- 75 % of observations sit within 11 m, but a
#: tail beyond ~90 m comes from trips whose supplied .route covers only part of
#: the journey (trips 00000033, 00000054 and 00000002 cover 13-19 % of their
#: track).  For those stretches there is no reference arc to learn from, so the
#: rows are kept in the table, flagged, and excluded from fitting and scoring.
LABEL_RESIDUAL_MAX_M = 30.0


def build_trip_frame(trip, radius_m=DEFAULT_RADIUS_M,
                     max_candidates=DEFAULT_MAX_CANDIDATES,
                     with_labels=True) -> pd.DataFrame:
    """One DataFrame per trip: a row per (observation, candidate arc) pair."""
    network = build_network(trip)
    obs_idx, arc_id, feats = build_candidates(trip, network, radius_m, max_candidates)

    df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(1, "obs_idx", obs_idx)
    df.insert(2, "arc_id", arc_id)

    if with_labels and trip.route is not None:
        obs_xy = network.projection.to_xy(trip.track[:, 0], trip.track[:, 1])
        gt_arc, residual, _ = align_observations_to_route(obs_xy, network, trip.route)
        df["gt_arc_id"] = gt_arc[obs_idx]
        df["gt_residual_m"] = residual[obs_idx]
        df["label"] = (df["arc_id"].to_numpy() == gt_arc[obs_idx]).astype(np.int8)
        df["label_valid"] = (df["gt_residual_m"] <= LABEL_RESIDUAL_MAX_M).astype(np.int8)
    return df


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose derived label is trustworthy (see LABEL_RESIDUAL_MAX_M)."""
    return df[df["label_valid"] == 1] if "label_valid" in df.columns else df


def build_partition(partition_dir, out_path=None, trip_ids=None,
                    radius_m=DEFAULT_RADIUS_M,
                    max_candidates=DEFAULT_MAX_CANDIDATES,
                    with_labels=True, verbose=True) -> pd.DataFrame:
    frames = []
    ids = trip_ids if trip_ids is not None else list_trips(partition_dir)
    for i, tid in enumerate(ids, 1):
        trip = load_trip(partition_dir, tid)
        frames.append(build_trip_frame(trip, radius_m, max_candidates, with_labels))
        if verbose:
            print(f"  [{i:3d}/{len(ids)}] {tid}  obs={trip.n_obs:5d}  "
                  f"rows={len(frames[-1]):6d}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        if verbose:
            print(f"  wrote {out_path}  ({len(out):,} rows)")
    return out


def candidate_miss_rate(df: pd.DataFrame, only_valid: bool = True) -> float:
    """Fraction of observations whose reference arc is not in the candidate set.

    This is the ceiling on Top-1 accuracy: an observation whose true arc was
    never retrieved cannot be classified correctly no matter how good the model.
    """
    sub = labelled(df) if only_valid else df
    if sub.empty:
        return float("nan")
    per_obs = sub.groupby(["trip_id", "obs_idx"])["label"].max()
    return float((per_obs == 0).mean())
