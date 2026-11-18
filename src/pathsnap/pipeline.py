"""End-to-end inference: raw trip files in, reconstructed route out.

This is the single code path used by stage 5, the map generator and the
FastAPI backend, so what the evaluator sees demonstrated is literally the
code that produced the reported numbers.

The pipeline never reads the reference route.  Scoring against the reference
is a separate call (`evaluate_against_reference`) which the API makes *after*
prediction, server-side, so ground truth can never reach the model or the UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import (DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M,
                         FEATURE_NAMES, build_candidates)
from .hmm import HMMConfig, group_candidates, viterbi_decode
from .labeling import align_observations_to_route
from .metrics import route_continuity, sequence_metrics
from .network import build_network


@dataclass
class MatchResult:
    trip_id: str
    n_observations: int
    classifier_arcs: np.ndarray          # top-1 arc per observation
    hmm_arcs: np.ndarray                 # Viterbi-decoded arc per observation
    scores: np.ndarray                   # top-1 classifier score per observation
    candidates_per_obs: np.ndarray
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    network: object = None
    trip: object = None

    def route_arc_sequence(self, source: str = "hmm") -> np.ndarray:
        """Collapsed arc sequence (consecutive duplicates removed)."""
        seq = self.hmm_arcs if source == "hmm" else self.classifier_arcs
        seq = np.asarray(seq)
        seq = seq[seq >= 0]
        if len(seq) == 0:
            return seq
        return seq[np.concatenate([[True], seq[1:] != seq[:-1]])]

    def route_polyline(self, source: str = "hmm") -> list[list[float]]:
        """[[lat, lon], ...] along the reconstructed route, for the UI."""
        arcs = self.route_arc_sequence(source)
        pts: list[list[float]] = []
        for a in arcs:
            u, v = self.network.arcs[int(a)]
            p1 = [float(self.network.nodes_lonlat[u, 1]), float(self.network.nodes_lonlat[u, 0])]
            p2 = [float(self.network.nodes_lonlat[v, 1]), float(self.network.nodes_lonlat[v, 0])]
            if not pts or pts[-1] != p1:
                pts.append(p1)
            pts.append(p2)
        return pts


class PathSnapMatcher:
    """Classifier + HMM/Viterbi map matcher."""

    def __init__(self, model, hmm_config: HMMConfig | None = None,
                 radius_m: float = DEFAULT_RADIUS_M,
                 max_candidates: int = DEFAULT_MAX_CANDIDATES):
        self.model = model
        self.hmm_config = hmm_config or HMMConfig()
        self.radius_m = radius_m
        self.max_candidates = max_candidates

    def match(self, trip, network=None, run_hmm: bool = True) -> MatchResult:
        timings = {}
        t0 = time.perf_counter()
        network = network or build_network(trip)
        timings["network_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        obs_idx, arc_id, feats = build_candidates(
            trip, network, self.radius_m, self.max_candidates)
        timings["candidates_s"] = time.perf_counter() - t0

        T = trip.n_obs
        if len(obs_idx) == 0:
            empty = np.full(T, -1, dtype=np.int64)
            return MatchResult(trip.trip_id, T, empty, empty.copy(),
                               np.zeros(T), np.zeros(T, dtype=np.int64),
                               timings, {"no_candidates": True}, network, trip)

        t0 = time.perf_counter()
        X = pd.DataFrame(feats, columns=FEATURE_NAMES)
        scores = self.model.predict_proba(X)[:, 1]
        timings["classifier_s"] = time.perf_counter() - t0

        df = pd.DataFrame({"obs_idx": obs_idx, "arc_id": arc_id, "score": scores})

        # classifier-only: highest-scoring candidate per observation
        best_rows = df.loc[df.groupby("obs_idx")["score"].idxmax()]
        clf_arcs = np.full(T, -1, dtype=np.int64)
        clf_scores = np.zeros(T, dtype=np.float64)
        clf_arcs[best_rows.obs_idx.to_numpy()] = best_rows.arc_id.to_numpy()
        clf_scores[best_rows.obs_idx.to_numpy()] = best_rows.score.to_numpy()

        counts = np.zeros(T, dtype=np.int64)
        vc = df.obs_idx.value_counts()
        counts[vc.index.to_numpy()] = vc.to_numpy()

        diagnostics = {
            "observations_without_candidates": int(np.sum(counts == 0)),
            "mean_candidates_per_obs": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
            "classifier_route_continuity": route_continuity(clf_arcs, network),
        }

        hmm_arcs = clf_arcs.copy()
        if run_hmm:
            t0 = time.perf_counter()
            cands, scrs = group_candidates(df, "score", n_obs=T)
            obs_xy = network.projection.to_xy(trip.track[:, 0], trip.track[:, 1])
            decoded = viterbi_decode(cands, scrs, obs_xy, trip.track[:, 2],
                                     network, self.hmm_config)
            timings["viterbi_s"] = time.perf_counter() - t0
            hmm_arcs = decoded["arcs"]
            diagnostics.update({
                "hmm_log_prob": decoded["log_prob"],
                "hmm_unreachable_transitions": decoded["n_unreachable"],
                "hmm_gap_steps": decoded["n_gaps"],
                "hmm_route_continuity": route_continuity(hmm_arcs, network),
            })

        timings["total_s"] = sum(v for k, v in timings.items() if k.endswith("_s"))
        return MatchResult(trip.trip_id, T, clf_arcs, hmm_arcs, clf_scores,
                           counts, timings, diagnostics, network, trip)


def evaluate_against_reference(result: MatchResult, trip, network,
                               label_residual_max_m: float = 30.0) -> dict:
    """Score a MatchResult against the supplied reference route.

    Called only on the server side.  Observations whose aligned reference arc
    is further than `label_residual_max_m` away are excluded, for the same
    reason they are excluded from training: for those stretches the supplied
    route does not describe where the vehicle was.
    """
    if trip.route is None or len(trip.route) == 0:
        return {"has_reference": False}

    obs_xy = network.projection.to_xy(trip.track[:, 0], trip.track[:, 1])
    gt_arc, residual, _ = align_observations_to_route(obs_xy, network, trip.route)
    valid = residual <= label_residual_max_m

    out = {
        "has_reference": True,
        "n_observations": int(len(gt_arc)),
        "n_scored_observations": int(valid.sum()),
        "frac_scored": float(valid.mean()),
        "reference_route_arcs": int(len(trip.route)),
    }
    for name, arcs in (("classifier", result.classifier_arcs), ("hmm", result.hmm_arcs)):
        sel = valid & (arcs >= 0)
        acc = float((arcs[sel] == gt_arc[sel]).mean()) if sel.any() else float("nan")
        seq = sequence_metrics(arcs[valid], gt_arc[valid])
        out[f"{name}_top1_accuracy"] = acc
        out[f"{name}_route_continuity"] = route_continuity(arcs, network)
        out[f"{name}_arc_set_precision"] = seq["arc_set_precision"]
        out[f"{name}_arc_set_recall"] = seq["arc_set_recall"]
        out[f"{name}_arc_set_jaccard"] = seq["arc_set_jaccard"]
    out["hmm_improvement_pp"] = (out["hmm_top1_accuracy"] -
                                 out["classifier_top1_accuracy"]) * 100
    return out
