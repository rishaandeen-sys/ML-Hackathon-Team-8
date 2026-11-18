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
    expanded_arcs: np.ndarray | None = None   # route with intermediate arcs filled in
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    network: object = None
    trip: object = None

    def route_arc_sequence(self, source: str = "expanded") -> np.ndarray:
        """Collapsed arc sequence (consecutive duplicates removed).

        "expanded" is the connected route: the decoded sequence with the
        intermediate arcs filled in.  "hmm" and "classifier" are the raw
        per-observation decisions.
        """
        if source == "expanded" and self.expanded_arcs is not None:
            return np.asarray(self.expanded_arcs)
        seq = self.hmm_arcs if source in ("hmm", "expanded") else self.classifier_arcs
        seq = np.asarray(seq)
        seq = seq[seq >= 0]
        if len(seq) == 0:
            return seq
        return seq[np.concatenate([[True], seq[1:] != seq[:-1]])]

    def route_polyline(self, source: str = "expanded") -> list[list[float]]:
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


def expand_to_connected_route(arc_seq, network, max_fill_m: float = 2000.0):
    """Fill the gaps in a decoded arc sequence with shortest paths.

    At 1 Hz a vehicle can cross a short arc entirely between two fixes, so two
    consecutive decoded arcs are often connected only *through* an arc that no
    observation ever landed on.  Inserting those arcs turns a sequence of
    per-observation decisions into a route that is actually drivable.

    Returns (expanded arc sequence, number of gaps that could not be bridged).
    """
    arcs = np.asarray(arc_seq)
    arcs = arcs[arcs >= 0]
    if len(arcs) == 0:
        return np.empty(0, dtype=np.int64), 0
    arcs = arcs[np.concatenate([[True], arcs[1:] != arcs[:-1]])]

    out = [int(arcs[0])]
    unbridged = 0
    for nxt in arcs[1:]:
        nxt = int(nxt)
        head = int(network.arcs[out[-1], 1])
        tail = int(network.arcs[nxt, 0])
        if head != tail:
            path = network.shortest_path_arcs(head, tail, max_fill_m)
            if path is None:
                unbridged += 1
            else:
                out.extend(int(a) for a in path)
        out.append(nxt)

    out = np.asarray(out, dtype=np.int64)
    return out[np.concatenate([[True], out[1:] != out[:-1]])], unbridged


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
                               np.zeros(T), np.zeros(T, dtype=np.int64), None,
                               timings, {"no_candidates": True}, network, trip)

        t0 = time.perf_counter()
        # the models are fitted on plain arrays, so pass one (a DataFrame here
        # triggers a feature-name mismatch warning and buys nothing)
        scores = self.model.predict_proba(feats)[:, 1]
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

            t0 = time.perf_counter()
            expanded, unbridged = expand_to_connected_route(hmm_arcs, network)
            timings["expand_s"] = time.perf_counter() - t0
            diagnostics.update({
                "expanded_route_arcs": int(len(expanded)),
                "expanded_unbridged_gaps": int(unbridged),
                "expanded_route_continuity": route_continuity(expanded, network),
            })

        timings["total_s"] = sum(v for k, v in timings.items() if k.endswith("_s"))
        return MatchResult(trip.trip_id, T, clf_arcs, hmm_arcs, clf_scores,
                           counts, expanded if run_hmm else None,
                           timings, diagnostics, network, trip)


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
    if result.expanded_arcs is not None and len(result.expanded_arcs):
        out["expanded_route_arcs"] = int(len(result.expanded_arcs))
        out["expanded_route_continuity"] = route_continuity(result.expanded_arcs,
                                                            network)
        ref = set(int(a) for a in trip.route)
        pred = set(int(a) for a in result.expanded_arcs)
        inter = len(ref & pred)
        out["expanded_arc_set_precision"] = inter / len(pred) if pred else float("nan")
        out["expanded_arc_set_recall"] = inter / len(ref) if ref else float("nan")
    out["hmm_improvement_pp"] = (out["hmm_top1_accuracy"] -
                                 out["classifier_top1_accuracy"]) * 100
    return out
