"""HMM formulation of map-matching and Viterbi decoding.

Hidden states
    At observation t the hidden state is "which directed arc the vehicle was
    actually on". The state space is the candidate set generated for that
    observation, so it changes from step to step: a lattice, not a fixed
    alphabet.

Emission probabilities
    The classifier score for the (observation, candidate) pair, normalised
    over the candidates of that observation. This is where the supervised
    model enters. The HMM does not re-learn the geometry, it only enforces
    that the sequence of choices is physically coherent.

Transition probabilities
    Following Newson and Krumm (2009). For a move from candidate i at time t
    to candidate j at time t+1, compare the on-road travel distance

        d_route = (1 - t_i) * len_i + shortest_path(head_i, tail_j) + t_j * len_j

    with the straight-line distance d_gps between the two fixes. A real move
    has d_route close to d_gps; teleporting between unconnected streets does
    not. The mismatch is modelled as exponential:

        P(j | i) proportional to exp(-|d_route - d_gps| / beta)

    Pairs with no path inside the search radius are not forbidden outright,
    they take a fixed penalty, so a thin candidate set or a GPS dropout can
    never make the whole lattice dead-end.

Gaps
    When consecutive fixes are far apart in time the search radius grows with
    the gap and beta is relaxed, which lets the decoder bridge the hole
    instead of forcing an implausible local jump.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HMMConfig:
    beta: float = 12.0                  # scale on the route/GPS mismatch, metres
    emission_weight: float = 1.0        # weight of the classifier term
    transition_weight: float = 1.0      # weight of the connectivity term
    unreachable_penalty: float = 12.0   # -log penalty when no path is found
    same_arc_bonus: float = 0.0         # optional stickiness for staying put
    max_search_factor: float = 4.0      # radius = factor * d_gps + margin
    max_search_margin_m: float = 300.0
    gap_time_s: float = 5.0             # above this dt the step counts as a gap
    gap_beta_multiplier: float = 4.0
    min_emission_prob: float = 1e-9

    def to_dict(self):
        return asdict(self)


def _normalise_emissions(scores: np.ndarray, floor: float) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), floor, None)
    total = s.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(s), 1.0 / len(s))
    return s / total


def _projection_param(network, arc_ids, point_xy):
    """Where along each arc the observation projects, 0..1."""
    from .geo import point_segment_distance
    _, t = point_segment_distance(
        point_xy[0], point_xy[1],
        network.arc_a[arc_ids, 0], network.arc_a[arc_ids, 1],
        network.arc_b[arc_ids, 0], network.arc_b[arc_ids, 1])
    return t


def viterbi_decode(candidates, scores, obs_xy, timestamps, network,
                   config: HMMConfig | None = None):
    """Decode the most likely sequence of arcs.

    Parameters
    ----------
    candidates : list of int arrays, candidate arc ids per observation
    scores     : list of float arrays, classifier score per candidate
    obs_xy     : (T, 2) projected observation coordinates in metres
    timestamps : (T,) seconds
    network    : RoadNetwork for this trip

    Returns a dict with the decoded arc per observation (-1 where an
    observation had no candidates), the path log probability and diagnostics.
    """
    cfg = config or HMMConfig()
    T = len(candidates)
    empty = {"arcs": np.full(T, -1, dtype=np.int64), "log_prob": float("-inf"),
             "n_unreachable": 0, "n_gaps": 0, "n_decoded": 0}
    if T == 0:
        return empty

    active = [t for t in range(T) if len(candidates[t]) > 0]
    decoded = np.full(T, -1, dtype=np.int64)
    if not active:
        return empty

    log_em = [np.log(_normalise_emissions(scores[t], cfg.min_emission_prob))
              if len(candidates[t]) else np.empty(0) for t in range(T)]

    delta = cfg.emission_weight * log_em[active[0]]
    backptr: list[np.ndarray] = []
    n_unreachable = 0
    n_gaps = 0

    for k in range(1, len(active)):
        t_prev, t_cur = active[k - 1], active[k]
        prev_arcs, cur_arcs = candidates[t_prev], candidates[t_cur]
        d_gps = float(np.hypot(*(obs_xy[t_cur] - obs_xy[t_prev])))
        dt = float(timestamps[t_cur] - timestamps[t_prev])

        is_gap = dt > cfg.gap_time_s
        n_gaps += int(is_gap)
        beta = cfg.beta * (cfg.gap_beta_multiplier if is_gap else 1.0)
        limit = cfg.max_search_factor * d_gps + cfg.max_search_margin_m
        if is_gap:
            limit *= cfg.gap_beta_multiplier

        t_prev_proj = _projection_param(network, prev_arcs, obs_xy[t_prev])
        t_cur_proj = _projection_param(network, cur_arcs, obs_xy[t_cur])

        head_nodes = network.arcs[prev_arcs, 1]
        tail_nodes = network.arcs[cur_arcs, 0]
        target_set = {int(n) for n in tail_nodes}

        sp_cache: dict[int, dict[int, float]] = {}
        for h in {int(n) for n in head_nodes}:
            sp_cache[h] = network.shortest_path_lengths(h, target_set, limit)

        trans = np.empty((len(prev_arcs), len(cur_arcs)), dtype=np.float64)
        for i in range(len(prev_arcs)):
            ai = int(prev_arcs[i])
            ti = float(t_prev_proj[i])
            rem_i = (1.0 - ti) * network.arc_length[ai]
            reach = sp_cache[int(head_nodes[i])]
            for j in range(len(cur_arcs)):
                aj = int(cur_arcs[j])
                tj = float(t_cur_proj[j])
                if ai == aj:
                    d_route = abs(tj - ti) * network.arc_length[ai]
                    trans[i, j] = -abs(d_route - d_gps) / beta + cfg.same_arc_bonus
                    continue
                sp = reach.get(int(tail_nodes[j]))
                if sp is None:
                    trans[i, j] = -cfg.unreachable_penalty
                    n_unreachable += 1
                else:
                    d_route = rem_i + sp + tj * network.arc_length[aj]
                    trans[i, j] = -abs(d_route - d_gps) / beta

        total = delta[:, None] + cfg.transition_weight * trans
        best_prev = np.argmax(total, axis=0)
        delta = total[best_prev, np.arange(len(cur_arcs))] + \
            cfg.emission_weight * log_em[t_cur]
        backptr.append(best_prev)

    j = int(np.argmax(delta))
    log_prob = float(delta[j])
    decoded[active[-1]] = candidates[active[-1]][j]
    for k in range(len(active) - 1, 0, -1):
        j = int(backptr[k - 1][j])
        decoded[active[k - 1]] = candidates[active[k - 1]][j]

    return {"arcs": decoded, "log_prob": log_prob,
            "n_unreachable": int(n_unreachable), "n_gaps": int(n_gaps),
            "n_decoded": int(np.sum(decoded >= 0))}


def group_candidates(df, score_col: str = "score", n_obs: int | None = None):
    """Split a scored candidate frame for one trip into per-observation lists."""
    if n_obs is None:
        n_obs = int(df["obs_idx"].max()) + 1
    cands = [np.empty(0, dtype=np.int64) for _ in range(n_obs)]
    scrs = [np.empty(0, dtype=np.float64) for _ in range(n_obs)]
    for obs, grp in df.groupby("obs_idx", sort=False):
        cands[int(obs)] = grp["arc_id"].to_numpy(dtype=np.int64)
        scrs[int(obs)] = grp[score_col].to_numpy(dtype=np.float64)
    return cands, scrs
