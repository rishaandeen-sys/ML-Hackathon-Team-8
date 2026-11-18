"""Candidate generation and per-candidate feature engineering.

Map-matching is cast as a *candidate ranking* problem.  For every GPS
observation we retrieve the nearby directed arcs and build one row per
(observation, candidate arc) pair.  A binary classifier then scores each pair;
the per-observation scores are normalised into a distribution over candidates,
which is exactly the emission model the HMM stage needs.

Nothing in here touches the reference route -- labels are attached separately
in `dataset.py` -- so the same code path runs unchanged at inference time.
"""
from __future__ import annotations

import numpy as np

from .geo import angle_difference

#: ordered feature names; the model and the API both rely on this order
FEATURE_NAMES = [
    "dist_m",                 # point-to-segment distance to the candidate arc
    "dist_rank",              # rank of this candidate by distance (0 = nearest)
    "dist_minus_best",        # metres worse than the nearest candidate
    "dist_ratio_best",        # dist / (best dist + 1 m)
    "dist_z_in_obs",          # standardised within the observation's candidate set
    "n_candidates",           # how crowded this part of the network is
    "arc_length",             # geometry of the candidate arc
    "proj_t",                 # normalised foot-of-perpendicular along the arc
    "proj_is_endpoint",       # 1 when the foot lands on a vertex (t = 0 or 1)
    "heading_diff",           # |GPS heading - arc bearing|, radians
    "heading_cos",            # cos of the same, signed: separates the two directions
    "heading_valid",          # 0 when the vehicle is too slow for a reliable heading
    "speed_mps",              # local speed estimate
    "accel_mps2",             # local acceleration
    "dt_prev",                # sampling irregularity
    "dt_next",
    "step_prev_m",            # distance to neighbouring observations
    "step_next_m",
    "dist_prev_obs_to_arc",   # does this arc also explain the previous fix?
    "dist_next_obs_to_arc",   # ... and the next one?
    "arc_out_degree",         # network context at the arc's head
    "arc_in_degree",          # ... and its tail
    "has_reverse_arc",        # 1 if the opposite direction exists (two-way street)
    "local_arc_density",      # arcs within the search radius
    "obs_frac",               # position within the trip, 0..1
    "turn_angle",             # heading change across the observation
]

DEFAULT_RADIUS_M = 60.0
DEFAULT_MAX_CANDIDATES = 10


def _reverse_arc_lookup(network):
    """Map arc -> 1 when the reversed (to, from) arc also exists."""
    key = network.arcs[:, 0].astype(np.int64) * network.n_nodes + network.arcs[:, 1]
    rev = network.arcs[:, 1].astype(np.int64) * network.n_nodes + network.arcs[:, 0]
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rev)
    pos = np.clip(pos, 0, len(key) - 1)
    return (key[order][pos] == rev).astype(np.float32)


def build_candidates(trip, network, radius_m: float = DEFAULT_RADIUS_M,
                     max_candidates: int = DEFAULT_MAX_CANDIDATES):
    """Return (obs_index, arc_id, feature_matrix) for one trip."""
    obs_xy = network.projection.to_xy(trip.track[:, 0], trip.track[:, 1])
    ts = trip.track[:, 2]
    T = len(obs_xy)

    # ---- per-observation kinematics (shared by every candidate of that fix)
    dt_prev = np.empty(T); dt_next = np.empty(T)
    dt_prev[0] = np.nan; dt_prev[1:] = np.diff(ts)
    dt_next[-1] = np.nan; dt_next[:-1] = np.diff(ts)

    step = np.hypot(*(np.diff(obs_xy, axis=0).T))     # (T-1,)
    step_prev = np.concatenate([[np.nan], step])
    step_next = np.concatenate([step, [np.nan]])

    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.nansum(np.stack([step_prev, step_next]), axis=0) / \
            np.nansum(np.stack([dt_prev, dt_next]), axis=0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0)
    accel = np.gradient(speed) if T > 1 else np.zeros(T)

    # heading from the smoothed neighbour-to-neighbour vector
    nxt = np.vstack([obs_xy[1:], obs_xy[-1]])
    prv = np.vstack([obs_xy[0], obs_xy[:-1]])
    hvec = nxt - prv
    hlen = np.hypot(hvec[:, 0], hvec[:, 1])
    heading = np.arctan2(hvec[:, 0], hvec[:, 1])
    heading_valid = (hlen > 2.0).astype(np.float32)   # < 2 m of travel: heading is noise

    turn = np.zeros(T)
    if T > 2:
        turn[1:-1] = angle_difference(heading[2:], heading[:-2])

    rev_lookup = _reverse_arc_lookup(network)
    obs_frac = np.arange(T) / max(T - 1, 1)

    out_obs, out_arc, out_feat = [], [], []
    for t in range(T):
        x, y = obs_xy[t]
        near = network.arcs_near(x, y, radius_m)
        if near.size == 0:                    # widen once for sparse areas
            near = network.arcs_near(x, y, radius_m * 4)
            if near.size == 0:
                continue
        dist, proj_t = network.distance_to_arcs(x, y, near)
        n_local = near.size
        if n_local > max_candidates:
            keep = np.argpartition(dist, max_candidates - 1)[:max_candidates]
            near, dist, proj_t = near[keep], dist[keep], proj_t[keep]
        order = np.argsort(dist, kind="stable")
        near, dist, proj_t = near[order], dist[order], proj_t[order]
        k = len(near)

        best = dist[0]
        std = dist.std() if k > 1 else 1.0
        std = std if std > 1e-6 else 1.0

        dprev = (network.distance_to_arcs(*obs_xy[t - 1], near)[0]
                 if t > 0 else np.full(k, np.nan))
        dnext = (network.distance_to_arcs(*obs_xy[t + 1], near)[0]
                 if t < T - 1 else np.full(k, np.nan))

        hd = angle_difference(np.full(k, heading[t]), network.arc_bearing[near])
        feat = np.column_stack([
            dist,
            np.arange(k, dtype=np.float64),
            dist - best,
            dist / (best + 1.0),
            (dist - dist.mean()) / std,
            np.full(k, float(k)),
            network.arc_length[near],
            proj_t,
            ((proj_t <= 1e-9) | (proj_t >= 1 - 1e-9)).astype(np.float64),
            hd,
            np.cos(np.full(k, heading[t]) - network.arc_bearing[near]),
            np.full(k, heading_valid[t]),
            np.full(k, speed[t]),
            np.full(k, accel[t]),
            np.full(k, dt_prev[t]),
            np.full(k, dt_next[t]),
            np.full(k, step_prev[t]),
            np.full(k, step_next[t]),
            dprev,
            dnext,
            network.out_degree[network.arcs[near, 1]].astype(np.float64),
            network.in_degree[network.arcs[near, 0]].astype(np.float64),
            rev_lookup[near],
            np.full(k, float(n_local)),
            np.full(k, obs_frac[t]),
            np.full(k, turn[t]),
        ])
        out_obs.append(np.full(k, t, dtype=np.int64))
        out_arc.append(near.astype(np.int64))
        out_feat.append(feat)

    if not out_obs:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty((0, len(FEATURE_NAMES)), np.float64))

    return (np.concatenate(out_obs), np.concatenate(out_arc),
            np.vstack(out_feat).astype(np.float64))
