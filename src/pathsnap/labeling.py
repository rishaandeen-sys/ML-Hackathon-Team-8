"""Deriving a per-observation reference arc from the supplied .route file.

The .route file gives the chronological sequence of arc ids the vehicle
traversed, but *not* which GPS observation belongs to which arc.  The two
sequences have different lengths (e.g. 1315 observations against 525 route
arcs), so the mapping has to be recovered.

Picking the nearest route arc independently for each observation is wrong: GPS
noise makes it jump backwards and forwards along the route, producing labels
that no causal model could reproduce.  Instead we solve a monotonic alignment
with dynamic programming -- the same shape as DTW -- which forces the assigned
route index to be non-decreasing in time:

    f[t, r] = cost[t, r] + min_{r' <= r} f[t-1, r']

The running prefix-minimum makes this O(T x R).  This is a *derived* label and
inherits whatever error the reference route carries; the EDA script reports the
residual distances so the assumption can be judged rather than trusted.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment_distance


def align_observations_to_route(obs_xy: np.ndarray, network, route: np.ndarray,
                                max_route_arcs_chunk: int = 4096):
    """Monotonically assign each observation to one arc of the reference route.

    Returns (gt_arc_id per observation, residual distance in metres,
    assigned index into `route`).
    """
    T = len(obs_xy)
    R = len(route)
    if T == 0 or R == 0:
        return (np.full(T, -1, dtype=np.int64), np.full(T, np.nan),
                np.full(T, -1, dtype=np.int64))

    ax, ay = network.arc_a[route, 0], network.arc_a[route, 1]
    bx, by = network.arc_b[route, 0], network.arc_b[route, 1]

    # cost[t, r] = distance from observation t to route arc r
    cost = np.empty((T, R), dtype=np.float32)
    step = max(1, max_route_arcs_chunk * 1024 // max(R, 1))
    for s in range(0, T, step):
        e = min(T, s + step)
        d, _ = point_segment_distance(
            obs_xy[s:e, 0:1], obs_xy[s:e, 1:2], ax[None, :], ay[None, :],
            bx[None, :], by[None, :])
        cost[s:e] = d.astype(np.float32)

    # forward pass with prefix minimum
    f = np.empty((T, R), dtype=np.float32)
    back = np.empty((T, R), dtype=np.int32)
    f[0] = cost[0]
    back[0] = -1
    idx = np.arange(R, dtype=np.int32)
    for t in range(1, T):
        prev = f[t - 1]
        run_min = np.minimum.accumulate(prev)
        # argmin of the prefix: index where the running minimum last improved
        improved = prev <= run_min
        arg = np.maximum.accumulate(np.where(improved, idx, -1))
        f[t] = cost[t] + run_min
        back[t] = arg

    # backtrack
    assign = np.empty(T, dtype=np.int64)
    assign[T - 1] = int(np.argmin(f[T - 1]))
    for t in range(T - 1, 0, -1):
        assign[t - 1] = int(back[t, assign[t]])

    gt_arc = route[assign]
    residual = cost[np.arange(T), assign].astype(np.float64)
    return gt_arc, residual, assign
