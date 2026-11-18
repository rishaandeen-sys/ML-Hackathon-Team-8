"""Road-network representation, spatial index and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, point_segment_distance


class RoadNetwork:
    """Directed road network for a single trip, projected to local metres.

    Arcs are directed: a two-way street appears as two arcs (u->v and v->u)
    with identical geometry but opposite bearing.  We keep both, because the
    reference route is a sequence of *directed* arc ids and the direction of
    travel is exactly what the heading features are there to resolve.
    """

    #: arcs longer than this (metres) are densified so the spatial index
    #: cannot miss a long arc whose midpoint is far from the query point
    SAMPLE_SPACING_M = 25.0

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray,
                 projection: LocalProjection):
        self.projection = projection
        self.nodes_lonlat = nodes_lonlat
        self.node_xy = projection.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)

        self.arc_a = self.node_xy[arcs[:, 0]]          # (M, 2) start point
        self.arc_b = self.node_xy[arcs[:, 1]]          # (M, 2) end point
        delta = self.arc_b - self.arc_a
        self.arc_length = np.hypot(delta[:, 0], delta[:, 1])
        self.arc_bearing = np.arctan2(delta[:, 0], delta[:, 1])
        self.arc_mid = 0.5 * (self.arc_a + self.arc_b)

        self._build_spatial_index()
        self._build_adjacency()

    # ------------------------------------------------------------------ index
    def _build_spatial_index(self) -> None:
        """KD-tree over points sampled along every arc."""
        n_extra = np.maximum(
            0, np.ceil(self.arc_length / self.SAMPLE_SPACING_M).astype(np.int64) - 1)
        samples, owner = [self.arc_a, self.arc_b], [
            np.arange(self.n_arcs), np.arange(self.n_arcs)]

        if n_extra.max(initial=0) > 0:
            for k in range(1, int(n_extra.max()) + 1):
                sel = np.nonzero(n_extra >= k)[0]
                frac = k / (n_extra[sel] + 1.0)
                samples.append(self.arc_a[sel] + frac[:, None] *
                               (self.arc_b[sel] - self.arc_a[sel]))
                owner.append(sel)

        self._sample_xy = np.vstack(samples)
        self._sample_arc = np.concatenate(owner)
        self._tree = cKDTree(self._sample_xy)

    def arcs_near(self, x: float, y: float, radius_m: float) -> np.ndarray:
        """Unique ids of arcs with any sampled point within `radius_m`."""
        idx = self._tree.query_ball_point([x, y], radius_m)
        if not idx:
            return np.empty(0, dtype=np.int64)
        return np.unique(self._sample_arc[np.asarray(idx, dtype=np.int64)])

    def distance_to_arcs(self, x: float, y: float, arc_ids: np.ndarray):
        """Exact point-to-segment distance (metres) and projection parameter."""
        return point_segment_distance(
            x, y,
            self.arc_a[arc_ids, 0], self.arc_a[arc_ids, 1],
            self.arc_b[arc_ids, 0], self.arc_b[arc_ids, 1])

    # -------------------------------------------------------------- adjacency
    def _build_adjacency(self) -> None:
        """CSR adjacency over nodes, carrying the arc id of every edge."""
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self._adj_arc = order.astype(np.int64)
        self._adj_to = self.arcs[order, 1].astype(np.int64)
        self._adj_w = self.arc_length[order]
        counts = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        self._adj_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        self.out_degree = counts
        self.in_degree = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)

    def successors(self, node: int):
        s, e = self._adj_start[node], self._adj_start[node + 1]
        return self._adj_to[s:e], self._adj_w[s:e], self._adj_arc[s:e]

    def shortest_path_lengths(self, source: int, targets: set[int],
                              limit_m: float) -> dict[int, float]:
        """Bounded Dijkstra from `source`, stopping at `limit_m`.

        Returns only the targets reached within the limit.  Consecutive 1 Hz
        GPS observations are metres apart, so the explored subgraph stays tiny
        and this is far cheaper than any all-pairs precomputation.
        """
        if not targets:
            return {}
        remaining = set(targets)
        found: dict[int, float] = {}
        if source in remaining:
            found[source] = 0.0
            remaining.discard(source)
            if not remaining:
                return found

        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit_m:
                if d > limit_m:
                    break
                continue
            nbrs, weights, _ = self.successors(u)
            for v, w in zip(nbrs, weights):
                nd = d + w
                if nd > limit_m:
                    continue
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, int(v)))
                    if v in remaining:
                        found[int(v)] = nd
                        remaining.discard(int(v))
            if not remaining:
                break
        return found

    # ------------------------------------------------------------- diagnostics
    def shortest_path_arcs(self, source: int, target: int,
                           limit_m: float) -> list[int] | None:
        """Bounded Dijkstra returning the arc ids along the path, or None.

        Used to expand a decoded arc sequence into a fully connected route:
        at 1 Hz a vehicle can traverse a short arc entirely between two GPS
        fixes, so consecutive decoded arcs are often connected *through* an
        arc that no observation ever landed on.
        """
        if source == target:
            return []
        dist = {source: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit_m:
                if d > limit_m:
                    return None
                continue
            if u == target:
                path, node = [], target
                while node != source:
                    pnode, arc = prev[node]
                    path.append(arc)
                    node = pnode
                return path[::-1]
            nbrs, weights, arc_ids = self.successors(u)
            for v, w, a in zip(nbrs, weights, arc_ids):
                nd = d + w
                if nd > limit_m:
                    continue
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[int(v)] = (int(u), int(a))
                    heapq.heappush(heap, (nd, int(v)))
        return None

    def connectivity_summary(self) -> dict:
        """Weakly-connected-component statistics, for the EDA report."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        data = np.ones(self.n_arcs, dtype=np.int8)
        m = coo_matrix((data, (self.arcs[:, 0], self.arcs[:, 1])),
                       shape=(self.n_nodes, self.n_nodes))
        n_comp, labels = connected_components(m, directed=True, connection="weak")
        sizes = np.bincount(labels)
        isolated = int(np.sum((self.out_degree == 0) & (self.in_degree == 0)))
        return {
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_weak_components": int(n_comp),
            "largest_component_frac": float(sizes.max() / self.n_nodes),
            "n_isolated_nodes": isolated,
            "mean_out_degree": float(self.out_degree.mean()),
            "arc_length_mean_m": float(self.arc_length.mean()),
            "arc_length_median_m": float(np.median(self.arc_length)),
            "arc_length_max_m": float(self.arc_length.max()),
            "n_zero_length_arcs": int(np.sum(self.arc_length == 0)),
        }


def build_network(trip, projection: LocalProjection | None = None) -> RoadNetwork:
    if projection is None:
        projection = LocalProjection(float(np.mean(trip.track[:, 0])),
                                     float(np.mean(trip.track[:, 1])))
    return RoadNetwork(trip.nodes, trip.arcs, projection)
