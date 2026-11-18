"""Raw-data loading and inventory.

Raw layout (verified against the supplied archive, not assumed):

    <partition>/<trip_id>/<trip_id>.track   lon lat timestamp_seconds   (whitespace, no header)
    <partition>/<trip_id>/<trip_id>.nodes   lon lat                     (row index = node_id)
    <partition>/<trip_id>/<trip_id>.arcs    from_node_id to_node_id     (row index = arc_id)
    <partition>/<trip_id>/<trip_id>.route   arc_id                      (chronological, optional)

Every trip ships its *own* road sub-network, so node and arc ids are local to
the trip and must never be compared across trips.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray          # (T, 3) lon, lat, timestamp_s
    nodes: np.ndarray          # (N, 2) lon, lat
    arcs: np.ndarray           # (M, 2) from_node, to_node
    route: np.ndarray | None   # (R,) arc ids, or None when not supplied

    @property
    def n_obs(self) -> int:
        return len(self.track)

    @property
    def lonlat(self) -> np.ndarray:
        return self.track[:, :2]

    @property
    def timestamps(self) -> np.ndarray:
        return self.track[:, 2]


def list_trips(partition_dir: str | os.PathLike) -> list[str]:
    p = Path(partition_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"partition directory not found: {p}")
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def load_trip(partition_dir: str | os.PathLike, trip_id: str,
              partition: str | None = None) -> Trip:
    base = Path(partition_dir) / trip_id
    track = np.loadtxt(base / f"{trip_id}.track", dtype=np.float64, ndmin=2)
    nodes = np.loadtxt(base / f"{trip_id}.nodes", dtype=np.float64, ndmin=2)
    arcs = np.loadtxt(base / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)

    route_path = base / f"{trip_id}.route"
    route = None
    if route_path.exists() and route_path.stat().st_size > 0:
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)

    if track.shape[1] != 3:
        raise ValueError(f"{trip_id}: .track has {track.shape[1]} columns, expected 3")
    if nodes.shape[1] != 2:
        raise ValueError(f"{trip_id}: .nodes has {nodes.shape[1]} columns, expected 2")
    if arcs.shape[1] != 2:
        raise ValueError(f"{trip_id}: .arcs has {arcs.shape[1]} columns, expected 2")

    return Trip(trip_id=trip_id, partition=partition or Path(partition_dir).name,
                track=track, nodes=nodes, arcs=arcs, route=route)


def iter_trips(partition_dir: str | os.PathLike, trip_ids: list[str] | None = None):
    partition_dir = Path(partition_dir)
    for tid in (trip_ids if trip_ids is not None else list_trips(partition_dir)):
        yield load_trip(partition_dir, tid)
