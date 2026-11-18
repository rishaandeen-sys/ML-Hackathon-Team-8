"""Geographic helpers.

The dataset covers a single city-scale region (Darmstadt, Germany; lon ~8.6,
lat ~49.85).  At that scale a local equirectangular ("flat-earth") projection
centred on the trip is accurate to well under 0.1 % in distance, is exactly
invertible and is far cheaper than a full geodesic computation.  We therefore
project once per trip into metres and do all geometry in that plane.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # mean Earth radius, metres (IUGG)


class LocalProjection:
    """Equirectangular projection to metres about a reference lon/lat."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = R_EARTH * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self._ky = R_EARTH * np.pi / 180.0

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return np.stack([(lon - self.lon0) * self._kx,
                         (lat - self.lat0) * self._ky], axis=-1)

    def to_lonlat(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([xy[..., 0] / self._kx + self.lon0,
                         xy[..., 1] / self._ky + self.lat0], axis=-1)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (np.asarray(lon1, float), np.asarray(lat1, float),
                                              np.asarray(lon2, float), np.asarray(lat2, float)))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments A->B, plus the projection parameter t.

    All inputs broadcast.  Returns (distance, t) with t clipped to [0, 1]:
    t is the normalised position of the foot of the perpendicular along A->B.
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom > 0, (apx * abx + apy * aby) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t


def bearing_rad(dx, dy):
    """Bearing of a planar vector, radians, measured clockwise from north."""
    return np.arctan2(dx, dy)


def angle_difference(a, b):
    """Smallest absolute difference between two angles (radians), in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)
