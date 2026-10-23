"""Coordinate frames: GPS (WGS84) <-> world ENU <-> runway frame.

Frames
------
GPS     latitude / longitude (deg), altitude (m above the WGS84 ellipsoid)
world   local ENU tangent plane at the runway centre: x east, y north, z up (m).
        This is the frame PyBullet runs in.
runway  x along the runway (toward the heading), y to the LEFT of travel, z up.
        The origin is the runway centre. This is the frame for the coverage
        grid, lane centre lines and crack detections.

The runway frame is the world frame rotated about z so that +x points along the
runway heading. With heading measured clockwise from north (aviation
convention): heading 90 deg = due east, so runway axes equal world axes.

GPS <-> ENU goes through ECEF using the exact WGS84 ellipsoid, so the round
trip is exact, with no flat-earth approximation error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GeoFrame"]

_A = 6378137.0                     # WGS84 semi-major axis (m)
_F = 1 / 298.257223563             # WGS84 flattening
_E2 = _F * (2 - _F)                # first eccentricity squared


def _geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    n = _A / math.sqrt(1 - _E2 * math.sin(lat_rad) ** 2)
    x = (n + alt_m) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (n + alt_m) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (n * (1 - _E2) + alt_m) * math.sin(lat_rad)
    return np.array([x, y, z])


def _ecef_to_geodetic(p: np.ndarray) -> tuple[float, float, float]:
    """Iterative ECEF -> geodetic (converges to well below 1 mm in a few steps)."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1 - _E2))
    alt = 0.0
    for _ in range(10):
        n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
        alt = r / math.cos(lat) - n if abs(lat) < math.radians(89.0) else abs(z) / math.sin(abs(lat)) - n * (1 - _E2)
        new_lat = math.atan2(z, r * (1 - _E2 * n / (n + alt)))
        if abs(new_lat - lat) < 1e-14:
            lat = new_lat
            break
        lat = new_lat
    return lat, lon, alt


@dataclass(frozen=True)
class GeoFrame:
    """Transforms anchored at a GPS origin and a runway heading."""

    lat_deg: float
    lon_deg: float
    alt_m: float
    heading_deg: float

    def __post_init__(self):
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
        # rows are the east, north, up unit vectors expressed in ECEF
        rot = np.array([[-so,       co,      0.0],
                        [-sl * co, -sl * so, cl],
                        [ cl * co,  cl * so, sl]])
        # runway heading: clockwise from north. +x_runway = (sin h, cos h) in (E, N).
        h = math.radians(self.heading_deg)
        c, s = math.sin(h), math.cos(h)          # (east, north) components of +x_runway
        # world->runway: x_r = c*E + s*N ; y_r = -s*E + c*N  (y_r = left of travel)
        w2r = np.array([[c,  s, 0.0],
                        [-s, c, 0.0],
                        [0.0, 0.0, 1.0]])
        object.__setattr__(self, "_ecef0", _geodetic_to_ecef(lat, lon, self.alt_m))
        object.__setattr__(self, "_ecef2enu", rot)
        object.__setattr__(self, "_w2r", w2r)

    # -- GPS <-> world -------------------------------------------------------- #
    def gps_to_world(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        """GPS -> world ENU (x east, y north, z up), metres from the origin."""
        p = _geodetic_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m)
        return self._ecef2enu @ (p - self._ecef0)

    def world_to_gps(self, xyz) -> tuple[float, float, float]:
        """World ENU -> (lat_deg, lon_deg, alt_m)."""
        p = self._ecef0 + self._ecef2enu.T @ np.asarray(xyz, dtype=float)
        lat, lon, alt = _ecef_to_geodetic(p)
        return math.degrees(lat), math.degrees(lon), alt

    # -- world <-> runway ----------------------------------------------------- #
    def world_to_runway(self, xyz) -> np.ndarray:
        return self._w2r @ np.asarray(xyz, dtype=float)

    def runway_to_world(self, xyz) -> np.ndarray:
        return self._w2r.T @ np.asarray(xyz, dtype=float)

    # -- GPS <-> runway ------------------------------------------------------- #
    def gps_to_runway(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        return self.world_to_runway(self.gps_to_world(lat_deg, lon_deg, alt_m))

    def runway_to_gps(self, xyz) -> tuple[float, float, float]:
        return self.world_to_gps(self.runway_to_world(xyz))

    @property
    def rotation_world_to_runway(self) -> np.ndarray:
        """3x3 matrix R with p_runway = R @ p_world (a copy; the origin is shared)."""
        return self._w2r.copy()

    def runway_to_world_array(self, pts) -> np.ndarray:
        """Vectorised runway -> world for an (N, 3) array of points or vectors."""
        return np.asarray(pts, dtype=float) @ self._w2r

    # -- orientation ---------------------------------------------------------- #
    def runway_yaw_world_rad(self) -> float:
        """Yaw (about +z, counter-clockwise from east) of the runway +x axis in the world frame."""
        return math.atan2(math.cos(math.radians(self.heading_deg)),
                          math.sin(math.radians(self.heading_deg)))
