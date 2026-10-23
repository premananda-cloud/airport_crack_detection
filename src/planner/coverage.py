"""Classical coverage planner: lawnmower (boustrophedon) over the runway, then a timed trajectory.

Three steps, all pure NumPy, all in the RUNWAY frame (x along the runway, y left, z up):

1. lane_waypoints(cfg)   ordered lane end points. Lane centres come from cfg.lane_layout()
                         (camera footprint, overlap, runway width). Lanes run along the runway
                         and alternate direction.
2. path                  straight lanes joined by semicircular U-turns of radius pitch / 2.
                         The semicircle overshoots the lane end by that radius, which keeps the
                         path smooth (continuous heading) and needs no spline fitting.
3. speed profile         v(s) <= v_max and v(s) <= sqrt(a_lat / curvature) (so turns are slow),
                         then forward and backward passes limit the along-track acceleration.
                         Starts and ends at rest. Integrated to time and sampled at control_hz.

The result is a Trajectory: reference position and velocity per control step. RAPTOR has no
look-ahead, so it is fed only the error to the reference at the current step.

    traj = plan_coverage(cfg)          # full lawnmower
    traj = hover_trajectory(cfg, 10)   # sanity test: hold the start point, zero velocity
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.config import Config

__all__ = ["Trajectory", "lane_x_range", "lane_waypoints", "plan_coverage", "hover_trajectory"]


@dataclass(frozen=True)
class Trajectory:
    dt: float               # seconds between samples (= 1 / control_hz)
    pos: np.ndarray         # (N, 3) reference position, runway frame, metres
    vel: np.ndarray         # (N, 3) reference velocity, runway frame, m/s
    path_length_m: float
    n_lanes: int
    label: str

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def duration_s(self) -> float:
        return (self.n - 1) * self.dt

    def at(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Reference (position, velocity) at control step k. Past the end: hold the final point at rest."""
        if k >= self.n - 1:
            return self.pos[-1], np.zeros(3)
        return self.pos[max(k, 0)], self.vel[max(k, 0)]

    def to_world(self, geo) -> tuple[np.ndarray, np.ndarray]:
        """(pos, vel) converted to the world frame, e.g. for plotting in the GUI."""
        return geo.runway_to_world_array(self.pos), geo.runway_to_world_array(self.vel)

    def summary(self) -> str:
        speed = np.linalg.norm(self.vel, axis=1)
        return (f"{self.label}: {self.n_lanes} lane(s), path {self.path_length_m:.1f} m, "
                f"{self.duration_s:.1f} s, max speed {speed.max():.2f} m/s, "
                f"mean {speed.mean():.2f} m/s, {self.n} samples at {1 / self.dt:g} Hz")


# --------------------------------------------------------------------------- #
# lanes
# --------------------------------------------------------------------------- #
def lane_x_range(cfg: Config) -> tuple[float, float]:
    """Where the camera centre must travel along the runway.

    Inset by half the along-track footprint so the footprint edge just reaches each runway end.
    The inset is capped so the lane keeps at least 20% of the runway length.
    """
    length = cfg.runway.length_m
    _, along = cfg.footprint_m()
    inset = min(along / 2.0, 0.4 * length)
    return -length / 2.0 + inset, length / 2.0 - inset


def lane_waypoints(cfg: Config) -> np.ndarray:
    """Ordered (2 * n_lanes, 2) lane end points (x, y) in the runway frame, lawnmower order."""
    x0, x1 = lane_x_range(cfg)
    pts = []
    for i, y in enumerate(cfg.lane_layout().centres_y_m):
        a, b = (x0, x1) if i % 2 == 0 else (x1, x0)
        pts += [(a, y), (b, y)]
    return np.array(pts)


# --------------------------------------------------------------------------- #
# path
# --------------------------------------------------------------------------- #
def _build_path(x0: float, x1: float, ys: tuple[float, ...], ds: float):
    """Dense path samples: xy (N, 2), heading (N,) radians, curvature (N,) 1/m, arc length s (N,)."""
    segs = []
    for i, y in enumerate(ys):
        fwd = i % 2 == 0
        xa, xb = (x0, x1) if fwd else (x1, x0)
        n = max(2, math.ceil(abs(xb - xa) / ds) + 1)
        xs = np.linspace(xa, xb, n)
        segs.append((np.column_stack([xs, np.full(n, y)]),
                     np.full(n, 0.0 if fwd else math.pi),
                     np.zeros(n)))
        if i < len(ys) - 1:
            r = (ys[i + 1] - y) / 2.0
            cx, cy = xb, y + r                      # arc centre, halfway to the next lane
            n = max(3, math.ceil(math.pi * r / ds) + 1)
            if fwd:     # heading +x, next lane on the left: counter-clockwise, bulges toward +x
                th = np.linspace(-math.pi / 2, math.pi / 2, n)
                head = th + math.pi / 2
            else:       # heading -x, next lane on the right of travel: clockwise, bulges toward -x
                th = np.linspace(-math.pi / 2, -3 * math.pi / 2, n)
                head = th - math.pi / 2
            segs.append((np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]),
                         head, np.full(n, 1.0 / r)))

    xy = np.concatenate([segs[0][0]] + [s[0][1:] for s in segs[1:]])        # drop duplicate joints
    heading = np.concatenate([segs[0][1]] + [s[1][1:] for s in segs[1:]])
    kappa = np.concatenate([segs[0][2]] + [s[2][1:] for s in segs[1:]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    return xy, heading, kappa, s


def _speed_profile(s: np.ndarray, kappa: np.ndarray, v_max: float,
                   a_lat: float, a_long: float) -> np.ndarray:
    """Speed along the path: curvature limit, then forward / backward acceleration limits."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-12)))
    v[0] = 0.0
    for i in range(len(s) - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * a_long * (s[i + 1] - s[i])))
    v[-1] = 0.0
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_long * (s[i + 1] - s[i])))
    return v


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #
def plan_coverage(cfg: Config, ds: float = 0.05) -> Trajectory:
    """Full lawnmower coverage trajectory sampled at cfg.sim.control_hz."""
    lanes = cfg.lane_layout()
    x0, x1 = lane_x_range(cfg)
    xy, heading, kappa, s = _build_path(x0, x1, lanes.centres_y_m, ds)
    if len(s) < 3:
        raise ValueError("coverage path is too short to plan: check runway length and footprint")

    m = cfg.mission
    v = _speed_profile(s, kappa, m.max_speed_mps, m.max_lateral_accel_mps2,
                       m.max_longitudinal_accel_mps2)

    # time at each path sample: constant acceleration between samples, dt = 2 ds / (v0 + v1)
    seg_t = 2.0 * np.diff(s) / (v[:-1] + v[1:])
    t_path = np.concatenate([[0.0], np.cumsum(seg_t)])

    dt = 1.0 / cfg.sim.control_hz
    tk = np.append(np.arange(0.0, t_path[-1], dt), t_path[-1])
    sk = np.interp(tk, t_path, s)
    vk = np.interp(tk, t_path, v)
    px, py = np.interp(sk, s, xy[:, 0]), np.interp(sk, s, xy[:, 1])
    tx, ty = np.interp(sk, s, np.cos(heading)), np.interp(sk, s, np.sin(heading))
    norm = np.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm

    z = cfg.inspection.altitude_m
    pos = np.column_stack([px, py, np.full(len(tk), z)])
    vel = np.column_stack([vk * tx, vk * ty, np.zeros(len(tk))])
    return Trajectory(dt, pos, vel, float(s[-1]), lanes.n_lanes, "coverage")


def hover_trajectory(cfg: Config, duration_s: float = 10.0) -> Trajectory:
    """Stationary target at the start of the coverage path (altitude, zero velocity).

    The first sanity check: the drone must hold this point without oscillation or drift.
    """
    x0, _ = lane_x_range(cfg)
    y0 = cfg.lane_layout().centres_y_m[0]
    n = int(round(duration_s * cfg.sim.control_hz)) + 1
    point = np.array([x0, y0, cfg.inspection.altitude_m])
    return Trajectory(1.0 / cfg.sim.control_hz, np.tile(point, (n, 1)), np.zeros((n, 3)),
                      0.0, 1, "hover")
