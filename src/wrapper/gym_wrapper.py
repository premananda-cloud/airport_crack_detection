"""Gymnasium environment: a quadrotor tracking a reference trajectory over the runway.

    env = RunwayInspectionEnv(cfg)                 # lawnmower coverage trajectory
    env = RunwayInspectionEnv(cfg, hover_trajectory(cfg))   # sanity test
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Action  Box(-1, 1, (4,)): RAPTOR's raw motor outputs, order FR, BR, BL, FL.
        Mapped to a 0..1 motor effort as u = (a + 1) / 2   (ASSUMPTION: see load_policy.py).
Obs     Box(22,): exactly what RAPTOR expects, see src/policy/load_policy.py.

Frames. PyBullet runs in the world frame (ENU). The policy is given everything in the RUNWAY
frame (world rotated by the runway heading), and the drone is spawned with its nose along the
runway. That way the policy's "forward" is the runway direction for any heading. The RAPTOR
paper does not state the frame of position and velocity; this is the assumption to revisit if
the hover test misbehaves. For heading 90 deg the runway and world frames are identical.

Episode ends:
    terminated  left_inspection_volume | crashed | flipped | lost_track
    truncated   the reference trajectory is finished (plus cfg.sim.end_hold_s)

Reward is a placeholder, minus the tracking error: -(|p - p_ref| + 0.1 |v - v_ref|), and -100 on
termination. The pretrained policy does not use it; it is there for logging and for any
later learning on top.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from src.config import Config
from src.environment.runway import Runway, connect
from src.planner.coverage import Trajectory, plan_coverage
from src.sim.quadrotor import Quadrotor

__all__ = ["RunwayInspectionEnv"]

_CRASH_HEIGHT_M = 0.1


class RunwayInspectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None, trajectory: Trajectory | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else Config()
        self.client = connect(self.cfg)
        self.runway = Runway(self.cfg).build(self.client)
        self.geo = self.runway.geo
        self.quad = Quadrotor(self.cfg, self.client)
        self.trajectory = trajectory if trajectory is not None else plan_coverage(self.cfg)

        self._w2r = self.geo.rotation_world_to_runway
        self._substeps = self.cfg.sim.substeps
        self._hold_steps = round(self.cfg.sim.end_hold_s * self.cfg.sim.control_hz)
        self._k = 0
        self._prev_action = np.zeros(4)
        self._sense: dict = {}

        self.observation_space = spaces.Box(-np.inf, np.inf, (22,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # ------------------------------------------------------------------ gym API #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._k = 0
        self._prev_action = np.zeros(4)
        start_world = self.geo.runway_to_world(self.trajectory.pos[0])
        self.quad.reset(start_world, yaw_rad=self.geo.runway_yaw_world_rad())
        return self._observe(), self._info(None)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        u = 0.5 * (a + 1.0)
        for _ in range(self._substeps):
            self.quad.apply(u)
            p.stepSimulation(physicsClientId=self.client)
        self._k += 1
        self._prev_action = a

        obs = self._observe()
        s = self._sense
        reason = None
        if not self.runway.in_inspection_volume(s["pos_world"]):
            reason = "left_inspection_volume"
        elif s["pos_world"][2] < _CRASH_HEIGHT_M:
            reason = "crashed"
        elif s["rot_world"][2, 2] < 0.0:
            reason = "flipped"
        elif s["pos_err"] > self.cfg.sim.max_tracking_error_m:
            reason = "lost_track"

        terminated = reason is not None
        truncated = (not terminated) and self._k >= self.trajectory.n - 1 + self._hold_steps
        reward = -100.0 if terminated else -(s["pos_err"] + 0.1 * s["vel_err"])
        return obs, float(reward), terminated, truncated, self._info(reason)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # ---------------------------------------------------------------- internals #
    def _observe(self) -> np.ndarray:
        pos_w, quat, vel_w, ang_w = self.quad.state()
        rot_wb = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)    # body -> world
        ref_pos, ref_vel = self.trajectory.at(self._k)                       # runway frame

        pos_err = self._w2r @ pos_w - ref_pos          # runway frame, shared origin
        vel_err = self._w2r @ vel_w - ref_vel
        rot_rb = self._w2r @ rot_wb                    # body -> runway
        omega_body = rot_wb.T @ ang_w                  # PyBullet reports world-frame angular velocity

        self._sense = {
            "pos_world": pos_w, "rot_world": rot_wb,
            "pos_err": float(np.linalg.norm(pos_err)), "vel_err": float(np.linalg.norm(vel_err)),
            "speed": float(np.linalg.norm(vel_w)),
        }
        obs = np.concatenate([pos_err, rot_rb.ravel(), vel_err, omega_body, self._prev_action])
        return obs.astype(np.float32)

    def _info(self, reason: str | None) -> dict:
        s = self._sense
        return {
            "time_s": self._k / self.cfg.sim.control_hz,
            "pos_error_m": s["pos_err"],
            "vel_error_mps": s["vel_err"],
            "speed_mps": s["speed"],
            "progress": min(self._k / max(self.trajectory.n - 1, 1), 1.0),
            "termination": reason,
        }
