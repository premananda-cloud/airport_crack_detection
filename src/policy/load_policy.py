"""RAPTOR policy loader (the `foundation-policy` package, pure NumPy, no GPU).

    from src.policy.load_policy import RaptorPolicy
    policy = RaptorPolicy()
    policy.reset()                  # clear the recurrent (GRU) state at every episode start
    action = policy.act(obs)        # obs: 22 floats -> action: 4 floats in [-1, 1]

Observation, 22 values in this order (source: foundation_policy README / examples and the
RAPTOR paper, arXiv 2509.11481):

    [0:3]    position error       p - p_ref
    [3:12]   rotation matrix      body -> world, row-major (R.ravel())
    [12:15]  linear velocity error v - v_ref
    [15:18]  angular velocity     BODY frame (FLU), rad/s
    [18:22]  previous action      the policy's own last output, in [-1, 1]; zeros at reset

Action: 4 individual motor commands in [-1, 1], order front-right, back-right, back-left,
front-left (CCW, CW, CCW, CW). The package ends in tanh, so the range is [-1, 1]. How that maps
to thrust is NOT specified by the package; src/wrapper/gym_wrapper.py maps it to a 0..1 motor
effort as (a + 1) / 2.

The policy is recurrent: the same instance must see consecutive steps of one episode, and
reset() must be called between episodes.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RaptorPolicy", "OBS_DIM", "ACT_DIM"]

OBS_DIM = 22
ACT_DIM = 4


class RaptorPolicy:
    def __init__(self) -> None:
        try:
            from foundation_policy import QuadrotorPolicy   # alias: Raptor
        except ImportError as exc:
            raise ImportError(
                "foundation-policy is not installed. In the project: `uv add foundation-policy`"
            ) from exc
        # loading also self-checks the checkpoint against a stored example output
        self._net = QuadrotorPolicy()

    @property
    def description(self) -> str:
        """Layer summary, e.g. Sequential(Dense(..), GRU(..), ...)."""
        return self._net.description()

    def reset(self) -> None:
        """Zero the GRU hidden state. Call at the start of every episode."""
        self._net.reset()

    def act(self, obs) -> np.ndarray:
        """One control step. Returns 4 actions in [-1, 1]."""
        o = np.asarray(obs, dtype=np.float32)
        if o.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {o.shape}")
        out = np.asarray(self._net.evaluate_step(o[None, :])[0], dtype=np.float64)
        if out.shape != (ACT_DIM,):
            raise ValueError(f"policy returned shape {out.shape}, expected ({ACT_DIM},)")
        return np.clip(out, -1.0, 1.0)
