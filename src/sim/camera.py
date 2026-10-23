"""Cameras for watching the simulation: a follow camera for the GUI and still screenshots.

Both look at a target point (usually the drone) from a given DISTANCE, from the same oblique
angle every time: the runway side, 25 degrees around from broadside, 30 degrees down. Screenshots
use PyBullet's CPU renderer, so they work with or without a GUI window. Debug lines (the yellow
cuboid, axes) are not drawn in screenshots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from src.config import Config
from src.environment.runway import _write_png

__all__ = ["capture", "follow"]

PITCH_DEG = -30.0
YAW_OFFSET_DEG = 25.0


def _yaw_deg(cfg: Config) -> float:
    return cfg.runway.heading_deg - 90.0 + YAW_OFFSET_DEG


def follow(client: int, cfg: Config, target, distance: float) -> None:
    """GUI: point the debug camera at `target` from `distance`. Call every step to track.

    While this is called every step it overrides the mouse, so orbit/zoom will snap back.
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=distance, cameraYaw=_yaw_deg(cfg), cameraPitch=PITCH_DEG,
        cameraTargetPosition=[float(v) for v in target], physicsClientId=client)


def capture(client: int, cfg: Config, target, distance: float, path: str | Path,
            width: int = 1280, height: int = 720) -> Path:
    """Render one PNG of `target` from `distance`. Returns the written path."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(v) for v in target], distance=distance,
        yaw=_yaw_deg(cfg), pitch=PITCH_DEG, roll=0.0, upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05,
                                        farVal=2000.0, physicsClientId=client)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, np.ascontiguousarray(rgb))
    return path
