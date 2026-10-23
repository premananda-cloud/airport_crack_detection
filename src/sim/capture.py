"""Ground-camera photos of the drone hovering at chosen altitudes. Headless, no window needed.

For each altitude the drone is placed level over the runway centre, nose along the runway, and a
camera standing on the ground (eye height 1.6 m) at a horizontal standoff looks straight at it.
Each altitude gives two PNGs:

    ground_<h>m.png        the full view
    ground_<h>m_zoom.png   a crop around the drone, enlarged 4x (nearest-neighbour), to see detail

Camera range is printed too. A 0.3 m drone at 25 m altitude and 30 m standoff is about 39 m away,
so it is only a few pixels wide in the full view; that is what the zoom crop is for.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pybullet as p

from src.config import Config
from src.environment.runway import Runway, _write_png, connect
from src.sim.quadrotor import Quadrotor

__all__ = ["capture_altitude_shots"]


def _project(view, proj, point, width: int, height: int) -> tuple[float, float]:
    """World point -> pixel (x right, y down), using PyBullet's column-major matrices."""
    v = np.array(view).reshape(4, 4, order="F")
    pr = np.array(proj).reshape(4, 4, order="F")
    clip = pr @ v @ np.append(np.asarray(point, dtype=float), 1.0)
    ndc = clip[:3] / clip[3]
    return (ndc[0] + 1.0) / 2.0 * width, (1.0 - ndc[1]) / 2.0 * height


def capture_altitude_shots(cfg: Config, altitudes, out_dir, standoff_m: float = 30.0,
                           eye_height_m: float = 1.6, width: int = 1600, height: int = 900,
                           fov_deg: float = 45.0, zoom_px: int = 160, zoom_factor: int = 4) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, gui=False))

    client = connect(cfg)
    runway = Runway(cfg).build(client)
    quad = Quadrotor(cfg, client)
    geo = runway.geo
    yaw = geo.runway_yaw_world_rad()
    proj = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=width / height, nearVal=0.1,
                                        farVal=1000.0, physicsClientId=client)

    saved: list[Path] = []
    try:
        for h in altitudes:
            drone_w = geo.runway_to_world([0.0, 0.0, h])
            eye_w = geo.runway_to_world([0.0, -standoff_m, eye_height_m])   # on the ground, across the runway
            quad.reset(drone_w, yaw_rad=yaw)

            view = p.computeViewMatrix(eye_w.tolist(), drone_w.tolist(), [0, 0, 1], physicsClientId=client)
            _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj,
                                                renderer=p.ER_TINY_RENDERER, physicsClientId=client)
            img = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
            img = np.ascontiguousarray(img)

            full = out / f"ground_{h:g}m.png"
            _write_png(full, img)
            saved.append(full)

            px, py = _project(view, proj, drone_w, width, height)
            half = zoom_px // 2
            x0 = int(min(max(px - half, 0), width - zoom_px))
            y0 = int(min(max(py - half, 0), height - zoom_px))
            crop = img[y0:y0 + zoom_px, x0:x0 + zoom_px]
            crop = np.repeat(np.repeat(crop, zoom_factor, axis=0), zoom_factor, axis=1)
            zoom = out / f"ground_{h:g}m_zoom.png"
            _write_png(zoom, np.ascontiguousarray(crop))
            saved.append(zoom)

            dist = float(np.linalg.norm(drone_w - eye_w))
            elev = math.degrees(math.atan2(h - eye_height_m, standoff_m))
            print(f"altitude {h:g} m: camera range {dist:.1f} m, elevation {elev:.0f} deg, "
                  f"drone at pixel ({px:.0f}, {py:.0f}) -> {full.name}, {zoom.name}")
    finally:
        p.disconnect(client)
    return saved
