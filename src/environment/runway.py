"""The runway environment: ground, asphalt slab, painted markings, debug overlay.

Static scene only. Nothing here moves, so this file should rarely change. What
does change (size, heading, GPS origin, altitude, look) comes from src/config.py.

    from src.config import Config
    from src.environment.runway import Runway, connect

    cfg = Config()
    client = connect(cfg)
    runway = Runway(cfg).build(client)

Run it by itself to look at the scene (from the project root):

    uv run python -m src.environment.runway            # defaults
    uv run python -m src.environment.runway my_run.toml

Everything lives in the world frame (ENU: x east, y north, z up, origin at the
runway centre, runway top surface at z = 0). The runway is the box that is
rotated by `heading` inside that frame. Use `runway.geo` to convert between GPS,
world and runway coordinates.
"""
from __future__ import annotations

import dataclasses
import hashlib
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pybullet as p

from src.config import Config
from src.environment.geo import GeoFrame

__all__ = ["Runway", "connect"]

_GRASS = [0.22, 0.38, 0.17, 1.0]
_CONCRETE_EDGE = [0.45, 0.45, 0.45, 1.0]   # slab sides
_ASPHALT_FLAT = [0.23, 0.23, 0.24, 1.0]    # surface colour when no texture is used
_SURFACE_LIFT_M = 0.001                    # textured surface sits 1 mm above the slab top (no z-fighting)
_MAX_TEXTURE_WIDTH_PX = 4096


def connect(cfg: Config) -> int:
    """Open a PyBullet client: GUI window or headless, per cfg.sim.gui."""
    return p.connect(p.GUI if cfg.sim.gui else p.DIRECT)


class Runway:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.geo = GeoFrame(cfg.geo.lat_deg, cfg.geo.lon_deg, cfg.geo.alt_m,
                            cfg.runway.heading_deg)
        self.client: int | None = None
        self.body_ids: dict[str, int] = {}

    # ------------------------------------------------------------------ build #
    def build(self, client: int) -> "Runway":
        """Create the scene in an existing PyBullet client. Call once per client."""
        self.client = client
        sim = self.cfg.sim
        p.setGravity(0, 0, -sim.gravity_mps2, physicsClientId=client)
        p.setTimeStep(1.0 / sim.physics_hz, physicsClientId=client)

        self._build_ground()
        self._build_slab()
        self._build_surface()
        if sim.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            self._frame_camera()
            if sim.draw_debug:
                self._draw_debug()
        return self

    def _build_ground(self) -> None:
        c, r = self.client, self.cfg.runway
        half, thick = r.ground_size_m / 2, 1.0
        extents = [half, half, thick / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=_GRASS,
                                  physicsClientId=c)
        # ground top sits at z = -thickness, so the slab stands `thickness_m` proud of it
        self.body_ids["ground"] = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[0, 0, -r.thickness_m - thick / 2], physicsClientId=c)

    def _runway_orientation(self) -> list[float]:
        return p.getQuaternionFromEuler([0, 0, self.geo.runway_yaw_world_rad()])

    def _build_slab(self) -> None:
        c, r = self.client, self.cfg.runway
        extents = [r.length_m / 2, r.width_m / 2, r.thickness_m / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=_CONCRETE_EDGE,
                                  physicsClientId=c)
        # top surface at z = 0
        self.body_ids["slab"] = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[0, 0, -r.thickness_m / 2],
            baseOrientation=self._runway_orientation(), physicsClientId=c)

    def _build_surface(self) -> None:
        """One textured quad over the slab top: asphalt noise and painted markings.

        u runs along the runway and v across it. The painted pattern is symmetric
        end to end and side to side, so a flipped image axis cannot change the look.
        """
        c, r = self.client, self.cfg.runway
        hl, hw = r.length_m / 2, r.width_m / 2
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=[[-hl, -hw, 0], [hl, -hw, 0], [hl, hw, 0], [-hl, hw, 0]],
            indices=[0, 1, 2, 0, 2, 3],
            normals=[[0, 0, 1]] * 4,
            uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
            rgbaColor=_ASPHALT_FLAT if not (r.texture or r.markings) else [1, 1, 1, 1],
            specularColor=[0, 0, 0],
            physicsClientId=c)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
            basePosition=[0, 0, _SURFACE_LIFT_M],
            baseOrientation=self._runway_orientation(), physicsClientId=c)
        self.body_ids["surface"] = body
        if r.texture or r.markings:
            tex = p.loadTexture(str(_texture_file(r)), physicsClientId=c)
            p.changeVisualShape(body, -1, textureUniqueId=tex, physicsClientId=c)

    # ----------------------------------------------------------------- helpers #
    def on_runway(self, pos_world) -> bool:
        """True if the (x, y) of a world position is over the runway surface."""
        x, y, _ = self.geo.world_to_runway(pos_world)
        return abs(x) <= self.cfg.runway.length_m / 2 and abs(y) <= self.cfg.runway.width_m / 2

    def in_inspection_volume(self, pos_world) -> bool:
        """True if a world position is inside the inspection cuboid (e.g. episode bounds)."""
        x, y, z = self.geo.world_to_runway(pos_world)
        vl, vw, vh = self.cfg.inspection_volume_m()
        return abs(x) <= vl / 2 and abs(y) <= vw / 2 and 0.0 <= z <= vh

    # ------------------------------------------------------------- GUI extras #
    def _frame_camera(self) -> None:
        r = self.cfg.runway
        p.resetDebugVisualizerCamera(
            cameraDistance=max(r.length_m * 0.7, 15.0),
            cameraYaw=r.heading_deg - 90.0 + 25.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.client)

    def _draw_debug(self) -> None:
        c = self.client
        vl, vw, vh = self.cfg.inspection_volume_m()
        pts = {}
        for iz in (0, 1):
            for ix in (0, 1):
                for iy in (0, 1):
                    pts[(iz, ix, iy)] = self.geo.runway_to_world(
                        [(2 * ix - 1) * vl / 2, (2 * iy - 1) * vw / 2, iz * vh])
        for a in pts:                                   # 12 cuboid edges
            for axis in range(3):
                if a[axis] == 0:
                    b = list(a)
                    b[axis] = 1
                    p.addUserDebugLine(pts[a], pts[tuple(b)], [1, 0.8, 0], 1.5,
                                       physicsClientId=c)
        # world ENU axes: red = east, green = north, blue = up
        for vec, rgb in (([5, 0, 0], [1, 0, 0]), ([0, 5, 0], [0, 1, 0]), ([0, 0, 5], [0, 0, 1])):
            p.addUserDebugLine([0, 0, 0.02], [vec[0], vec[1], vec[2] + 0.02], rgb, 3, physicsClientId=c)
        # runway +x (direction of travel for the lanes)
        tip = self.geo.runway_to_world([10, 0, 0.02])
        p.addUserDebugLine(self.geo.runway_to_world([0, 0, 0.02]), tip, [1, 1, 0], 3, physicsClientId=c)
        p.addUserDebugText("runway +x", tip, [1, 1, 0], 1.2, physicsClientId=c)


# ---------------------------------------------------------------- texture #
def _texture_file(r) -> Path:
    """Generate the asphalt/markings PNG once; reuse it while the runway settings are unchanged."""
    key = hashlib.sha1(repr(r).encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / "airport_crack_detection" / f"runway_{key}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_png(path, _make_image(r))
    return path


def _smooth_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    """Smooth random field in roughly [-3, 3]: random grid, linearly interpolated, no SciPy."""
    cell = max(1, int(cell))
    gh, gw = h // cell + 2, w // cell + 2
    grid = rng.normal(0.0, 1.0, (gh, gw))
    xs, ys = np.arange(w) / cell, np.arange(h) / cell
    along_x = np.stack([np.interp(xs, np.arange(gw), grid[i]) for i in range(gh)])      # (gh, w)
    return np.stack([np.interp(ys, np.arange(gh), along_x[:, j]) for j in range(w)], axis=1)  # (h, w)


def _make_image(r) -> np.ndarray:
    """RGB uint8 image, columns along the runway, rows across it."""
    rng = np.random.default_rng(r.texture_seed)
    ppm = max(1.0, min(r.texture_ppm, _MAX_TEXTURE_WIDTH_PX / r.length_m))
    w, h = max(4, round(r.length_m * ppm)), max(4, round(r.width_m * ppm))

    img = np.full((h, w, 3), 0.23)
    if r.texture:
        img += rng.normal(0.0, 0.025, (h, w, 1))                              # grain
        img += (0.015 * _smooth_noise(rng, h, w, ppm * 0.5))[..., None]       # small blotches
        img += (0.015 * _smooth_noise(rng, h, w, ppm * 4.0))[..., None]       # large patches

    if r.markings:
        # pixel-centre coordinates in the runway frame, metres from the centre
        X = ((np.arange(w) + 0.5) * (r.length_m / w) - r.length_m / 2)[None, :]
        Y = ((np.arange(h) + 0.5) * (r.width_m / h) - r.width_m / 2)[:, None]
        paint = np.zeros((h, w), dtype=bool)

        # edge lines, one line-width in from each edge
        edge_outer = r.width_m / 2 - r.line_width_m
        paint |= (np.abs(Y) <= edge_outer) & (np.abs(Y) >= edge_outer - r.line_width_m)

        bars_x_inner = r.length_m / 2 - r.bar_offset_m - r.bar_length_m
        if bars_x_inner > 0:
            # centreline dashes: a dash is centred on x = 0, so the pattern is symmetric
            period = r.dash_m + r.gap_m
            in_dash = np.mod(X + r.dash_m / 2, period) < r.dash_m
            paint |= (np.abs(Y) <= r.line_width_m / 2) & in_dash & (np.abs(X) <= bars_x_inner - 1.0)

            # threshold bars at both ends, evenly spread across 80% of the width
            pitch = r.bar_width_m + r.bar_gap_m
            n = int((0.8 * r.width_m + r.bar_gap_m) // pitch)
            in_x = (np.abs(X) >= bars_x_inner) & (np.abs(X) <= bars_x_inner + r.bar_length_m)
            for i in range(n):
                yc = (i - (n - 1) / 2) * pitch
                paint |= (np.abs(Y - yc) <= r.bar_width_m / 2) & in_x

        count = int(paint.sum())
        img[paint] = 0.88 - 0.12 * rng.random((count, 1))                     # worn white paint

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal 8-bit RGB PNG writer (avoids a Pillow dependency)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))   # filter type 0 per row

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


# ------------------------------------------------------------------- demo #
def main() -> None:
    cfg = Config.from_toml(sys.argv[1]) if len(sys.argv) > 1 else Config()
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, gui=True))
    print(cfg.summary())
    for msg in cfg.warnings():
        print("WARNING:", msg)

    client = connect(cfg)
    Runway(cfg).build(client)
    print("Window open. Drag to orbit, scroll to zoom, Ctrl+C or close the window to exit.")
    dt = 1.0 / cfg.sim.physics_hz
    try:
        while p.isConnected(client):
            p.stepSimulation(physicsClientId=client)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
