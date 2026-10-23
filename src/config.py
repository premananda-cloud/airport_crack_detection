"""Simulation configuration.

Inputs (things you choose or measure) are plain dataclass fields with defaults.
Derived values (camera footprint, lane layout, inspection volume) are computed
from them, so changing the camera or the altitude never means recalculating by
hand.

Use the defaults, or override from a TOML file with the same section names:

    cfg = Config()                          # defaults
    cfg = Config.from_toml("my_run.toml")   # any key you leave out keeps its default

Frames (see src/environment/geo.py):
    world  = local ENU (x east, y north, z up), origin at the runway centre
    runway = x along the runway, y to the left of travel, z up, same origin
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "GeoCfg", "RunwayCfg", "InspectionCfg", "CameraCfg", "MissionCfg", "SimCfg",
    "DroneCfg", "LaneLayout", "Config",
]


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


@dataclass(frozen=True)
class GeoCfg:
    """GPS point at the CENTRE of the runway. PLACEHOLDER (0, 0): set a surveyed point."""
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0

    def __post_init__(self):
        _check(-90 <= self.lat_deg <= 90, "[geo] lat_deg must be in [-90, 90]")
        _check(-180 <= self.lon_deg <= 180, "[geo] lon_deg must be in [-180, 180]")


@dataclass(frozen=True)
class RunwayCfg:
    length_m: float = 100.0
    width_m: float = 6.0
    heading_deg: float = 90.0     # true heading of the long axis, clockwise from north (90 = east)
    thickness_m: float = 0.1      # slab height above the surrounding ground; top surface is z = 0
    ground_size_m: float = 400.0  # square of grass around the runway
    # surface look (visual only)
    texture: bool = True          # procedural asphalt noise; false = flat dark grey
    texture_ppm: float = 20.0     # texture pixels per metre of runway (auto-capped for long runways)
    texture_seed: int = 0
    markings: bool = True         # edge lines, centreline dashes, threshold bars
    line_width_m: float = 0.15
    dash_m: float = 3.0
    gap_m: float = 2.0
    bar_length_m: float = 2.0     # threshold bar length along the runway
    bar_width_m: float = 0.3
    bar_gap_m: float = 0.3
    bar_offset_m: float = 0.5     # distance of the bars from each runway end

    def __post_init__(self):
        for name in ("length_m", "width_m", "thickness_m", "ground_size_m", "texture_ppm",
                     "line_width_m", "dash_m", "gap_m", "bar_length_m", "bar_width_m", "bar_gap_m"):
            _check(getattr(self, name) > 0, f"[runway] {name} must be > 0")
        _check(0 <= self.heading_deg < 360, "[runway] heading_deg must be in [0, 360)")
        _check(self.ground_size_m >= 1.5 * self.length_m,
               "[runway] ground_size_m must be at least 1.5x length_m")


@dataclass(frozen=True)
class InspectionCfg:
    altitude_m: float = 15.0
    volume_margin_m: float = 5.0    # inspection cuboid extends this far past the runway on each side
    volume_ceiling_m: float = 25.0  # top of the cuboid, above the runway surface

    def __post_init__(self):
        _check(self.altitude_m > 0, "[inspection] altitude_m must be > 0")
        _check(self.volume_margin_m >= 0, "[inspection] volume_margin_m must be >= 0")
        _check(self.volume_ceiling_m > self.altitude_m,
               "[inspection] volume_ceiling_m must be above altitude_m")


@dataclass(frozen=True)
class CameraCfg:
    """PLACEHOLDER profile: the real camera is not chosen yet."""
    name: str = "placeholder_1080p"
    width_px: int = 1920
    height_px: int = 1080
    hfov_deg: float = 55.0   # horizontal FOV; the long image side points across the flight direction

    def __post_init__(self):
        _check(self.width_px > 0 and self.height_px > 0, "[camera] pixel counts must be > 0")
        _check(0 < self.hfov_deg < 180, "[camera] hfov_deg must be in (0, 180)")

    @property
    def vfov_deg(self) -> float:
        t = math.tan(math.radians(self.hfov_deg) / 2) * self.height_px / self.width_px
        return math.degrees(2 * math.atan(t))

    def footprint_m(self, altitude_m: float) -> tuple[float, float]:
        """(across-track, along-track) ground footprint of a nadir camera."""
        across = 2 * altitude_m * math.tan(math.radians(self.hfov_deg) / 2)
        return across, across * self.height_px / self.width_px

    def gsd_m(self, altitude_m: float) -> float:
        """Ground sample distance, metres per pixel across the track."""
        return self.footprint_m(altitude_m)[0] / self.width_px


@dataclass(frozen=True)
class MissionCfg:
    overlap: float = 0.20            # 0.10 to 0.30 typical
    max_speed_mps: float = 3.0
    max_lateral_accel_mps2: float = 1.5
    max_longitudinal_accel_mps2: float = 1.5   # speeding up / slowing down along the path

    def __post_init__(self):
        _check(0 <= self.overlap < 1, "[mission] overlap must be in [0, 1)")
        _check(self.max_speed_mps > 0, "[mission] max_speed_mps must be > 0")
        _check(self.max_lateral_accel_mps2 > 0, "[mission] max_lateral_accel_mps2 must be > 0")
        _check(self.max_longitudinal_accel_mps2 > 0, "[mission] max_longitudinal_accel_mps2 must be > 0")


@dataclass(frozen=True)
class SimCfg:
    physics_hz: int = 500
    control_hz: int = 100
    gravity_mps2: float = 9.81
    gui: bool = False
    draw_debug: bool = True   # GUI only: inspection cuboid and axes
    max_tracking_error_m: float = 5.0   # episode ends ("lost_track") beyond this distance from the reference
    end_hold_s: float = 1.0             # keep running this long after the reference trajectory finishes

    def __post_init__(self):
        _check(self.physics_hz > 0 and self.control_hz > 0, "[sim] rates must be > 0")
        _check(self.physics_hz % self.control_hz == 0, "[sim] control_hz must divide physics_hz")

    @property
    def substeps(self) -> int:
        """Physics steps per control step."""
        return self.physics_hz // self.control_hz


@dataclass(frozen=True)
class DroneCfg:
    """Generic inspection quadrotor, X layout. PLACEHOLDER values: replace with the real vehicle.

    RAPTOR was trained on 20 g to 5 kg and thrust-to-weight 1.5 to 5, so these stay inside that range.
    Rotor order (RAPTOR's): front-right, back-right, back-left, front-left. Body frame is FLU.
    """
    mass_kg: float = 1.2
    arm_m: float = 0.16                 # centre to each motor
    ixx_kgm2: float = 0.0035
    iyy_kgm2: float = 0.0035
    izz_kgm2: float = 0.0066
    thrust_to_weight: float = 2.5       # total maximum thrust / weight
    moment_constant_m: float = 0.016    # yaw reaction torque per newton of rotor thrust
    motor_tau_s: float = 0.05           # first-order motor lag
    spin: tuple[int, ...] = (1, -1, 1, -1)   # +1 = CCW seen from above. RAPTOR: CCW, CW, CCW, CW

    def __post_init__(self):
        for name in ("mass_kg", "arm_m", "ixx_kgm2", "iyy_kgm2", "izz_kgm2",
                     "moment_constant_m", "motor_tau_s"):
            _check(getattr(self, name) > 0, f"[drone] {name} must be > 0")
        _check(self.thrust_to_weight > 1.0, "[drone] thrust_to_weight must be > 1")
        object.__setattr__(self, "spin", tuple(int(s) for s in self.spin))
        _check(len(self.spin) == 4 and all(s in (-1, 1) for s in self.spin),
               "[drone] spin must be four values, each +1 or -1")

    @property
    def hover_command(self) -> float:
        """Normalised motor command (0..1) that holds hover, with thrust = max_thrust * command^2."""
        return 1.0 / math.sqrt(self.thrust_to_weight)


@dataclass(frozen=True)
class LaneLayout:
    footprint_across_m: float
    swath_m: float            # new coverage per lane = footprint * (1 - overlap)
    n_lanes: int
    pitch_m: float
    actual_overlap: float     # 1 - pitch / footprint; rounding lanes up adds some
    centres_y_m: tuple[float, ...]   # lane centre lines in the runway frame


_SECTIONS = {"geo": GeoCfg, "runway": RunwayCfg, "inspection": InspectionCfg,
             "camera": CameraCfg, "mission": MissionCfg, "sim": SimCfg, "drone": DroneCfg}


@dataclass(frozen=True)
class Config:
    geo: GeoCfg = field(default_factory=GeoCfg)
    runway: RunwayCfg = field(default_factory=RunwayCfg)
    inspection: InspectionCfg = field(default_factory=InspectionCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        unknown = set(raw) - set(_SECTIONS)
        _check(not unknown, f"unknown section(s): {sorted(unknown)}")
        built = {}
        for f in fields(cls):
            try:
                built[f.name] = _SECTIONS[f.name](**raw.get(f.name, {}))
            except TypeError as exc:   # unknown key inside a section
                raise ValueError(f"[{f.name}] {exc}") from None
        return cls(**built)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with Path(path).open("rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    # -- derived ------------------------------------------------------------ #
    def inspection_volume_m(self) -> tuple[float, float, float]:
        """(length, width, height) of the inspection cuboid."""
        m = self.inspection.volume_margin_m
        return (self.runway.length_m + 2 * m,
                self.runway.width_m + 2 * m,
                self.inspection.volume_ceiling_m)

    def footprint_m(self) -> tuple[float, float]:
        return self.camera.footprint_m(self.inspection.altitude_m)

    def gsd_m(self) -> float:
        return self.camera.gsd_m(self.inspection.altitude_m)

    def lane_layout(self) -> LaneLayout:
        across, _ = self.footprint_m()
        swath = across * (1 - self.mission.overlap)
        w = self.runway.width_m
        n = max(1, math.ceil(w / swath - 1e-9))
        pitch = w / n
        centres = tuple(-w / 2 + pitch * (i + 0.5) for i in range(n))
        return LaneLayout(across, swath, n, pitch, 1 - pitch / across, centres)

    def warnings(self) -> list[str]:
        out = []
        if self.geo.lat_deg == 0.0 and self.geo.lon_deg == 0.0:
            out.append("[geo] origin is lat 0, lon 0: a placeholder, not a real runway location.")
        lanes = self.lane_layout()
        if lanes.n_lanes == 1:
            out.append(
                f"One lane covers the full {self.runway.width_m:g} m width at "
                f"{self.inspection.altitude_m:g} m (footprint {lanes.footprint_across_m:.1f} m): "
                "lane-end turns and multi-lane logic will not be exercised.")
        if self.runway.length_m < 3 * self.runway.width_m:
            out.append("Runway is shorter than 3x its width; markings may be skipped.")
        return out

    def summary(self) -> str:
        r, i, c = self.runway, self.inspection, self.camera
        across, along = self.footprint_m()
        lanes = self.lane_layout()
        vl, vw, vh = self.inspection_volume_m()
        return "\n".join([
            f"runway      {r.length_m:g} x {r.width_m:g} m, heading {r.heading_deg:g} deg",
            f"geo origin  lat {self.geo.lat_deg:.6f}, lon {self.geo.lon_deg:.6f}, alt {self.geo.alt_m:g} m",
            f"inspection  altitude {i.altitude_m:g} m, volume {vl:g} x {vw:g} x {vh:g} m",
            f"camera      {c.name}: {c.width_px}x{c.height_px}, hfov {c.hfov_deg:g} deg (vfov {c.vfov_deg:.1f})",
            f"footprint   {across:.2f} x {along:.2f} m, GSD {self.gsd_m() * 1000:.2f} mm/px",
            f"lanes       {lanes.n_lanes} at pitch {lanes.pitch_m:.2f} m "
            f"(swath {lanes.swath_m:.2f} m, actual overlap {lanes.actual_overlap:.0%})",
            f"limits      {self.mission.max_speed_mps:g} m/s, {self.mission.max_lateral_accel_mps2:g} m/s^2 lateral",
            f"sim         physics {self.sim.physics_hz} Hz, control {self.sim.control_hz} Hz",
            f"drone       {self.drone.mass_kg:g} kg, arm {self.drone.arm_m:g} m, "
            f"T/W {self.drone.thrust_to_weight:g}, hover command {self.drone.hover_command:.2f}",
        ])
