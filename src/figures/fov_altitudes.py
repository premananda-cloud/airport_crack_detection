"""Paper figure: camera field of view at several altitudes, as vector graphics (PDF and SVG).

One panel per altitude, all at the SAME scale, so the growth of the footprint with height is
honest. Each panel is an across-track cross-section (the camera points straight down):

    camera cone and its field-of-view angle
    ground footprint of the camera       (blue, thick)
    runway width                         (dark bar on the ground line)
    altitude and footprint dimension lines
    below each panel: footprint (across x along), ground sample distance, lanes needed

Everything is computed from src/config.py (camera FOV and resolution, runway width, overlap),
so the figure follows the config. The drone symbol is schematic, not to scale.

    uv run python -m src.figures.fov_altitudes
    uv run python -m src.figures.fov_altitudes --altitudes 3 12 25 --out figures/fov
    uv run python -m src.figures.fov_altitudes my_run.toml --formats pdf

Needs matplotlib:  uv add --dev matplotlib
PDF fonts are embedded as TrueType (accepted by publishers); SVG text is converted to outlines
so it renders the same everywhere.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, Polygon, Rectangle

from src.config import Config

INK = "#111111"
BLUE = "#0072B2"          # colour-blind safe; prints as a dark grey
GREY = "#6b6b6b"
RUNWAY = "#3a3a3a"
STRIP = 0.9               # drawn depth of the ground / runway section, metres
FOV_ARC_R = 2.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "hatch.linewidth": 0.4,
})


def _stats(cfg: Config, h: float):
    """Footprint, GSD and lane layout at altitude h (the config's own derived-value code)."""
    insp = dataclasses.replace(cfg.inspection, altitude_m=h,
                               volume_ceiling_m=max(cfg.inspection.volume_ceiling_m, h + 1.0))
    c = dataclasses.replace(cfg, inspection=insp)
    across, along = c.footprint_m()
    return across, along, c.gsd_m() * 1000.0, c.lane_layout()


def _drone(ax, x: float, y: float, s: float = 0.65) -> None:
    """Schematic quadrotor seen from the front. (x, y) is the camera lens, at the bottom centre."""
    ax.add_patch(Rectangle((x - 0.55 * s, y), 1.1 * s, 0.5 * s, facecolor=INK, lw=0, zorder=6))
    for sign in (-1, 1):
        ax.plot([x + sign * 0.55 * s, x + sign * 1.5 * s], [y + 0.5 * s, y + 0.7 * s],
                color=INK, lw=1.0, zorder=6)
        ax.add_patch(Ellipse((x + sign * 1.5 * s, y + 0.8 * s), 1.3 * s, 0.16 * s,
                             facecolor=INK, lw=0, zorder=6))
    ax.plot([x], [y], marker="o", ms=2.5, color=INK, zorder=7)


def _dimension(ax, p0, p1, colour: str) -> None:
    ax.annotate("", xy=p1, xytext=p0, zorder=5,
                arrowprops=dict(arrowstyle="<->", color=colour, lw=0.7, shrinkA=0, shrinkB=0,
                                mutation_scale=6))


def _panel(ax, cfg: Config, h: float, title: str, half_w: float, top: float, bottom: float) -> None:
    across, along, gsd_mm, lanes = _stats(cfg, h)
    width = cfg.runway.width_m
    hfov = cfg.camera.hfov_deg

    # ground: hatched grass either side, dark runway section, thin surface line
    ax.add_patch(Rectangle((-half_w, -STRIP), 2 * half_w, STRIP, facecolor="none",
                           edgecolor=GREY, hatch="////", lw=0, zorder=2))
    ax.add_patch(Rectangle((-width / 2, -STRIP), width, STRIP, facecolor=RUNWAY,
                           edgecolor=INK, lw=0.6, zorder=3))
    ax.plot([-half_w, half_w], [0, 0], color=INK, lw=0.8, zorder=3)
    for yc in lanes.centres_y_m:                       # lane centres, white ticks on the runway
        ax.plot([yc], [-STRIP / 2], ls="none", marker="|", ms=4.5, mew=1.1, color="white", zorder=4)

    # camera cone, footprint, field-of-view angle
    ax.add_patch(Polygon([(0, h), (-across / 2, 0), (across / 2, 0)], closed=True,
                         facecolor=BLUE, alpha=0.13, lw=0, zorder=1))
    ax.plot([-across / 2, 0, across / 2], [0, h, 0], color=BLUE, lw=1.0, zorder=4)
    ax.plot([-across / 2, across / 2], [0, 0], color=BLUE, lw=3.4, solid_capstyle="butt", zorder=5)
    r = min(0.3 * h, FOV_ARC_R)
    ax.add_patch(Arc((0, h), 2 * r, 2 * r, theta1=270 - hfov / 2, theta2=270 + hfov / 2,
                     color=BLUE, lw=0.8, zorder=4))
    d = r + 0.4                                        # label just outside the cone, beside the arc
    ax.text(0.52 * d * (hfov / 55.0) + 0.6, h - d, f"{hfov:g}°", color=BLUE, fontsize=7.5,
            ha="left", va="center")
    _drone(ax, 0, h)

    # altitude dimension (left)
    xd = -half_w + 2.2
    ax.plot([-1.8, xd - 0.4], [h, h], color=GREY, lw=0.5, ls=":", zorder=1)
    _dimension(ax, (xd, 0), (xd, h), INK)
    ax.text(xd + 0.45, h / 2, f"{h:g} m", ha="left", va="center", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)

    # footprint and runway dimensions (below)
    # runway row nearest the ground, footprint row below it, so the footprint's extension
    # lines (which lie outside the runway in b and c) never run through the runway label
    y1, y2 = -STRIP - 1.1, -STRIP - 4.5
    for x in (-width / 2, width / 2):
        ax.plot([x, x], [-STRIP, y1 - 0.2], color=GREY, lw=0.5, ls=":", zorder=1)
    _dimension(ax, (-width / 2, y1), (width / 2, y1), INK)
    ax.text(0, y1 - 0.45, f"{width:g} m", color=INK, ha="center", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)
    for x in (-across / 2, across / 2):
        ax.plot([x, x], [-STRIP, y2 - 0.2], color=BLUE, lw=0.5, ls=":", zorder=1)
    _dimension(ax, (-across / 2, y2), (across / 2, y2), BLUE)
    ax.text(0, y2 - 0.45, f"{across:.1f} m", color=BLUE, ha="center", va="top", fontsize=8)

    lane_txt = (f"{lanes.n_lanes} lanes, pitch {lanes.pitch_m:.1f} m" if lanes.n_lanes > 1
                else "1 lane covers the runway")
    ax.text(0.5, -0.01, f"footprint {across:.1f} × {along:.1f} m\nGSD {gsd_mm:.1f} mm/px\n{lane_txt}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.4)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")

    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def make_figure(cfg: Config, altitudes, note: bool = True):
    footprints = [_stats(cfg, h)[0] for h in altitudes]
    half_w = max(footprints) / 2 + 4.2
    top = max(altitudes) + 3.0
    bottom = -STRIP - 4.5 - 2.9

    n = len(altitudes)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n, 4.5), squeeze=False)
    for i, (ax, h) in enumerate(zip(axes[0], altitudes)):
        _panel(ax, cfg, h, f"({chr(97 + i)})", half_w, top, bottom)

    if note:
        c = cfg.camera
        fig.text(0.5, 0.005,
                 f"Across-track view, camera pointing straight down; drone not to scale. All panels share one scale. "
                 f"White ticks: lane centres.\nCamera {c.name} ({c.width_px}×{c.height_px} px, "
                 f"HFOV {c.hfov_deg:g}°, VFOV {c.vfov_deg:.0f}°), overlap {cfg.mission.overlap:.0%}.",
                 ha="center", va="bottom", fontsize=6.5, color=GREY, linespacing=1.4)
        fig.subplots_adjust(bottom=0.13, wspace=0.04, left=0.01, right=0.99, top=0.95)
    else:
        fig.subplots_adjust(bottom=0.1, wspace=0.04, left=0.01, right=0.99, top=0.95)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="optional TOML file overriding config defaults")
    ap.add_argument("--altitudes", type=float, nargs="+", default=[3.0, 12.0, 25.0])
    ap.add_argument("--out", default="figures/fov_altitudes", help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg"], choices=["pdf", "svg", "png"])
    ap.add_argument("--note", action=argparse.BooleanOptionalAction, default=True,
                    help="print the small footnote inside the figure (default on)")
    args = ap.parse_args()

    cfg = Config.from_toml(args.config) if args.config else Config()
    fig = make_figure(cfg, args.altitudes, note=args.note)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        path = out.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
