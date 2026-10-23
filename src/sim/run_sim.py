"""Simulation entry point. Run from the project root.

    uv run python -m src.sim.run_sim                         # world only: just look at the runway
    uv run python -m src.sim.run_sim --mode hover            # RAPTOR holds the start point (sanity test)
    uv run python -m src.sim.run_sim --mode fly              # RAPTOR flies the lawnmower coverage path
    uv run python -m src.sim.run_sim --mode fly --follow 25  # GUI camera tracks the drone from 25 m
    uv run python -m src.sim.run_sim --mode fly --headless --shots 3,12,25
                                                             # screenshots from 3, 12 and 25 m away
    uv run python -m src.sim.run_sim --snapshot out.png      # world only, one headless image

Screenshots are taken when the flight reaches each --shot-progress fraction (default 10%, 50%,
90% of the trajectory), one image per --shots distance, saved as
<shots-dir>/p050_d12m.png (progress 50%, camera 12 m from the drone).

An optional TOML file overrides config defaults:  ... run_sim --mode fly my_run.toml

hover / fly need gymnasium:  uv add gymnasium
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # allows `python src/sim/run_sim.py` as well: put the project root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pybullet as p

from src.config import Config
from src.environment.runway import Runway, connect
from src.sim.camera import capture, follow


def _floats(text: str, name: str) -> list[float]:
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise SystemExit(f"{name}: expected comma-separated numbers, got {text!r}") from None


def run_world(cfg: Config, snap: Path | None) -> None:
    client = connect(cfg)
    Runway(cfg).build(client)
    if snap is not None:
        capture(client, cfg, [0, 0, 0], max(cfg.runway.length_m * 0.7, 15.0), snap)
        print(f"saved {snap}")
        p.disconnect(client)
        return
    print("Window open. Drag to orbit, scroll to zoom, Ctrl+C or close the window to exit.")
    dt = 1.0 / cfg.sim.physics_hz
    try:
        while p.isConnected(client):
            p.stepSimulation(physicsClientId=client)
            time.sleep(dt)
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def run_episode(cfg: Config, args) -> None:
    # imported here so the world-only mode works without gymnasium / the policy installed
    from src.planner.coverage import hover_trajectory, plan_coverage
    from src.policy.load_policy import RaptorPolicy
    from src.wrapper.gym_wrapper import RunwayInspectionEnv

    traj = hover_trajectory(cfg, args.hover_seconds) if args.mode == "hover" else plan_coverage(cfg)
    print(traj.summary())

    shot_dists = _floats(args.shots, "--shots") if args.shots else []
    pending = sorted(_floats(args.shot_progress, "--shot-progress")) if shot_dists else []
    shots_dir = Path(args.shots_dir)

    policy = RaptorPolicy()
    print("policy:", policy.description)
    env = RunwayInspectionEnv(cfg, traj)
    obs, info = env.reset()
    policy.reset()

    def watch(info) -> None:
        """Move the follow camera and take any screenshots that are due."""
        drone = env.quad.state()[0]
        if args.follow and cfg.sim.gui:
            follow(env.client, cfg, drone, args.follow)
        while pending and info["progress"] >= pending[0]:
            frac = pending.pop(0)
            for d in shot_dists:
                path = capture(env.client, cfg, drone, d, shots_dir / f"p{round(frac * 100):03d}_d{d:g}m.png")
                print(f"saved {path}")

    dt = 1.0 / cfg.sim.control_hz
    errs, steps, t0 = [], 0, time.perf_counter()
    try:
        watch(info)
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            steps += 1
            errs.append(info["pos_error_m"])
            watch(info)
            if steps % cfg.sim.control_hz == 0:
                print(f"t={info['time_s']:6.1f}s  pos err {info['pos_error_m']:.3f} m  "
                      f"speed {info['speed_mps']:.2f} m/s  progress {info['progress']:.0%}")
            if terminated or truncated:
                break
            if cfg.sim.gui:     # keep GUI playback at real time
                time.sleep(max(0.0, t0 + steps * dt - time.perf_counter()))
    except (KeyboardInterrupt, p.error):
        print("stopped")
        return
    finally:
        env.close()

    errs = np.array(errs)
    outcome = info["termination"] or "finished"
    print(f"\n{outcome}: {steps} steps ({steps * dt:.1f} s), "
          f"position error mean {errs.mean():.3f} m, max {errs.max():.3f} m, final {errs[-1]:.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="optional TOML file overriding config defaults")
    ap.add_argument("--mode", choices=["world", "hover", "fly"], default="world")
    ap.add_argument("--hover-seconds", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true", help="no GUI window")
    ap.add_argument("--snapshot", metavar="PNG", help="world mode: render one headless image and exit")
    ap.add_argument("--follow", type=float, metavar="DIST",
                    help="GUI: keep the camera on the drone from DIST metres away")
    ap.add_argument("--shots", metavar="D1,D2,..", help="screenshot distances from the drone, e.g. 3,12,25")
    ap.add_argument("--shot-progress", default="0.1,0.5,0.9", metavar="F1,F2,..",
                    help="trajectory fractions (0 to 1) at which to take the screenshots")
    ap.add_argument("--shots-dir", default="shots", help="where screenshots are written")
    args = ap.parse_args()

    cfg = Config.from_toml(args.config) if args.config else Config()
    gui = not (args.headless or args.snapshot)
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, gui=gui))

    print(cfg.summary())
    for msg in cfg.warnings():
        print("WARNING:", msg)

    if args.mode == "world":
        run_world(cfg, Path(args.snapshot) if args.snapshot else None)
    else:
        run_episode(cfg, args)


if __name__ == "__main__":
    main()
