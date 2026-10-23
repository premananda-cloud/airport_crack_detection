"""Simulation entry point. Run from the project root.

    uv run python -m src.sim.run_sim                         # world only: just look at the runway
    uv run python -m src.sim.run_sim --mode hover            # RAPTOR holds the start point (sanity test)
    uv run python -m src.sim.run_sim --mode fly              # RAPTOR flies the lawnmower coverage path
    uv run python -m src.sim.run_sim --mode hover --headless # no window, prints the numbers only
    uv run python -m src.sim.run_sim --snapshot out.png      # world only, one headless image

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
from src.environment.runway import Runway, _write_png, connect


def snapshot(cfg: Config, client: int, path: Path, width: int = 1280, height: int = 720) -> None:
    """Render one oblique view with PyBullet's CPU renderer (works without a display)."""
    r = cfg.runway
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=max(r.length_m * 0.7, 15.0),
        yaw=r.heading_deg - 90.0 + 25.0,
        pitch=-30.0, roll=0.0, upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.1,
                                        farVal=2000.0, physicsClientId=client)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    _write_png(path, np.ascontiguousarray(rgb))
    print(f"saved {path}")


def run_world(cfg: Config, snap: Path | None) -> None:
    client = connect(cfg)
    Runway(cfg).build(client)
    if snap is not None:
        snapshot(cfg, client, snap)
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


def run_episode(cfg: Config, mode: str, hover_s: float) -> None:
    # imported here so the world-only mode works without gymnasium / the policy installed
    from src.planner.coverage import hover_trajectory, plan_coverage
    from src.policy.load_policy import RaptorPolicy
    from src.wrapper.gym_wrapper import RunwayInspectionEnv

    traj = hover_trajectory(cfg, hover_s) if mode == "hover" else plan_coverage(cfg)
    print(traj.summary())

    policy = RaptorPolicy()
    print("policy:", policy.description)
    env = RunwayInspectionEnv(cfg, traj)
    obs, info = env.reset()
    policy.reset()

    dt = 1.0 / cfg.sim.control_hz
    errs, steps, t0 = [], 0, time.perf_counter()
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            steps += 1
            errs.append(info["pos_error_m"])
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
        run_episode(cfg, args.mode, args.hover_seconds)


if __name__ == "__main__":
    main()
