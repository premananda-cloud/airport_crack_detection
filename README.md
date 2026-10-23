# Airport Crack Detection: autonomous flight

Simulation and control for a drone that flies over an airport runway so a vision model can find cracks and damage. The project has two halves, and this repository is the **flight** half:

1. **Autonomous flight** (this repo): a PyBullet runway world, a classical lawnmower coverage planner, and the pretrained [RAPTOR](https://github.com/rl-tools/raptor) foundation policy as the motor-level controller.
2. **Crack detection** (separate): a perception module that consumes images taken during the flight.

There is no learning in this repo. RAPTOR is used as it is; it is a motor-level tracking controller, not a planner, so the planner feeds it a reference trajectory and it returns motor commands.

## Status

| Part | State |
|---|---|
| Runway world (ground, asphalt, markings, GPS frame, inspection volume) | working, checked visually |
| RAPTOR loader, quadrotor model, coverage planner, Gymnasium wrapper | written; hover and fly modes reported working by the author |
| Follow camera and screenshots (`src/sim/camera.py`) | written, not yet confirmed on a run |
| Coverage grid, crack-detection interface, replanner | not started (see [Roadmap](#roadmap)) |

Camera, GPS origin, drone parameters and several mission limits are **placeholders**. They are listed under [Configuration](#configuration).

## Setup

Python 3.11 or newer (the config loader uses `tomllib`). The project is managed with [uv](https://docs.astral.sh/uv/).

```
uv sync
uv add gymnasium        # needed for hover and fly modes
```

Dependencies: `numpy`, `pybullet`, `foundation-policy` (which brings `h5py`), and `gymnasium`. No GPU and no SciPy are needed.

PyBullet 3.2.7 compiles from source and did not build on Python 3.13 in one test environment. If installation fails there, use Python 3.12.

## Run

Always from the project root.

```
uv run python -m src.sim.run_sim                          # world only: look at the runway
uv run python -m src.sim.run_sim --mode hover             # RAPTOR holds the start point
uv run python -m src.sim.run_sim --mode fly               # RAPTOR flies the lawnmower coverage path
uv run python -m src.sim.run_sim --mode fly --headless    # no window, prints the numbers only
uv run python -m src.sim.run_sim --snapshot out.png       # world only, one headless image
```

Run **hover first**. The drone must hold its start point without oscillation or drift; position error is printed every second and should settle near zero. If it does not, see [Assumptions to verify](#assumptions-to-verify).

Camera and screenshots:

```
uv run python -m src.sim.run_sim --mode fly --follow 25
uv run python -m src.sim.run_sim --mode fly --headless --shots 3,12,25
```

- `--follow D`: in the GUI the camera tracks the drone from D metres away. While it is on it overrides the mouse every step, so orbit and zoom snap back.
- `--shots D1,D2,..`: one PNG per distance, taken with the camera that far from the drone, written to `shots/p<progress>_d<distance>m.png`.
- `--shot-progress F1,F2,..`: when to take them, as fractions of the trajectory (default `0.1,0.5,0.9`). `--shots-dir` changes the folder.
- Screenshots do not include the yellow cuboid and axes; only the GUI draws those.

Override any config value with a TOML file:

```
uv run python -m src.sim.run_sim --mode fly my_run.toml
```

## Layout

```
src/
  config.py                 all inputs (dataclasses with defaults) and derived values
  environment/
    geo.py                  GPS <-> world <-> runway coordinate frames
    runway.py               the static scene: ground, asphalt slab, markings, debug overlay
  planner/
    coverage.py             lane layout, U-turn path, speed profile, 100 Hz trajectory
  policy/
    load_policy.py          RAPTOR loader: 22 numbers in, 4 motor actions out
  sim/
    quadrotor.py            quadrotor rigid body with four motors
    camera.py               follow camera and screenshots
    run_sim.py              entry point
  wrapper/
    gym_wrapper.py          Gymnasium environment tying it together
docs/
  PLAN.md                   architecture and build order
  MATH.md                   derivations and numbers behind the parameter choices
```

The split is deliberate: `config.py` is what you change between runs, and the rest of the code reads from it and hardcodes none of those values.

## Coordinate frames

| Frame | Axes | Used for |
|---|---|---|
| GPS | latitude, longitude, altitude (WGS84) | real-world anchoring |
| world | ENU: x east, y north, z up, origin at the runway centre | PyBullet |
| runway | x along the runway, y to the left, z up, same origin | planner, policy inputs, future coverage grid and detections |

The runway frame is the world frame rotated about z by the runway heading. Heading is clockwise from north, so 90° (the default) is due east and the runway and world frames coincide. The runway top surface is at z = 0. `Runway.geo` converts between all three. GPS to world goes through ECEF with the exact WGS84 ellipsoid, so the round trip has no flat-earth error.

## Configuration

Everything is in `src/config.py`. Sections and the values worth knowing:

| Section | Notes |
|---|---|
| `geo` | GPS point at the **runway centre**. **Placeholder** (0, 0): set a surveyed point. |
| `runway` | 100 × 6 m, heading 90°, procedural asphalt and markings. A scaled-down test runway, not a real one (real: 2,500 to 4,000 m long, 45 to 60 m wide). |
| `inspection` | altitude 15 m; the inspection cuboid extends 5 m past the runway and up to 25 m. |
| `camera` | **Placeholder** 1920×1080, 55° horizontal FOV. The real camera is not chosen. |
| `mission` | overlap 20%, 3 m/s, 1.5 m/s² lateral and along-track. |
| `drone` | **Placeholder** 1.2 kg, 0.16 m arm, thrust-to-weight 2.5. Inside RAPTOR's trained range. |
| `sim` | physics 500 Hz, control 100 Hz, tracking-error limit 5 m. |

Derived values are computed from these, so changing the camera or the altitude never means recalculating by hand: footprint, ground sample distance, lane count and pitch, and the inspection volume. With the defaults the footprint is 15.6 × 8.8 m (GSD about 8.1 mm/px) and **one lane covers the whole 6 m width**, so the lane-end turn logic is not exercised. `Config.warnings()` says so at startup. To test multi-lane behaviour, lower the altitude or widen the runway; `docs/MATH.md` section 5 has the numbers.

## How it works

**Planner** (`planner/coverage.py`). Lane centres come from the footprint and overlap: `swath = footprint × (1 − overlap)`, `lanes = ceil(width / swath)`, `pitch = width / lanes`. Lanes run along the runway and alternate direction, joined by semicircular U-turns of radius `pitch / 2`. Speed along the path is limited to `v_max` and to `sqrt(a_lat / curvature)`, then forward and backward passes enforce the along-track acceleration limit, starting and ending at rest. The result is integrated to time and sampled at the control rate, giving a reference position and velocity per step.

**Policy** (`policy/load_policy.py`). RAPTOR is a recurrent network that takes 22 numbers and returns 4 motor commands in [-1, 1]. It has no look-ahead: it sees only the error to the reference at the current step.

| Index | Content |
|---|---|
| 0–2 | position error `p − p_ref` |
| 3–11 | rotation matrix, body to frame, row-major |
| 12–14 | velocity error `v − v_ref` |
| 15–17 | angular velocity, body frame (FLU) |
| 18–21 | previous action |

Motor order is front-right, back-right, back-left, front-left, spinning CCW, CW, CCW, CW. Call `reset()` at every episode start.

**Environment** (`wrapper/gym_wrapper.py`). `RunwayInspectionEnv` is a Gymnasium environment: the action is RAPTOR's 4 raw outputs, the observation is the 22 numbers above. The drone is spawned at the start of the trajectory, nose along the runway. An episode ends (terminated) if the drone leaves the inspection volume, crashes, flips, or strays more than `max_tracking_error_m` from the reference. It ends (truncated) when the trajectory is finished plus `end_hold_s`. The reward is a placeholder, minus the tracking error.

**Drone model** (`sim/quadrotor.py`). Rigid body with a first-order motor lag and thrust proportional to the square of the command. No drag, ground effect or wind yet. External forces are applied before every physics step, because PyBullet clears them after each one.

## Assumptions to verify

The RAPTOR paper and package leave three things unstated. The code makes these choices, and the hover test is how to find out if one is wrong:

1. **Frame of position and velocity.** The paper says only that they are offsets from the trajectory. Here they are given in the runway frame.
2. **Action to thrust.** The package ends in `tanh`, so outputs are in [-1, 1]. Here a motor command is `(a + 1) / 2` and thrust is proportional to its square.
3. **Yaw torque sign.** Here a CCW rotor pushes the body clockwise.

Likely symptoms: a wrong mapping in (2) makes the drone climb or fall in hover; a wrong sign in (3) makes it spin about its vertical axis; a wrong frame in (1) makes it drift sideways.

Also unverified: the 100 Hz control rate (the paper does not state one) and the real runway dimension ranges, which come from general knowledge and a prior discussion rather than a cited source.

## Roadmap

- Coverage grid over the runway footprint, marked as the camera footprint passes, giving a coverage metric.
- Detection interface: per-frame `(x, y)` in the runway frame, confidence in [0, 1] and a class label, mapped to coverage cells. Suggested additions: timestamp, drone pose at capture, and detection size.
- Camera tilt: a fixed nadir camera shifts its footprint by about `altitude × a / g` when the drone accelerates (3 m at 15 m altitude and 2 m/s²), so use a gimbal or correct detections with the measured attitude.
- Real camera profile, then recompute footprint, GSD and lanes.
- Wind and sensor noise.
- AI replanner, only if detections drive non-trivial re-routing.

## References

- RAPTOR repository: https://github.com/rl-tools/raptor
- Eschmann et al., *RAPTOR: A Foundation Policy for Quadrotor Control*, arXiv 2509.11481: https://arxiv.org/abs/2509.11481
- `foundation-policy` on PyPI, the package that ships the RAPTOR checkpoint.
