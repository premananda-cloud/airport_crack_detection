# Simulation Math Reference

Companion to PLAN.md. Every number below was recomputed in Python from the formulas shown, not copied from the earlier DeepSeek discussion. Where DeepSeek's numbers did not hold up, that is flagged.

## 0. Summary of findings

1. **The 1:40 scale claim is only partly true.** Length scales 25x to 40x, width 7.5x to 10x, and altitude is not scaled at all. The sim runway is much "fatter" than a real one (aspect 16.7 vs about 50 to 80).
2. **At 15 m altitude one lane covers the whole 6 m runway.** The lawnmower planner, turn waypoints and lane spacing from PLAN.md are never exercised. A real 45 to 60 m runway at 15 m needs 4 to 5 lanes.
3. **2 to 5 m/s cruise is not compatible with the image quality needed at 15 m.** At 5 m/s blur is 1.3 px at 1/1000 s exposure. About 2 m/s is the limit for 0.5 px at 1/1000 s.
4. **1 to 2 m/s^2 lateral acceleration forces slow turns.** A U-turn between lanes 3 m apart is limited to about 1.2 to 1.7 m/s, so the planner must decelerate before every lane end.
5. **Tilt moves the camera footprint** by up to 3 m at 15 m altitude with a fixed downward camera. Either use a gimbal or correct detections using attitude.

## 1. Notation and assumptions

| Symbol | Meaning | Value used |
|---|---|---|
| L, W | runway length, width (sim) | 100 m, 6 m |
| h | inspection altitude | 15 m (DeepSeek) |
| Wc, Lc | camera footprint across / along track at h | 15.6 m, 11.7 m (DeepSeek) |
| o | overlap fraction | 0.20 |
| g | gravity | 9.81 m/s^2 |
| N_px | pixels across the footprint | 4000 (**assumed**, 4000 x 3000 sensor) |

Assumptions that are mine or DeepSeek's and not verified:

- The footprint of 15.6 x 11.7 m at 15 m is DeepSeek's "typical drone camera". It back-solves to a field of view of 54.95 deg (horizontal) x 42.61 deg (vertical), a 4:3 sensor. Replace with the real camera spec once your teammate picks one.
- The sensor is 4000 px wide. GSD and blur scale linearly with this.
- Camera is nadir (pointing straight down) and rigidly mounted unless stated.
- The real runway ranges (2500 to 4000 m long, 45 to 60 m wide) match my general knowledge but I did not look them up. I also did not verify the RAPTOR tracking envelope quoted in the discussion. Check both against the RAPTOR README and an airport source.

## 2. Scale

Scale factors (real / sim):

- Length: 2500 / 100 = 25, 4000 / 100 = 40.
- Width: 45 / 6 = 7.5, 60 / 6 = 10.
- Aspect ratio L/W: real 4000 / 50 = 80, sim 100 / 6 = 16.7.
- Altitude: 15 m in sim. The 15 to 60 m real range comes from DeepSeek and is not scaled, so the camera sees a much larger fraction of the sim runway than it would in reality.

Consequence: what matters for planner testing is the **number of lanes**, not the geometric scale. See section 5.

The cuboid inspection volume must be taller than the inspection altitude plus a margin (for example 15 m altitude needs a cuboid height of at least 20 m).

## 3. Camera geometry

For a nadir camera with field of view theta at altitude h:

    footprint = 2 * h * tan(theta / 2)

From the footprint above, the footprint per metre of altitude is 1.04 m (across) and 0.78 m (along).

So the footprint at altitude h is Wc(h) = 1.04 h and Lc(h) = 0.78 h.

## 4. Image quality

### Ground sample distance

    GSD = Wc / N_px = 1.04 h / 4000

| h (m) | GSD (mm/px) |
|---|---|
| 5 | 1.30 |
| 7 | 1.82 |
| 10 | 2.60 |
| 15 | 3.90 |

Rule of thumb (not verified, ask the vision teammate): a crack should span at least about 2 px to be detectable. At 15 m (3.9 mm/px) cracks narrower than roughly 8 mm are marginal. Lower altitude buys resolution directly.

### Motion blur

    blur_px = v * t_exposure / GSD

At h = 15 m (GSD 3.9 mm/px):

| v (m/s) | 1/500 s | 1/1000 s | 1/2000 s |
|---|---|---|---|
| 2 | 1.03 px | 0.51 px | 0.26 px |
| 3 | 1.54 px | 0.77 px | 0.38 px |
| 5 | 2.56 px | 1.28 px | 0.64 px |

For a 0.5 px blur budget: v_max = 0.5 * GSD / t_exp, which gives 1.95 m/s at 1/1000 s and 3.9 m/s at 1/2000 s. Either cap cruise speed near 2 m/s or specify a 1/2000 s (or faster) shutter. The vibration and attitude jitter of the real drone add to this.

## 5. Coverage lanes

Number of lanes and lane pitch for a runway of width W:

    swath    = Wc(h) * (1 - o)
    n_lanes  = ceil(W / swath)
    pitch    = W / n_lanes
    centres  : y_i = -W/2 + pitch * (i + 0.5),   i = 0 .. n_lanes-1
    actual overlap = 1 - pitch / Wc

Rounding n up means the real overlap is usually larger than 20%.

Sim runway, W = 6 m:

| h (m) | footprint across (m) | swath (m) | lanes | pitch (m) | actual overlap |
|---|---|---|---|---|---|
| 5 | 5.20 | 4.16 | 2 | 3.0 | 42% |
| 7 | 7.28 | 5.82 | 2 | 3.0 | 59% |
| 10 | 10.40 | 8.32 | 1 | 6.0 | 42% |
| 15 | 15.60 | 12.48 | 1 | 6.0 | 62% |

At h = 15 m with one centred lane, there is 4.8 m of lateral slack each side (3.24 m if you insist on keeping the 20% margin).

Real runway at h = 15 m: 4 lanes for 45 m wide, 5 lanes for 60 m wide. At 30 m: 2 and 3 lanes.

To keep the sim planner honest, pick one of these:

- **A. Lower the altitude.** On a 6 m runway: 2 lanes needs h <= 3.6 m, 3 lanes needs h <= 2.4 m, 4 lanes needs h <= 1.8 m. Very low altitude gives good GSD but is risky near the ground and changes the flight regime.
- **B. Widen the sim runway at 15 m.** Two lanes: W between 12.5 and 25 m. Three lanes: 25 to 37.4 m. Four lanes: 37.4 to 49.9 m. (At h = 10 m: 2 lanes for 8.3 to 16.6 m, 3 lanes for 16.6 to 25 m.)
- **C. Keep 6 m x 100 m at 15 m** and accept a single lane. Then test the lawnmower and turn logic separately on a wider test rectangle.

Along-track: the drone centre only needs to span x from Lc/2 to L - Lc/2 to cover the runway ends (5.85 to 94.15 m at h = 15 m) if the camera is exactly nadir. Using 0 to L is simply conservative.

Frame spacing along track, assuming the same 20% forward overlap (a decision, stitching often wants more): 11.7 * 0.8 = 9.36 m. Frame interval = 9.36 / v: 4.68 s at 2 m/s, 3.12 s at 3 m/s, 1.87 s at 5 m/s.

## 6. Acceleration, tilt and camera shift

A quadrotor accelerating horizontally at a tilts by:

    tilt = atan(a / g)
    thrust / weight = sqrt(1 + (a/g)^2)

| a (m/s^2) | tilt (deg) | thrust/weight | footprint shift at h = 15 m |
|---|---|---|---|
| 1 | 5.82 | 1.005 | 1.53 m |
| 1.5 | 8.69 | 1.012 | 2.29 m |
| 2 | 11.52 | 1.021 | 3.06 m |
| 3 | 17.00 | 1.046 | not needed |

The shift of a fixed camera is h * tan(tilt) = h * a / g. At a = 2 m/s^2 this is 3.06 m, over half the along-track half-footprint (5.85 m). The coverage grid and detection coordinates must either use a gimbal or project the camera ray using the measured attitude. Thrust headroom is not a concern at these accelerations.

## 7. Turns and trajectory limits

Constant-speed turn radius: r = v^2 / a_lat.

| v (m/s) | a_lat (m/s^2) | r (m) |
|---|---|---|
| 5 | 2 | 12.5 |
| 5 | 1 | 25.0 |
| 3 | 1.5 | 6.0 |

A 12.5 m radius does not fit a 6 m wide runway, so cruise speed cannot be held through lane-end turns.

For a semicircular U-turn between adjacent lanes with pitch p (radius p/2), the maximum speed is:

    v_turn = sqrt(a_lat * p / 2)

| pitch p (m) | a = 1 | a = 1.5 | a = 2 |
|---|---|---|---|
| 3.0 | 1.22 | 1.50 | 1.73 |
| 4.17 | 1.44 | 1.77 | 2.04 |
| 6.0 | 1.73 | 2.12 | 2.45 |

(m/s). Deceleration distance from cruise to turn speed, d = (v^2 - v_t^2) / (2 a): 3 to 1.73 m/s at 1.5 m/s^2 takes 2.0 m, 5 to 1.73 m/s takes 7.34 m, 3 to 1.22 m/s at 1.0 m/s^2 takes 3.76 m.

Design rule for the trajectory generator: after splining, clamp speed with the local curvature, v(s) <= sqrt(a_lat / kappa(s)), then run a forward-backward pass for the along-track acceleration limit. Sampling at 100 Hz gives a 2 cm step at 2 m/s and 5 cm at 5 m/s.

## 8. Mission time (100 m lane, trapezoidal speed profile)

| v_cruise (m/s) | a (m/s^2) | accel distance each end | total time |
|---|---|---|---|
| 2 | 1.0 | 2.0 m | 52.0 s |
| 3 | 1.5 | 3.0 m | 35.3 s |
| 5 | 1.5 | 8.33 m | 23.3 s |

For scale: one 4000 m real lane at 3 m/s is 22.2 min. A real runway at 15 m altitude with 4 to 5 lanes is therefore well over an hour of flight, which matters for battery and for the later AI replanner.

## 9. Coverage grid

Grid cell count for a 100 m x 6 m runway: 1.0 m cells give 600, 0.5 m cells give 2400 (200 x 12), 0.25 m cells give 9600. Suggested start: 0.5 m.

Marking rule (a decision): mark a cell inspected only when the whole cell lies inside the camera footprint rectangle (rotated by yaw and shifted by the tilt effect in section 6). Cell centre only is cheaper but overstates coverage at footprint edges.

Coverage metric: inspected cells / total cells. With the lane design in section 5 it should reach 100% by construction, which makes a good test.

## 10. Detection interface

The proposed contract (x, y in the runway frame, confidence 0 to 1, class string) is enough to flag a cell. Suggested additions:

- Frame timestamp and drone pose at capture, so errors can be traced.
- Size (length, width in metres) or a bounding box, because one (x, y) point cannot say how much of a cell is affected.
- An explicit definition of the runway frame: origin at one corner or the centre, x along the runway, y across, z up.

Detection position error is dominated by pose error (section 6), so quote the expected accuracy with the contract.

## 11. Decisions still open

1. Altitude and runway width (options A, B, C in section 5).
2. Camera model: real FOV and pixel count, then recompute sections 3 to 5.
3. Cruise speed and shutter speed (section 4), then lane-end turn speed (section 7).
4. Fixed camera with attitude compensation vs gimbal (section 6).
5. Forward overlap, if different from 20%.
6. Cell size and marking rule (section 9).
