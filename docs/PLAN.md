# Plan for simulation

The autonomous drone flight in simulation require the following things.
1) Motor level RL policy
2) Misiion planner/path finder
3) simulation woRLd


## 1 Motor level RL policy

Motor policy is trained RL policy that find normalised mortor command, calculated from error relative to trejecotory and position.
Instead of training from scratch we are using RAPTOR [Foundation Policy for Quadrotor Control](https://github.com/rl-tools/raptor.git)

what it is not. It is not a planner. It is a motor-level tracking controller that expects you to feed it a target trajectory (or the error relative to one) and returns normalized motor commands. The README is explicit:


## 2 Mission Planner/Path finder

Classical coverage path planning. AI replanner deferred.

Two parts.

Coverage planner. Boustrophedon (lawnmower) over the runway rectangle. Along-runway sweep. Lane spacing from camera footprint at inspection altitude with overlap. Rounded turn waypoints at lane ends. Outputs ordered waypoint list.

Trajectory generator. Spline through waypoints. Samples at 100 Hz to produce target position and target velocity. Respects velocity and acceleration limits within RAPTOR tracking envelope.

Coverage grid. 2-D grid over runway footprint. Cells marked inspected when camera footprint covers them. Gives coverage metric and later serves as state for AI replanner

## 3 Simulation World
This gives digital simulation of the drone operating environment. we are planning "Pybullet" "gymnasium".

## 4 Build order
PyBullet scene. Ground plane, cuboid inspection volume, quadrotor asset.

RAPTOR loop at 100 Hz. FLU conversion and motor mapping.

Hover sanity check. Fixed target position, zero target velocity. Pass only when altitude held without oscillation or drift.

Runway geometry and lawnmower waypoint generator.

Spline trajectory generator at 100 Hz. Critical test: RAPTOR tracks spline smoothly.

Coverage grid and coverage metric.

Detection interface stub with synthetic detections.

AI replanner, only if detections drive non-trivial re-routing.


## 5 Open item
5 Open items
Runway dimensions. Length x width, or scaled test runway.

Inspection altitude.

Camera FOV and footprint at altitude.

Coverage overlap. 10-30 percent, 20 default.

Max cruise speed and max lateral acceleration within RAPTOR tracking.

Runway treated as clean rectangle for v1 or thresholds and taxiway junctions excluded now.

Crack detection interface. What the module emits per frame. How detections map to coverage cells and re-routing.
