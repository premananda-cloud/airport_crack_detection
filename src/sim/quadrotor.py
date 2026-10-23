"""Quadrotor rigid body for PyBullet, driven by four normalised motor commands.

Body frame is FLU (x forward, y left, z up), which is what PyBullet's base frame is here.
Rotor order follows RAPTOR: front-right, back-right, back-left, front-left.

Model (deliberately simple, all numbers come from DroneCfg):
    motor lag    u_eff += (1 - exp(-dt / tau)) * (u - u_eff)        u in [0, 1]
    thrust       f_i = thrust_max * u_eff_i^2                        thrust_max = T/W * m * g / 4
    force        F  = (0, 0, sum f_i) in the body frame
    torque       roll/pitch from r_i x f_i, yaw from -spin_i * moment_constant * f_i
                 (a CCW-spinning rotor pushes the body clockwise, i.e. negative z)
No aerodynamic drag, no ground effect, no wind yet.

PyBullet clears external forces after every stepSimulation, so apply() must be called before
EVERY physics step, not once per control step.

Front rotors are drawn red, back rotors blue, so the heading is visible in the GUI.
"""
from __future__ import annotations

import math

import numpy as np
import pybullet as p

from src.config import Config

__all__ = ["Quadrotor"]

_LINK_MASS_KG = 0.01          # each visual rotor link; compensated for in the base inertia
_BASE_HALF_M = [0.07, 0.07, 0.025]
_ROTOR_RADIUS_M = 0.06
_ROTOR_LENGTH_M = 0.006
_FRONT = [0.9, 0.15, 0.1, 1.0]
_BACK = [0.2, 0.4, 0.9, 1.0]


class Quadrotor:
    def __init__(self, cfg: Config, client: int):
        self.client = client
        self.d = cfg.drone
        d = self.d
        g = cfg.sim.gravity_mps2
        self.dt = 1.0 / cfg.sim.physics_hz
        self.thrust_max = d.thrust_to_weight * d.mass_kg * g / 4.0       # N per rotor
        self.alpha = 1.0 - math.exp(-self.dt / d.motor_tau_s)
        s = d.arm_m / math.sqrt(2.0)
        # FR, BR, BL, FL in FLU
        self.rotor_pos = np.array([[s, -s, 0.0], [-s, -s, 0.0], [-s, s, 0.0], [s, s, 0.0]])
        self.spin = np.asarray(d.spin, dtype=float)
        self.u_eff = np.full(4, d.hover_command)
        self.body = self._create_body()

    # ------------------------------------------------------------------ build #
    def _create_body(self) -> int:
        c, d, rp = self.client, self.d, self.rotor_pos
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=_BASE_HALF_M, physicsClientId=c)
        vis_base = p.createVisualShape(p.GEOM_BOX, halfExtents=_BASE_HALF_M,
                                       rgbaColor=[0.12, 0.12, 0.15, 1.0], physicsClientId=c)
        vis_rotors = [p.createVisualShape(p.GEOM_CYLINDER, radius=_ROTOR_RADIUS_M,
                                          length=_ROTOR_LENGTH_M, rgbaColor=colour, physicsClientId=c)
                      for colour in (_FRONT, _BACK, _BACK, _FRONT)]

        # the four rotor links carry a little mass: take their share out of the base inertia
        link_inertia = _LINK_MASS_KG * np.array([(rp[:, 1] ** 2).sum(),
                                                 (rp[:, 0] ** 2).sum(),
                                                 (rp[:, 0] ** 2 + rp[:, 1] ** 2).sum()])
        base_inertia = np.array([d.ixx_kgm2, d.iyy_kgm2, d.izz_kgm2]) - link_inertia
        if (base_inertia <= 0).any():
            raise ValueError("[drone] inertia is too small for the rotor visual links")

        body = p.createMultiBody(
            baseMass=d.mass_kg - 4 * _LINK_MASS_KG,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis_base,
            basePosition=[0.0, 0.0, 1.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=[_LINK_MASS_KG] * 4,
            linkCollisionShapeIndices=[-1] * 4,
            linkVisualShapeIndices=vis_rotors,
            linkPositions=rp.tolist(),
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * 4,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * 4,
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * 4,
            linkParentIndices=[0] * 4,            # 0 = the base
            linkJointTypes=[p.JOINT_FIXED] * 4,
            linkJointAxis=[[0.0, 0.0, 1.0]] * 4,
            physicsClientId=c)
        p.changeDynamics(body, -1, localInertiaDiagonal=base_inertia.tolist(), physicsClientId=c)
        for link in range(-1, 4):                 # PyBullet adds damping by default; remove it
            p.changeDynamics(body, link, linearDamping=0.0, angularDamping=0.0, physicsClientId=c)
        return body

    # ------------------------------------------------------------------ state #
    def reset(self, pos_world, yaw_rad: float = 0.0) -> None:
        """Place the drone level at pos_world, at rest, with rotors already at the hover command."""
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])
        p.resetBasePositionAndOrientation(self.body, list(pos_world), orn, physicsClientId=self.client)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self.u_eff[:] = self.d.hover_command

    def state(self):
        """(position_world, quaternion_xyzw, linear_velocity_world, angular_velocity_world)."""
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        lin, ang = p.getBaseVelocity(self.body, physicsClientId=self.client)
        return np.array(pos), tuple(orn), np.array(lin), np.array(ang)

    # ---------------------------------------------------------------- dynamics #
    def apply(self, u) -> None:
        """Advance the motors one physics step toward command u (4, in 0..1) and apply the wrench."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        self.u_eff += self.alpha * (u - self.u_eff)
        f = self.thrust_max * self.u_eff ** 2                              # N per rotor

        force_b = np.array([0.0, 0.0, f.sum()])
        torque_b = np.array([
            (self.rotor_pos[:, 1] * f).sum(),                              # r x (0,0,f) = (y f, -x f, 0)
            -(self.rotor_pos[:, 0] * f).sum(),
            -(self.spin * self.d.moment_constant_m * f).sum(),
        ])

        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)       # body -> world
        p.applyExternalForce(self.body, -1, (rot @ force_b).tolist(), pos, p.WORLD_FRAME,
                             physicsClientId=self.client)
        p.applyExternalTorque(self.body, -1, (rot @ torque_b).tolist(), p.WORLD_FRAME,
                              physicsClientId=self.client)
