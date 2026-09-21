"""The 3-D fly body (NeuroMechFly v2 / FlyGym 2.x, MuJoCo physics).

A legs-actuated walking fly with two extra actuated joints for the proboscis (rostrum pitch, haustellum pitch), so the
brain can make it walk, turn and extend the proboscis ("eat"). Walking itself is FlyGym's hybrid CPG + rule controller,
NOT the connectome's nerve cord: the brain only supplies a 2-element descending drive (left, right) and the eat signal.
"""
import numpy as np, mujoco
from flygym import Simulation
from flygym.anatomy import (BodySegment, ContactBodiesPreset, JointPreset, AxisOrder, Skeleton, ActuatedDOFPreset,
                            JointDOF, RotationAxis, PASSIVE_TARSAL_LINKS)
from flygym.compose import FlatGroundWorld, ActuatorType, NeuroMechFly, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import HybridTurningController, HybridControllerObservation, PreprogrammedSteps

PROBOSCIS_DOFS = [JointDOF(parent=BodySegment("c_head"), child=BodySegment("c_rostrum"), axis=RotationAxis.PITCH),
                  JointDOF(parent=BodySegment("c_rostrum"), child=BodySegment("c_haustellum"), axis=RotationAxis.PITCH)]
PROBOSCIS_EXTENDED = np.array([-1.2, -1.0])    # radians added to neutral when fully extended (negative pitch = down/out)


def make_fly(name="fly", colorize=True):
    neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL)
    fly = NeuroMechFly(name=name)
    joints = fly.add_joints(skeleton, neutral_pose=neutral, stiffness=0.05, damping=0.06)
    for dof, j in joints.items():
        if dof.child.link in PASSIVE_TARSAL_LINKS:
            j.stiffness[0] = 7.5; j.damping[0] = 1e-2
        elif dof not in PROBOSCIS_DOFS and not dof.child.is_leg():
            j.stiffness[0] = 5.0; j.damping[0] = 0.5          # hold head, wings, antennae still
    legs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(list(legs) + PROBOSCIS_DOFS, ActuatorType.POSITION, neutral_input=neutral, kp=45.0,
                      forcerange=(-65.0, 65.0))
    fly.add_leg_adhesion(gain=40.0)
    if colorize: fly.colorize()
    return fly, neutral


class FlyBody:
    """sim + walking controller + proboscis. step(left, right, eat) advances one physics step."""
    def __init__(self, camera="side", res=(360, 480), playback_speed=0.2, fps=30, sugar_at=None):
        self.fly, neutral = make_fly()
        cams = {"side": dict(pos_offset=(-0.5, -7.5, 0.0), rotation=Rotation3D("euler", (1.57, 0, 0)), fovy=35.0),
                "top":  dict(pos_offset=(0, 0, 12.0), rotation=Rotation3D("euler", (0, 0, 0)), fovy=35.0),
                "head": dict(pos_offset=(3.4, -3.0, 0.2), rotation=Rotation3D("euler", (1.40, 0, 0.85)), fovy=26.0)}
        cam = self.fly.add_tracking_camera(name="cam", **cams[camera])
        self.world = FlatGroundWorld()
        self.world.add_fly(self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                           bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
        if sugar_at is not None:                        # a visual-only sugar drop (no collision)
            self.world.mjcf_root.worldbody.add_geom(name="sugar", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.75, 0.6, 0.14],
                                                    pos=[float(sugar_at[0]), float(sugar_at[1]), 0.12],
                                                    rgba=[1.0, 0.8, 0.15, 1.0], contype=0, conaffinity=0)
        self.sim = Simulation(self.world)
        self.sim.set_renderer([cam], camera_res=res, playback_speed=playback_speed, output_fps=fps)
        self.steps = PreprogrammedSteps()
        self.all_dofs = self.fly.get_actuated_jointdofs_order("position")
        self.leg_dofs = [d for d in self.all_dofs if d not in PROBOSCIS_DOFS]
        self.prob_idx = [self.all_dofs.index(d) for d in PROBOSCIS_DOFS]
        self.leg_idx = [self.all_dofs.index(d) for d in self.leg_dofs]
        self.prob_neutral = np.array([neutral.joint_angles_lookup_rad.get(d, 0.0) for d in PROBOSCIS_DOFS])
        self.ctrl = HybridTurningController(timestep=self.sim.timestep, preprogrammed_steps=self.steps,
                                            output_dof_order=self.leg_dofs)
        self.thorax = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self.reset()

    def reset(self, seed=0):
        self.sim.reset(); self.ctrl.reset(seed=seed)
        q = np.zeros(len(self.all_dofs)); q[self.leg_idx] = self.steps.default_pose_by_dof_order(self.leg_dofs)
        q[self.prob_idx] = self.prob_neutral
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, q)
        self.sim.set_leg_adhesion_states(self.fly.name, np.ones(6, bool))
        self.sim.warmup()
        self.eat_level = 0.0

    def step(self, left=1.0, right=1.0, eat=0.0, render=True):
        """left/right: descending drive in [0, 1.2] (0 = stand still). eat in [0, 1]: proboscis extension target."""
        obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
        act = self.ctrl.step(np.array([left, right], float), obs)
        self.eat_level += 0.02 * (eat - self.eat_level)                 # smooth extension / retraction
        q = np.zeros(len(self.all_dofs)); q[self.leg_idx] = act.joint_angles
        q[self.prob_idx] = self.prob_neutral + self.eat_level * PROBOSCIS_EXTENDED
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, q)
        if act.adhesion_onoff is not None: self.sim.set_leg_adhesion_states(self.fly.name, act.adhesion_onoff)
        self.sim.step()
        if render: self.sim.render_as_needed()

    @property
    def position(self):
        return self.sim.get_body_positions(self.fly.name)[self.thorax]

    @property
    def heading(self):
        """yaw angle of the thorax in radians (0 = +x)."""
        w, x, y, z = self.sim.get_body_rotations(self.fly.name)[self.thorax]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @property
    def timestep(self): return self.sim.timestep

    def save_video(self, path): self.sim.renderer.save_video(path)
