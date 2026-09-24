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
    def __init__(self, camera="side", res=(360, 480), playback_speed=0.2, fps=30, sugar_at=None, bitter_at=None, water_at=None, buffer=True,
                 ctrl_every=1):
        """ctrl_every: run the walking controller once every this many physics steps and hold its commands in between.
        MuJoCo still integrates every 0.1 ms step; only the leg-pattern controller (Python, about 6x the cost of the
        physics itself) runs less often. 10 = a 1 kHz controller, plenty for walking."""
        self.ctrl_every = max(1, int(ctrl_every)); self._k = 0; self._held = None
        self.fly, neutral = make_fly()
        cams = {"side": dict(pos_offset=(-0.5, -7.5, 0.0), rotation=Rotation3D("euler", (1.57, 0, 0)), fovy=35.0),
                "top":  dict(pos_offset=(0, 0, 12.0), rotation=Rotation3D("euler", (0, 0, 0)), fovy=35.0),
                "head": dict(pos_offset=(3.4, -3.0, 0.2), rotation=Rotation3D("euler", (1.40, 0, 0.85)), fovy=26.0)}
        cam = self.fly.add_tracking_camera(name="cam", **cams[camera])
        self.world = FlatGroundWorld()
        # a dark gradient sky behind the fly (the scene has none, so the renderer shows plain white); looks only
        for t in self.world.mjcf_root.textures:
            if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT; t.rgb1 = [0.20, 0.24, 0.34]; t.rgb2 = [0.03, 0.035, 0.06]
                t.width = t.height = 256
        self.world.add_fly(self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                           bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
        if sugar_at is not None:                        # a visual-only sugar drop (no collision)
            self.world.mjcf_root.worldbody.add_geom(name="sugar", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.75, 0.6, 0.14],
                                                    pos=[float(sugar_at[0]), float(sugar_at[1]), 0.12],
                                                    rgba=[1.0, 0.8, 0.15, 1.0], contype=0, conaffinity=0)
        if bitter_at is not None:                       # a second, dark drop (bitter)
            self.world.mjcf_root.worldbody.add_geom(name="bitter", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.75, 0.6, 0.14],
                                                    pos=[float(bitter_at[0]), float(bitter_at[1]), 0.12],
                                                    rgba=[0.35, 0.2, 0.5, 1.0], contype=0, conaffinity=0)
        if water_at is not None:                        # a clear blue drop (water)
            self.world.mjcf_root.worldbody.add_geom(name="water", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.75, 0.6, 0.14],
                                                    pos=[float(water_at[0]), float(water_at[1]), 0.12],
                                                    rgba=[0.45, 0.75, 1.0, 0.85], contype=0, conaffinity=0)
        self.sim = Simulation(self.world)
        self.sim.set_renderer([cam], camera_res=res, playback_speed=playback_speed, output_fps=fps)
        self.sim.renderer.buffer_frames = buffer          # live mode: do not accumulate frames in memory
        self.cam_name = cam
        self.steps = PreprogrammedSteps()
        self.all_dofs = self.fly.get_actuated_jointdofs_order("position")
        self.leg_dofs = [d for d in self.all_dofs if d not in PROBOSCIS_DOFS]
        self.prob_idx = [self.all_dofs.index(d) for d in PROBOSCIS_DOFS]
        self.leg_idx = [self.all_dofs.index(d) for d in self.leg_dofs]
        self.prob_neutral = np.array([neutral.joint_angles_lookup_rad.get(d, 0.0) for d in PROBOSCIS_DOFS])
        self.ctrl = HybridTurningController(timestep=self.sim.timestep * self.ctrl_every, preprogrammed_steps=self.steps,
                                            output_dof_order=self.leg_dofs)
        # the one controller setting counted in STEPS rather than seconds: keep the same duration in real time
        self.ctrl.retraction_persistence_steps = max(1, int(round(self.ctrl.retraction_persistence_steps / self.ctrl_every)))
        self.thorax = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        # take-off pose: the default stance with femur-tibia joints straightened (tibia flexion -> 0) and coxae pushed back
        base = self.steps.default_pose_by_dof_order(self.leg_dofs); jp = np.array(base, float)
        for k, d in enumerate(self.leg_dofs):
            name = str(d)
            if "tibia" in name and "PITCH" in name: jp[k] = 0.0            # straighten the knee
            if "trochanterfemur" in name and "PITCH" in name: jp[k] = jp[k] * 0.5
        self._jump_pose = jp
        self.reset()

    def reset(self, seed=0):
        self.sim.reset(); self.ctrl.reset(seed=seed)
        q = np.zeros(len(self.all_dofs)); q[self.leg_idx] = self.steps.default_pose_by_dof_order(self.leg_dofs)
        q[self.prob_idx] = self.prob_neutral
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, q)
        self.sim.set_leg_adhesion_states(self.fly.name, np.ones(6, bool))
        self.sim.warmup()
        self.eat_level = 0.0

    def kick(self, strength=1.0):
        """Escape take-off: give the body an upward-backward velocity (mm/s) as if the legs pushed off."""
        v = self.sim.mj_data.qvel
        v[2] += 120.0 * strength; v[0] -= 40.0 * strength * np.cos(self.heading)

    def frame(self):
        """Render the current state from the fly's camera, on demand (live mode)."""
        r = self.sim.renderer
        cam_id = list(r._cameras_names2id.values())[0]
        r.mj_renderer.update_scene(self.sim.mj_data, cam_id, r.scene_option)
        return r.mj_renderer.render().copy()

    def step(self, left=1.0, right=1.0, eat=0.0, render=True, jump=0.0):
        """left/right: descending drive in [0, 1.2] (0 = stand still). eat in [0, 1]: proboscis extension target.
        jump in [0, 1]: escape pose (all legs pushed to extension, the take-off crouch-and-push)."""
        if jump > 0:                                                    # escape take-off: a push-off impulse on the body
            self.kick(jump)
        if self._k % self.ctrl_every == 0 or self._held is None:        # new leg commands; held for ctrl_every steps
            obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            act = self.ctrl.step(np.array([left, right], float), obs)
            alpha = 1 - (1 - 0.02) ** self.ctrl_every                  # same proboscis smoothing per second as before
            self.eat_level += alpha * (eat - self.eat_level)
            q = np.zeros(len(self.all_dofs)); q[self.leg_idx] = act.joint_angles
            q[self.prob_idx] = self.prob_neutral + self.eat_level * PROBOSCIS_EXTENDED
            self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, q)
            if act.adhesion_onoff is not None: self.sim.set_leg_adhesion_states(self.fly.name, act.adhesion_onoff)
            self._held = q
        self._k += 1
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
