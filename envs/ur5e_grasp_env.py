
import os
from pathlib import Path

import gymnasium as gym
import mink
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# importing the robot model from the Mujuco managerie set , along side the gripper . 

_MENAGERIE = os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    os.path.expanduser("~/mujoco_menagerie"),
)
_UR5E_XML = os.path.join(_MENAGERIE, "universal_robots_ur5e", "ur5e.xml")
_GRIPPER_XML = os.path.join(_MENAGERIE, "robotiq_2f85", "2f85.xml")

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# start position for the robot .
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


# mink solver paramaters . 
ACTION_POS_SCALE = 0.02   # max end-effector displacement per step (m)
IK_DT = 0.05               # IK solve / action rate (20 Hz)
GRIPPER_CTRL_MAX = 255.0

# spawn area setting for the random cube . 
GROUND_Z = 0.0

CUBE_HALF_SIZE = 0.02
SPAWN_CENTER_XY = np.array([-0.13, 0.49])
SPAWN_JITTER = 0.08                          # +/- box half-width (m)
CUBE_SPAWN_Z = GROUND_Z + CUBE_HALF_SIZE + 0.001

# workspace bounderies .
WORKSPACE_LOW = np.array([-0.35, 0.30, GROUND_Z + 0.02])
WORKSPACE_HIGH = np.array([0.10, 0.68, 0.55])

LIFT_SUCCESS_HEIGHT = 0.15   # cube must be lifted this far above the ground to "win"
MIN_PAD_FORCE = 0.5          # contact normal-force threshold to count as "grasping"
MAX_EPISODE_SECONDS = 8.0


class UR5eGraspEnv(gym.Env):
    """
    class that inherts from the env gym 
    defines the enviromemnt for training the agent and spawns the robot in MuJuCO with that attached gripper 
    Mink is used to simplify the action for the agent , instead of controlling joint velocity 
    Mink reduces the problem by solving the IK .
    
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.model, self.data = self._build_model()

        self._nsubsteps = max(1, int(round(IK_DT / self.model.opt.timestep)))
        self._arm_ctrl_range = self.model.actuator_ctrlrange[:6].copy()

        self._pinch_site = "gripper-pinch"
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._left_pad_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("gripper-left_pad1", "gripper-left_pad2")
        ]
        self._right_pad_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("gripper-right_pad1", "gripper-right_pad2")
        ]
        self._cube_qpos_adr = self.model.joint("cube_freejoint").qposadr[0]

        # mink setup for the robot end effector .
        self._config = mink.Configuration(self.model)
        self._ee_task = mink.FrameTask(
            frame_name=self._pinch_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(self.model, cost=1e-3)
        self._velocity_limit = mink.VelocityLimit(
            self.model, {n: np.pi for n in ARM_JOINT_NAMES}
        )
        # definition of observation and action space .
        obs_dim = 6 + 6 + 1 + 3 + 3 + 3 + 1  # qpos+qvel+gripper+pinch+cube+rel+grasped, see _get_obs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._ee_target = None
        self._target_rotation = None
        self._episode_time = 0.0
        self._spawn_height = CUBE_SPAWN_Z
        self._viewer = None

    # robot model construction . 
    
    def _build_model(self):
        arm = mujoco.MjSpec.from_file(_UR5E_XML)
        gripper = mujoco.MjSpec.from_file(_GRIPPER_XML)
        site = arm.site("attachment_site")
        site.attach_body(gripper.worldbody.first_body(), "gripper-", "")

        ground = arm.worldbody.add_body(name="ground")
        ground_geom = ground.add_geom(name="ground_geom", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
        ground_geom.rgba = [0.55, 0.55, 0.55, 1.0]

        cube = arm.worldbody.add_body(name="cube", pos=[SPAWN_CENTER_XY[0], SPAWN_CENTER_XY[1], CUBE_SPAWN_Z])
        cube.add_freejoint(name="cube_freejoint")
        cube_geom = cube.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
        )
        cube_geom.rgba = [0.1, 0.8, 0.2, 1.0]
        cube_geom.mass = 0.05
        cube_geom.friction = [1.2, 0.02, 0.0001]  # grippy, so the RL policy isn't fighting slip too

        model = arm.compile()
        data = mujoco.MjData(model)
        return model, data

    # setting the reset , and step for the training loop . 
    
    def reset(self, seed=None, options=None):
        
        # commeneted part of the random cube spawner that was used on a smaller batch , the model generalized well so it was incresed further .
        
        # super().reset(seed=seed)
        # mujoco.mj_resetData(self.model, self.data)

        # self.data.qpos[:6] = HOME_QPOS
        # self.data.ctrl[:6] = HOME_QPOS
        # self.data.ctrl[6] = 0.0  # gripper open

        # cube_xy = SPAWN_CENTER_XY + self.np_random.uniform(-SPAWN_JITTER, SPAWN_JITTER, size=2)
        # cube_qpos = np.array([cube_xy[0], cube_xy[1], CUBE_SPAWN_Z, 1.0, 0.0, 0.0, 0.0])
        # self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_qpos
        
        
        # radnom cube spawner in a larger space .
        
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:6] = HOME_QPOS
        self.data.ctrl[6] = 0.0  # gripper open

        # Independent jitter per axis for rectangular spawn zones:
        jitter_x = self.np_random.uniform(-0.25, 0.25)
        jitter_y = self.np_random.uniform(-0.15, 0.15)
        cube_xy = SPAWN_CENTER_XY + np.array([jitter_x, jitter_y])

        cube_qpos = np.array([cube_xy[0], cube_xy[1], CUBE_SPAWN_Z, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_qpos

        mujoco.mj_forward(self.model, self.data)
        self._config.update(self.data.qpos)
        self._posture_task.set_target(self._config.q)

        home_T = self._config.get_transform_frame_to_world(self._pinch_site, "site")
        self._ee_target = home_T.translation().copy()
        self._target_rotation = home_T.rotation()

        self._episode_time = 0.0
        self._spawn_height = CUBE_SPAWN_Z
        self._prev_dist = self._pinch_to_cube_distance()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta = action[:3] * ACTION_POS_SCALE
        gripper_cmd = (action[3] + 1.0) * 0.5 * GRIPPER_CTRL_MAX

        self._ee_target = np.clip(self._ee_target + delta, WORKSPACE_LOW, WORKSPACE_HIGH)
        target = mink.SE3.from_rotation_and_translation(self._target_rotation, self._ee_target)
        self._ee_task.set_target(target)

        vel = mink.solve_ik(
            self._config,
            [self._ee_task, self._posture_task],
            IK_DT,
            "daqp",
            1e-3,
            limits=[self._velocity_limit],
        )
        self._config.integrate_inplace(vel, IK_DT)

        self.data.ctrl[:6] = np.clip(self._config.q[:6], self._arm_ctrl_range[:, 0], self._arm_ctrl_range[:, 1])
        self.data.ctrl[6] = np.clip(gripper_cmd, 0.0, GRIPPER_CTRL_MAX)

        for _ in range(self._nsubsteps):
            mujoco.mj_step(self.model, self.data)
        self._config.update(self.data.qpos)
        self._episode_time += IK_DT

        pinch_pos = self._config.get_transform_frame_to_world(self._pinch_site, "site").translation()
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        dist = float(np.linalg.norm(pinch_pos - cube_pos))
        grasped = self._check_grasp()
        lift = float(cube_pos[2] - self._spawn_height)

        # reward function .
        reward = -dist                              # reach shaping (dense)
        reward += -0.01 * float(np.square(action).sum())  # small action penalty
        if grasped:
            reward += 1.0                            # bonus for holding contact on both pads
            reward += 5.0 * max(0.0, lift)            # reward lifting only while actually grasped

        success = grasped and lift > LIFT_SUCCESS_HEIGHT
        if success:
            reward += 20.0

        terminated = bool(success)
        truncated = bool(self._episode_time >= MAX_EPISODE_SECONDS or cube_pos[2] < 0.0)

        self._prev_dist = dist
        info = {"distance": dist, "grasped": grasped, "lift": lift, "success": success}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info
    
    
    # several helper functions required to simplify the problem .
    
    def _pinch_to_cube_distance(self):
        pinch_pos = self._config.get_transform_frame_to_world(self._pinch_site, "site").translation()
        cube_pos = self.data.xpos[self._cube_body_id]
        return float(np.linalg.norm(pinch_pos - cube_pos))

    def _check_grasp(self):
        """True if the cube has contact with both pad geoms simultaneously,
        each with at least MIN_PAD_FORCE of normal force."""
        left_force = right_force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self._cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == self._cube_geom_id else g1
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            normal_force = abs(force[0])
            if other in self._left_pad_ids:
                left_force = max(left_force, normal_force)
            elif other in self._right_pad_ids:
                right_force = max(right_force, normal_force)
        return left_force > MIN_PAD_FORCE and right_force > MIN_PAD_FORCE

    def _get_obs(self):
        qpos = self.data.qpos[:6].astype(np.float32)
        qvel = self.data.qvel[:6].astype(np.float32)
        gripper_qpos = np.array([self.data.ctrl[6] / GRIPPER_CTRL_MAX], dtype=np.float32)
        pinch_pos = self._config.get_transform_frame_to_world(self._pinch_site, "site").translation()
        cube_pos = self.data.xpos[self._cube_body_id]
        rel = cube_pos - pinch_pos
        grasped = np.array([1.0 if self._check_grasp() else 0.0], dtype=np.float32)
        return np.concatenate(
            [qpos, qvel, gripper_qpos, pinch_pos.astype(np.float32),
             cube_pos.astype(np.float32), rel.astype(np.float32), grasped]
        )

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None