"""Goal navigation through a frozen 50 Hz gait, with real joint actuation.

The observation/action contract of the lower controller follows DuckKing's
simulation.Driver and Pollen Robotics' infer_policy.py (Apache-2.0). Assets are
external, required, and never replaced by a toy or body-force controller.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import onnxruntime as ort
from gymnasium import spaces

from .constants import HOME, POLICY_JOINTS
from .env import default_robot_xml


@dataclass(frozen=True)
class NavigationConfig:
    robot_xml: str | None = None
    gait_policy: str | None = None
    episode_seconds: float = 12.0
    goal_radius: float = 0.10
    goal_hold_seconds: float = 0.3
    distance_min: float = 0.35
    distance_max: float = 0.65
    control_interval: float = 0.1
    max_speed: float = 0.25
    max_yaw_rate: float = 1.5

    def __post_init__(self):
        if not (0 < self.distance_min <= self.distance_max < 1.5):
            raise ValueError("Require 0 < distance_min <= distance_max < 1.5")
        if self.control_interval < 0.02 or not math.isclose(
            self.control_interval / 0.02, round(self.control_interval / 0.02)
        ):
            raise ValueError("control_interval must be a positive multiple of 0.02")
        if self.episode_seconds <= 0 or self.goal_radius <= 0 or self.goal_hold_seconds <= 0:
            raise ValueError("Episode duration, goal radius and hold time must be positive")


def resolve_assets(config: NavigationConfig) -> tuple[Path, Path]:
    robot = (
        Path(config.robot_xml).expanduser().resolve() if config.robot_xml else default_robot_xml()
    )
    if robot is None or not robot.is_file():
        raise FileNotFoundError("Official MicroDuck XML is required; use --robot-xml")
    explicit = config.gait_policy or os.getenv("MICRODUCK_GAIT_POLICY")
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [
            Path(__file__).resolve().parents[3] / "DuckKing/external/policies/velstand.onnx",
            Path("external/policies/velstand.onnx"),
        ]
    )
    gait = next((p.resolve() for p in candidates if p.is_file()), None)
    if gait is None:
        raise FileNotFoundError("Frozen velstand.onnx is required; use --gait-policy")
    return robot, gait


class NavigationEnv(gym.Env):
    """PPO controls forward velocity and yaw; a frozen ONNX drives 14 joints."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: NavigationConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or NavigationConfig()
        self.render_mode = render_mode
        robot, gait = resolve_assets(self.config)
        spec = mujoco.MjSpec.from_file(str(robot))
        spec.worldbody.add_geom(
            name="navigation_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3, 3, 0.1],
            rgba=[0.72, 0.78, 0.80, 1],
            friction=[1, 0.8, 0.02],
        )
        spec.worldbody.add_site(
            name="navigation_goal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.config.goal_radius, 0.002, 0.002],
            pos=[0.5, 0, 0.004],
            rgba=[0.05, 0.38, 0.88, 0.6],
        )
        # A real zero-height circular carpet is rendered in MuJoCo and moved with
        # the user-selected goal. The site above remains a compact sensor marker.
        spec.worldbody.add_geom(
            name="navigation_goal_carpet",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.config.goal_radius, 0.0015, 0],
            pos=[0.5, 0, 0.0015],
            rgba=[0.05, 0.38, 0.88, 0.72],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        self.model = spec.compile()
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.ids = np.array([self.model.actuator(name).id for name in POLICY_JOINTS])
        joints = self.model.actuator_trnid[self.ids, 0]
        self.q = self.model.jnt_qposadr[joints]
        self.v = self.model.jnt_dofadr[joints]
        free = self.model.joint("trunk_base_freejoint").id
        self.root = int(self.model.jnt_qposadr[free])
        self.root_v = int(self.model.jnt_dofadr[free])
        self.body = self.model.body("trunk_base").id
        self.gyro = int(self.model.sensor_adr[self.model.sensor("imu_ang_vel").id])
        self.goal_site = self.model.site("navigation_goal").id
        self.goal_geom = self.model.geom("navigation_goal_carpet").id
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.gait = ort.InferenceSession(str(gait), options, providers=["CPUExecutionProvider"])
        self.gait_input = self.gait.get_inputs()[0].name
        self.last_joint_action = np.zeros(14, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        self.goal = np.zeros(2)
        self.start = np.zeros(2)
        self.start_yaw = 0.0
        self.steps = 0
        self.hold = 0.0
        self._renderer = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def _rotation(self):
        quat = self.data.qpos[self.root + 3 : self.root + 7]
        yaw = math.atan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        )
        return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

    def _observe(self):
        rotation = self._rotation()
        relative = rotation.T @ (self.goal - self.data.qpos[self.root : self.root + 2])
        velocity = rotation.T @ self.data.qvel[self.root_v : self.root_v + 2]
        gravity = -self.data.xmat[self.body].reshape(3, 3)[2, :]
        return np.concatenate(
            [
                relative,
                [np.linalg.norm(relative)],
                velocity,
                [self.data.sensordata[self.gyro + 2]],
                gravity,
                self.last_action,
            ]
        ).astype(np.float32)

    def _gait_tick(self):
        gravity = -self.data.xmat[self.body].reshape(3, 3)[2, :]
        obs = np.concatenate(
            [
                self.data.sensordata[self.gyro : self.gyro + 3],
                gravity,
                self.data.qpos[self.q] - HOME,
                self.data.qvel[self.v],
                self.last_joint_action,
                self.command,
            ]
        ).astype(np.float32)
        action = self.gait.run(None, {self.gait_input: obs[None]})[0][0]
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError("Frozen gait returned invalid joint targets")
        self.last_joint_action[:] = action
        self.data.ctrl[self.ids] = HOME + action
        for _ in range(4):
            before = self.data.time
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= before or not np.isfinite(self.data.qpos).all():
                raise RuntimeError("MuJoCo numerical instability")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        yaw = float(options.get("yaw", self.np_random.uniform(-math.pi, math.pi)))
        if not np.isfinite(yaw) or abs(yaw) > math.pi:
            raise ValueError("yaw must be a finite angle in [-pi, pi]")
        start = np.asarray(options.get("start", [0.0, 0.0]), dtype=np.float64)
        if start.shape != (2,) or not np.isfinite(start).all() or np.any(np.abs(start) > 1.2):
            raise ValueError("start must contain two finite coordinates in [-1.2, 1.2]")
        self.data.qpos[self.root : self.root + 7] = [
            start[0],
            start[1],
            0.125,
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        ]
        self.data.qpos[self.q] = HOME
        self.data.ctrl[self.ids] = HOME
        self.last_joint_action.fill(0)
        self.last_action.fill(0)
        self.command.fill(0)
        mujoco.mj_forward(self.model, self.data)
        # Identical neutral settling for every strategy, never counted as navigation.
        for _ in range(30):
            self._gait_tick()
        distance = self.np_random.uniform(self.config.distance_min, self.config.distance_max)
        bearing = self.np_random.uniform(-math.pi, math.pi)
        goal = np.asarray(
            options.get("goal", [distance * math.cos(bearing), distance * math.sin(bearing)]),
            dtype=np.float64,
        )
        if goal.shape != (2,) or not np.isfinite(goal).all() or np.any(np.abs(goal) > 1.2):
            raise ValueError("goal must contain two finite coordinates in [-1.2, 1.2]")
        self.start[:] = start
        self.start_yaw = yaw
        self.goal[:] = goal
        self.model.site_pos[self.goal_site, :2] = self.goal
        self.model.geom_pos[self.goal_geom, :2] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold = 0
        return self._observe(), self._metrics()

    def _metrics(self):
        xy = self.data.qpos[self.root : self.root + 2]
        return {
            "distance": float(np.linalg.norm(self.goal - xy)),
            "upright": float(self.data.xmat[self.body].reshape(3, 3)[2, 2]),
            "x": float(xy[0]),
            "y": float(xy[1]),
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "start_x": float(self.start[0]),
            "start_y": float(self.start[1]),
            "start_yaw": float(self.start_yaw),
            "time": self.steps * self.config.control_interval,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Navigation action must contain two finite values")
        action = np.clip(action, -1, 1)
        before = self._metrics()["distance"]
        self.command[:] = 0
        self.command[0] = action[0] * self.config.max_speed
        self.command[2] = action[1] * self.config.max_yaw_rate
        for _ in range(round(self.config.control_interval / 0.02)):
            self._gait_tick()
        self.steps += 1
        info = self._metrics()
        fallen = info["upright"] < 0.45 or self.data.qpos[self.root + 2] < 0.06
        boundary = np.linalg.norm(self.data.qpos[self.root : self.root + 2]) > 1.8
        self.hold = (
            self.hold + self.config.control_interval
            if info["distance"] < self.config.goal_radius and not fallen
            else 0
        )
        success = self.hold + 1e-8 >= self.config.goal_hold_seconds
        timeout = info["time"] + 1e-8 >= self.config.episode_seconds
        reward = (
            30 * (before - info["distance"])
            - 0.015
            - 0.005 * float(np.mean((action - self.last_action) ** 2))
            + (10 if success else 0)
            - (5 if fallen or boundary else 0)
        )
        self.last_action[:] = action
        info.update(
            is_success=bool(success),
            fallen=bool(fallen),
            failure_reason="fall"
            if fallen
            else "boundary"
            if boundary
            else "timeout"
            if timeout and not success
            else "",
            hold_seconds=float(self.hold),
        )
        return (
            self._observe(),
            float(reward),
            bool(success or fallen or boundary),
            bool(timeout and not success and not fallen and not boundary),
            info,
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(640, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(480, self.model.vis.global_.offheight)
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.08]
        camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -60
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
