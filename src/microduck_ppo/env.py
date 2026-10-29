"""Gymnasium environment for balancing MicroDuck on a tilted rotating platform.

The environment reuses DuckKing's policy joint order and HOME pose.  Actions are
small residual target offsets added to a PD position controller, which keeps PPO
focused on recovering from platform motion rather than learning low-level motor
control from scratch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .constants import ACTION_SCALE, CONTROL_INTERVAL, HOME, PHYSICS_TIMESTEP, POLICY_JOINTS


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    half = angle * 0.5
    return np.array([math.cos(half), *(np.asarray(axis) * math.sin(half))], dtype=np.float64)


def default_robot_xml(rollers: bool = False) -> Path | None:
    """Find the official DuckKing/MicroDuck model without copying it into this repo."""

    candidates = []
    configured = os.getenv("MICRODUCK_ROBOT_XML")
    if configured:
        candidates.append(Path(configured))
    here = Path(__file__).resolve()
    filename = "robot_groundcontact_rollers.xml" if rollers else "robot_groundcontact.xml"
    candidates.extend(
        [
            here.parents[3] / "DuckKing" / "external/microduck_rl/src/mjlab_microduck/robot/microduck" / filename,
            here.parents[2] / "DuckKing" / "external/microduck_rl/src/mjlab_microduck/robot/microduck" / filename,
            Path("external/microduck_rl/src/mjlab_microduck/robot/microduck") / filename,
        ]
    )
    return next((path.resolve() for path in candidates if path.exists()), None)


@dataclass(frozen=True)
class BalanceConfig:
    """Physical and training parameters for one environment instance."""

    robot_xml: Path | None = None
    episode_seconds: float = 12.0
    control_interval: float = CONTROL_INTERVAL
    physics_timestep: float = PHYSICS_TIMESTEP
    platform_radius: float = 0.30
    platform_top: float = 0.0
    platform_tilt_x: float = math.radians(8.0)
    platform_tilt_y: float = math.radians(8.0)
    terrain: str = "turntable"
    terrain_friction: float = 1.0
    roughness: float = 0.0
    slope: float = 0.0
    omega_center: float = 0.75
    omega_amplitude: float = 0.55
    omega_frequency: float = 0.65
    max_initial_tilt: float = math.radians(5.0)
    max_initial_spin: float = 0.12
    curriculum: float = 1.0
    action_scale: float = ACTION_SCALE
    disturbance_enabled: bool = False
    disturbance_count: int = 6
    disturbance_start: float = 1.0
    disturbance_interval: float = 1.5
    disturbance_force: float = 0.35
    disturbance_torque: float = 0.06
    disturbance_duration: float = 0.05
    disturbance_seed: int | None = None
    render_width: int = 720
    render_height: int = 540

    def with_curriculum(self, level: float) -> BalanceConfig:
        return replace(self, curriculum=float(np.clip(level, 0.0, 1.0)))


class MicroDuckBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """PPO environment with real MuJoCo contacts and a non-uniform turntable.

    Observation layout:

    ``gravity(3), angular_velocity(3), joint_position_error(14), joint_velocity(14),
    foot_contact(2), foot_slip(2), platform_omega(3), platform_tilt(2),
    phase(sin/cos)(2), previous_action(14)``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: BalanceConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or BalanceConfig()
        self.render_mode = render_mode
        self._rng = np.random.default_rng()
        xml_candidate = self.config.robot_xml or default_robot_xml(self.config.terrain == "roller")
        self._toy = xml_candidate is None or not xml_candidate.exists()
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._mocap_id = -1
        self._platform_geom = -1
        self._trunk = -1
        self._gyro_adr = -1
        self._q = np.zeros(14, dtype=np.int32)
        self._v = np.zeros(14, dtype=np.int32)
        self._actuators = np.zeros(14, dtype=np.int32)
        self._feet = np.zeros(2, dtype=np.int32)
        self._foot_sites = np.zeros(2, dtype=np.int32)
        self._root = -1
        self._time = 0.0
        self._phase = 0.0
        self._omega_sign = 1.0
        self._last_action = np.zeros(14, dtype=np.float32)
        self._last_feet = np.zeros((2, 3), dtype=np.float64)
        self._last_info: dict[str, Any] = {}
        self._disturbances: list[dict[str, Any]] = []
        self._active_disturbance: dict[str, Any] | None = None
        self._build_model()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        # 3+3+14+14+2+2+3+2+2+14 = 59
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(59,), dtype=np.float32)

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("MuJoCo model is not initialized")
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        if self._data is None:
            raise RuntimeError("MuJoCo data is not initialized")
        return self._data

    def _build_model(self) -> None:
        xml = self.config.robot_xml or default_robot_xml(self.config.terrain == "roller")
        if self._toy:
            self._build_toy_model()
            return
        if xml is None:
            raise FileNotFoundError(
                "找不到 MicroDuck MJCF。请设置 MICRODUCK_ROBOT_XML，或运行 DuckKing 的 "
                "scripts/setup_simulator.sh 获取 external/microduck_rl。"
            )
        spec = mujoco.MjSpec.from_file(str(xml))
        # A mocap body gives the turntable a prescribed, time-varying pose while
        # preserving physical contact and friction for the robot feet.
        platform = spec.worldbody.add_body(name="turntable", mocap=True, pos=[0, 0, -0.035])
        platform.add_geom(
            name="turntable_surface",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.config.platform_radius, 0.035, 0],
            rgba=[0.08, 0.33, 0.58, 1],
            friction=[self.config.terrain_friction, self.config.terrain_friction * 0.8, 0.02],
            condim=6,
        )
        if self.config.terrain == "rough" and self.config.roughness > 0:
            for index, (x, y) in enumerate(
                ((-0.12, -0.12), (0.0, -0.12), (0.12, -0.12),
                 (-0.12, 0.0), (0.0, 0.0), (0.12, 0.0),
                 (-0.12, 0.12), (0.0, 0.12), (0.12, 0.12))
            ):
                height = self.config.roughness * (0.5 + 0.25 * (index % 3))
                platform.add_geom(
                    name=f"rough_patch_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, y, 0.035 + height / 2],
                    size=[0.045, 0.045, height / 2],
                    rgba=[0.12, 0.40, 0.62, 1],
                    friction=[self.config.terrain_friction, self.config.terrain_friction * 0.8, 0.02],
                    condim=6,
                )
        self._model = spec.compile()
        self._data = mujoco.MjData(self.model)
        self._configure_ids()
        self.model.opt.timestep = self.config.physics_timestep
        self._renderer = None

    def _build_toy_model(self) -> None:
        """Small fallback used for CI and demos when the licensed official asset is absent."""

        xml = """
        <mujoco model="microduck_toy"><option timestep="0.005" gravity="0 0 -9.81"/>
          <worldbody>
            <body name="turntable" mocap="true" pos="0 0 -0.03">
              <geom name="turntable_surface" type="cylinder" size="0.30 0.035" rgba="0.08 0.33 0.58 1" friction="1 .8 .02" condim="6"/>
            </body>
            <body name="trunk_base" pos="0 0 .12"><freejoint name="trunk_base_freejoint"/>
              <inertial pos="0 0 0" mass="0.20" diaginertia=".0002 .0002 .0002"/>
              <geom name="trunk" type="box" size=".045 .035 .045" rgba=".95 .75 .15 1"/>
              <site name="imu" pos="0 0 0"/>
              <body name="left_leg" pos="0 .025 -.05"><joint name="left_hip_roll" type="hinge" axis="1 0 0"/><geom type="capsule" fromto="0 0 0 0 0 -.07" size=".012"/><geom name="left_foot_collision" type="box" pos="0 0 -.075" size=".025 .025 .008"/></body>
              <body name="right_leg" pos="0 -.025 -.05"><joint name="right_hip_roll" type="hinge" axis="1 0 0"/><geom type="capsule" fromto="0 0 0 0 0 -.07" size=".012"/><geom name="right_foot_collision" type="box" pos="0 0 -.075" size=".025 .025 .008"/></body>
            </body>
          </worldbody>
          <actuator><position name="left_hip_roll" joint="left_hip_roll" kp="4"/><position name="right_hip_roll" joint="right_hip_roll" kp="4"/></actuator>
          <sensor><gyro name="imu_ang_vel" site="imu"/></sensor>
        </mujoco>"""
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self.model)
        # Map two toy joints to the first two policy channels; others remain zero.
        self._actuators[:2] = [0, 1]
        self._q[:2] = self.model.jnt_qposadr[self.model.actuator_trnid[:2, 0]]
        self._v[:2] = self.model.jnt_dofadr[self.model.actuator_trnid[:2, 0]]
        self._trunk = self.model.body("trunk_base").id
        self._root = self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id]
        self._gyro_adr = self.model.sensor_adr[self.model.sensor("imu_ang_vel").id]
        self._platform_geom = self.model.geom("turntable_surface").id
        self._mocap_id = self.model.body("turntable").mocapid[0]
        self._feet[:] = [self.model.geom("left_foot_collision").id, self.model.geom("right_foot_collision").id]
        self._foot_sites[:] = [self.model.site("imu").id, self.model.site("imu").id]

    def _configure_ids(self) -> None:
        self._actuators = np.array([self.model.actuator(name).id for name in POLICY_JOINTS], dtype=np.int32)
        joint_ids = self.model.actuator_trnid[self._actuators, 0]
        self._q = self.model.jnt_qposadr[joint_ids]
        self._v = self.model.jnt_dofadr[joint_ids]
        self._root = self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id]
        self._trunk = self.model.body("trunk_base").id
        self._gyro_adr = self.model.sensor_adr[self.model.sensor("imu_ang_vel").id]
        self._platform_geom = self.model.geom("turntable_surface").id
        self._mocap_id = int(self.model.body("turntable").mocapid[0])
        self._feet = np.array(self._find_foot_geoms(), dtype=np.int32)
        self._foot_sites = np.array(
            [self.model.site("left_foot").id, self.model.site("right_foot").id], dtype=np.int32
        )

    def _find_foot_geoms(self) -> list[int]:
        names = ("left_foot_collision", "right_foot_collision")
        try:
            return [self.model.geom(name).id for name in names]
        except KeyError:
            ankle_names = ("ankle_l_v1", "ankle_r_v1")
            result = []
            for ankle in ankle_names:
                body = self.model.body(ankle).id
                bodies = {body}
                for candidate in range(self.model.nbody):
                    parent = candidate
                    while parent > 0:
                        parent = int(self.model.body_parentid[parent])
                        if parent == body:
                            bodies.add(candidate)
                            break
                candidates = [
                    i for i in range(self.model.ngeom)
                    if self.model.geom_bodyid[i] in bodies and self.model.geom_contype[i] != 0
                ]
                if not candidates:
                    raise ValueError(f"找不到 {ankle} 的接触几何体") from None
                result.append(candidates[0])
            return result

    def set_curriculum(self, level: float) -> None:
        """Set difficulty in [0, 1]; useful from a Stable-Baselines3 callback."""

        self.config = self.config.with_curriculum(level)

    def _difficulty(self) -> float:
        return float(np.clip(self.config.curriculum, 0.0, 1.0))

    def _omega(self, t: float) -> float:
        level = self._difficulty()
        return self._omega_sign * level * (
            self.config.omega_center + self.config.omega_amplitude * math.sin(self.config.omega_frequency * t + self._phase)
        )

    def _angle(self, t: float) -> float:
        level = self._difficulty()
        frequency = self.config.omega_frequency
        integral = self.config.omega_center * t
        if abs(frequency) > 1e-9:
            integral += self.config.omega_amplitude / frequency * (
                math.cos(self._phase) - math.cos(frequency * t + self._phase)
            )
        return self._omega_sign * level * integral

    def _platform_tilt(self) -> tuple[float, float]:
        if self.config.terrain == "flat":
            return 0.0, 0.0
        level = self._difficulty()
        return (
            level * self.config.platform_tilt_x + self.config.slope,
            level * self.config.platform_tilt_y,
        )

    def _set_platform(self, t: float) -> None:
        tilt_x, tilt_y = self._platform_tilt()
        yaw = self._angle(t)
        q = _quat_mul(_axis_quat(np.array([0, 0, 1]), yaw), _quat_mul(
            _axis_quat(np.array([1, 0, 0]), tilt_x), _axis_quat(np.array([0, 1, 0]), tilt_y)
        ))
        if self._mocap_id >= 0:
            self.data.mocap_pos[self._mocap_id] = [0, 0, self.config.platform_top - 0.035]
            self.data.mocap_quat[self._mocap_id] = q

    def _reset_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._time = 0.0
        self._phase = float(self._rng.uniform(0, math.tau))
        self._omega_sign = float(self._rng.choice([-1.0, 1.0]))
        level = self._difficulty()
        root_quat = _quat_mul(
            _axis_quat(np.array([1, 0, 0]), self._rng.uniform(-self.config.max_initial_tilt, self.config.max_initial_tilt) * level),
            _axis_quat(np.array([0, 1, 0]), self._rng.uniform(-self.config.max_initial_tilt, self.config.max_initial_tilt) * level),
        )
        self.data.qpos[self._root : self._root + 7] = [0, 0, 0.12, *root_quat]
        if self._toy:
            self.data.qpos[self._q] = 0
            self.data.ctrl[:] = 0
        else:
            self.data.qpos[self._q] = HOME
            self.data.ctrl[self._actuators] = HOME
        self.data.qvel[:] = 0
        self._last_action.fill(0)
        self._prepare_disturbances()
        self._set_platform(0)
        mujoco.mj_forward(self.model, self.data)
        self._last_feet[:] = self.data.site_xpos[self._foot_sites]

    def _prepare_disturbances(self) -> None:
        self._disturbances = []
        self._active_disturbance = None
        if not self.config.disturbance_enabled:
            return
        seed = self.config.disturbance_seed
        rng = np.random.default_rng(
            seed if seed is not None else int(self._rng.integers(2**31 - 1))
        )
        for index in range(self.config.disturbance_count):
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-8)
            self._disturbances.append(
                {
                    "index": index,
                    "time": self.config.disturbance_start + index * self.config.disturbance_interval,
                    "duration": self.config.disturbance_duration,
                    "direction": direction.tolist(),
                    "force": self.config.disturbance_force,
                    "torque": float(
                        rng.uniform(-self.config.disturbance_torque, self.config.disturbance_torque)
                    ),
                }
            )

    def _apply_disturbance(self) -> dict[str, Any] | None:
        self.data.xfrc_applied[self._trunk] = 0
        self._active_disturbance = None
        for event in self._disturbances:
            if event["time"] <= self._time < event["time"] + event["duration"]:
                direction = np.asarray(event["direction"], dtype=np.float64)
                self.data.xfrc_applied[self._trunk, :3] = [
                    event["force"] * direction[0], event["force"] * direction[1], 0
                ]
                self.data.xfrc_applied[self._trunk, 5] = event["torque"]
                self._active_disturbance = event
                break
        return self._active_disturbance

    def _contact(self, geom_id: int) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == geom_id and int(c.geom2) == self._platform_geom:
                return True
            if int(c.geom2) == geom_id and int(c.geom1) == self._platform_geom:
                return True
        return False

    def _observation(self) -> np.ndarray:
        rotation = self.data.xmat[self._trunk].reshape(3, 3)
        gravity = -rotation[2, :]
        gyro = self.data.sensordata[self._gyro_adr : self._gyro_adr + 3]
        joint_pos = self.data.qpos[self._q] - (np.zeros(14) if self._toy else HOME)
        joint_vel = self.data.qvel[self._v]
        contacts = np.array([float(self._contact(int(g))) for g in self._feet])
        feet = self.data.site_xpos[self._foot_sites]
        slip = np.zeros(2)
        if self._time > 0:
            velocity = (feet - self._last_feet) / self.config.control_interval
            omega_vec = np.array([0, 0, self._omega(self._time)])
            platform_velocity = np.cross(omega_vec, feet)
            slip = np.linalg.norm(velocity - platform_velocity, axis=1)
        tilt = np.asarray(self._platform_tilt())
        platform_omega = np.array([0, 0, self._omega(self._time)])
        obs = np.concatenate(
            [gravity, gyro, joint_pos, joint_vel, contacts, slip, platform_omega, tilt,
             [math.sin(self._phase), math.cos(self._phase)], self._last_action]
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _reward(self, action: np.ndarray, previous_action: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
        obs = self._observation()
        gravity = obs[:3]
        gyro = obs[3:6]
        contacts = obs[34:36]
        slip = obs[36:38]
        upright = float(np.clip(-gravity[2], -1.0, 1.0))
        contact_reward = float(np.mean(contacts))
        slip_penalty = float(np.mean(np.square(slip)))
        angular_penalty = float(np.mean(np.square(gyro)))
        previous_action = self._last_action if previous_action is None else previous_action
        action_penalty = float(np.mean(np.square(action - previous_action)))
        energy_penalty = float(np.mean(np.square(action)))
        failed = self._has_fallen()
        reward = (
            1.8 * max(0.0, upright)
            + 0.45 * contact_reward
            - 0.18 * slip_penalty
            - 0.12 * angular_penalty
            - 0.015 * action_penalty
            - 0.004 * energy_penalty
            - (8.0 if failed else 0.0)
        )
        return float(reward), {
            "upright": upright,
            "contact": contact_reward,
            "slip": slip_penalty,
            "angular_velocity": angular_penalty,
            "failed": float(failed),
        }

    def _has_fallen(self) -> bool:
        rotation = self.data.xmat[self._trunk].reshape(3, 3)
        upright = float(rotation[2, 2])
        z = float(self.data.xpos[self._trunk, 2])
        xy = float(np.linalg.norm(self.data.xpos[self._trunk, :2]))
        return not np.isfinite(self.data.qpos).all() or z < 0.055 or upright < 0.35 or xy > self.config.platform_radius * 1.15

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        obs = self._observation()
        return obs, {
            "difficulty": self._difficulty(),
            "backend": "mujoco" if not self._toy else "mujoco_toy",
            "terrain": self.config.terrain,
            "disturbances": list(self._disturbances),
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(14)
        action = np.clip(action, -1.0, 1.0)
        previous = self._last_action.copy()
        n_substeps = max(1, round(self.config.control_interval / self.config.physics_timestep))
        for _ in range(n_substeps):
            self._set_platform(self._time)
            self._apply_disturbance()
            if self._toy:
                self.data.ctrl[:] = action[:2] * self.config.action_scale
            else:
                self.data.ctrl[self._actuators] = HOME + action * self.config.action_scale
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self._trunk] = 0
            self._time += self.config.physics_timestep
        reward, metrics = self._reward(action, previous)
        self._last_action = action
        obs = self._observation()
        self._last_feet[:] = self.data.site_xpos[self._foot_sites]
        terminated = self._has_fallen()
        truncated = self._time >= self.config.episode_seconds
        metrics.update(
            {
                "time": self._time,
                "omega": self._omega(self._time),
                "difficulty": self._difficulty(),
                "terrain": self.config.terrain,
                "disturbance_active": float(self._active_disturbance is not None),
                "disturbance_index": (
                    -1 if self._active_disturbance is None else self._active_disturbance["index"]
                ),
                "disturbance_count": len(self._disturbances),
                "failure_reason": "fallen" if terminated else "",
            }
        )
        self._last_info = metrics
        return obs, reward, terminated, truncated, metrics

    def render(self) -> np.ndarray | None:
        if self._toy and self.render_mode not in {"rgb_array", "human"}:
            return None
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), self.config.render_width
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), self.config.render_height
            )
            self._renderer = mujoco.Renderer(
                self.model, height=self.config.render_height, width=self.config.render_width
            )
        camera = mujoco.MjvCamera()
        camera.distance = 0.55
        camera.azimuth = math.degrees(self._angle(self._time)) + 135
        camera.elevation = -20
        camera.lookat[:] = [0, 0, 0.08]
        self._renderer.update_scene(self.data, camera)
        frame = self._renderer.render()
        if self.render_mode == "human":
            return frame
        return frame.copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = None
        self._data = None
