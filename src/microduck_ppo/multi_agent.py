"""Centralized-critic-friendly multi-MicroDuck coordination environment.

This task uses a compact MuJoCo model with the same physical free-body/contact
semantics as the official scene.  One PPO policy controls all ducks, which is a
centralized-critic baseline for formation, obstacle avoidance, and cooperative
goal reaching.  The API is intentionally vectorized: replace this toy body
with attached official MicroDuck XML when the multi-agent policy is ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class CoordinationConfig:
    ducks: int = 2
    episode_seconds: float = 10.0
    control_interval: float = 0.05
    physics_timestep: float = 0.01
    arena_size: float = 1.1
    goal_radius: float = 0.08
    formation_distance: float = 0.28
    collision_radius: float = 0.09
    force_scale: float = 0.9
    torque_scale: float = 0.08
    obstacle: bool = True
    robot_xml: Path | None = None


class MultiDuckCoordinationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Centralized PPO task for formation, collision avoidance and goal reach."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: CoordinationConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or CoordinationConfig()
        if not 2 <= self.config.ducks <= 4:
            raise ValueError("ducks must be between 2 and 4")
        self.render_mode = render_mode
        self._rng = np.random.default_rng()
        self._official = self._find_robot_xml() is not None
        self._model = self._build_model()
        self._data = mujoco.MjData(self._model)
        self._roots = np.array(
            [self._model.jnt_qposadr[self._model.joint(self._root_name(i)).id]
             for i in range(self.config.ducks)], dtype=np.int32
        )
        self._bodies = np.array([self._model.body(self._body_name(i)).id for i in range(self.config.ducks)])
        self._goals = np.zeros((self.config.ducks, 2), dtype=np.float64)
        self._time = 0.0
        self._last_action = np.zeros(self.config.ducks * 3, dtype=np.float32)
        self._renderer: mujoco.Renderer | None = None
        # Per duck: x/y, vx/vy, sin/cos yaw, goal dx/dy, goal distance = 9.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.config.ducks * 9 + 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.config.ducks * 3,), dtype=np.float32)

    def _find_robot_xml(self) -> Path | None:
        candidates = []
        if self.config.robot_xml:
            candidates.append(self.config.robot_xml)
        here = Path(__file__).resolve()
        candidates.extend([
            here.parents[3] / "DuckKing" / "external/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml",
            here.parents[2] / "DuckKing" / "external/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml",
            Path("external/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"),
        ])
        return next((p.resolve() for p in candidates if p.exists()), None)

    def _body_name(self, i: int) -> str:
        return f"duck_{i}_trunk_base" if self._official else f"duck_{i}"

    def _root_name(self, i: int) -> str:
        return f"duck_{i}_trunk_base_freejoint" if self._official else f"duck_{i}_free"

    def _build_model(self) -> mujoco.MjModel:
        obstacle = (
            '<geom name="obstacle" type="box" pos="0 0.12 0.06" size=".08 .18 .06" '
            'rgba=".5 .2 .15 1"/>' if self.config.obstacle else ""
        )
        robot_xml = self._find_robot_xml()
        if robot_xml is not None:
            spec = mujoco.MjSpec.from_string(
                f'<mujoco model="multi_microduck"><option timestep="{self.config.physics_timestep:.4f}" '
                f'gravity="0 0 -9.81"/><worldbody><geom name="floor" type="plane" '
                f'size="2 2 .05" friction="1 .8 .02"/>{obstacle}</worldbody></mujoco>'
            )
            for i in range(self.config.ducks):
                robot = mujoco.MjSpec.from_file(str(robot_xml))
                spec.attach(robot, prefix=f"duck_{i}_", frame=spec.worldbody.add_frame())
            return spec.compile()
        bodies = []
        for i in range(self.config.ducks):
            bodies.append(
                f'<body name="duck_{i}" pos="{-.22 + i * .44} -.35 .06">'
                f'<freejoint name="duck_{i}_free"/><inertial pos="0 0 0" mass=".20" '
                f'diaginertia=".0002 .0002 .0002"/><geom name="duck_geom_{i}" type="sphere" '
                'size=".055" rgba=".95 .75 .15 1"/>'
                + '</body>'
            )
        xml = (
            f'<mujoco model="multi_duck_coordination"><option timestep="{self.config.physics_timestep:.4f}" '
            f'gravity="0 0 -9.81"/><worldbody><geom name="floor" type="plane" size="2 2 .05" '
            f'friction="1 .8 .02"/>{obstacle}{"".join(bodies)}</worldbody></mujoco>'
        )
        return mujoco.MjModel.from_xml_string(xml)

    def _positions(self) -> np.ndarray:
        return self._data.xpos[self._bodies, :2].copy()

    def _yaw(self, i: int) -> float:
        q = self._data.qpos[self._roots[i] + 3 : self._roots[i] + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _observation(self) -> np.ndarray:
        rows = []
        for i in range(self.config.ducks):
            root = self._roots[i]
            pos = self._data.qpos[root : root + 2]
            vel = self._data.qvel[root : root + 2]
            dxdy = self._goals[i] - pos
            rows.extend([pos[0], pos[1], vel[0], vel[1], math.sin(self._yaw(i)), math.cos(self._yaw(i)),
                         dxdy[0], dxdy[1], float(np.linalg.norm(dxdy))])
        center = np.mean(self._positions(), axis=0)
        rows.extend([center[0], center[1]])
        return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)
        self._time = 0.0
        self._last_action.fill(0)
        starts = np.array([[-.22 + i * .44, -.35] for i in range(self.config.ducks)])
        starts += self._rng.normal(0, 0.015, starts.shape)
        goals_center = self._rng.uniform([-0.25, 0.15], [0.25, 0.5])
        offsets = np.array([[-.14, 0], [.14, 0], [-.14, .14], [.14, .14]])[: self.config.ducks]
        self._goals = goals_center + offsets
        for i, root in enumerate(self._roots):
            self._data.qpos[root : root + 7] = [*starts[i], .06, 1, 0, 0, 0]
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        return self._observation(), {"backend": "mujoco", "ducks": self.config.ducks, "goals": self._goals.tolist()}

    def _collision_count(self) -> int:
        positions = self._positions()
        return sum(
            float(np.linalg.norm(positions[i] - positions[j]) < self.config.collision_radius)
            for i in range(self.config.ducks) for j in range(i + 1, self.config.ducks)
        )

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.config.ducks, 3), -1, 1)
        previous = self._last_action.copy()
        substeps = max(1, round(self.config.control_interval / self.config.physics_timestep))
        for _ in range(substeps):
            for i, body in enumerate(self._bodies):
                self._data.xfrc_applied[body, :3] = [
                    action[i, 0] * self.config.force_scale,
                    action[i, 1] * self.config.force_scale,
                    0,
                ]
                self._data.xfrc_applied[body, 5] = action[i, 2] * self.config.torque_scale
            mujoco.mj_step(self._model, self._data)
            self._data.xfrc_applied[self._bodies] = 0
            self._time += self.config.physics_timestep
        positions = self._positions()
        distances = np.linalg.norm(self._goals - positions, axis=1)
        formation_error = abs(np.linalg.norm(positions[0] - positions[1]) - self.config.formation_distance)
        collisions = self._collision_count()
        boundary = bool(np.any(np.abs(positions) > self.config.arena_size))
        success = bool(np.all(distances < self.config.goal_radius) and formation_error < 0.10)
        action_rate = float(np.mean(np.square(action.reshape(-1) - previous)))
        reward = float(-np.mean(distances) - 0.8 * formation_error - 1.5 * collisions - 0.04 * action_rate)
        if success:
            reward += 8.0
        terminated = boundary or not np.isfinite(self._data.qpos).all()
        truncated = self._time >= self.config.episode_seconds
        self._last_action = action.reshape(-1).copy()
        info = {
            "time": self._time, "goal_error": float(np.mean(distances)),
            "formation_error": float(formation_error), "collisions": collisions,
            "success": float(success), "failure_reason": "boundary" if boundary else "",
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._model.vis.global_.offwidth = max(int(self._model.vis.global_.offwidth), 640)
            self._model.vis.global_.offheight = max(int(self._model.vis.global_.offheight), 480)
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 1.1, 90, -45
        camera.lookat[:] = [0, 0.1, 0]
        self._renderer.update_scene(self._data, camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
