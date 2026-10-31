"""In-process realtime MuJoCo sessions for the local audience playground."""

from __future__ import annotations

import io
import gc
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from .navigation import NavigationConfig, NavigationEnv
from .navigation_artifacts import read_json

PLAYGROUND_WORLD_LIMIT = 3.0
PLAYGROUND_EPISODE_SECONDS = 20.0
PLAYGROUND_FRAME_EVERY = 5
PLAYGROUND_STATE_EVERY = 2


class PlaygroundSession:
    def __init__(self, session_id, run, agents, goal):
        self.id = session_id
        self.run = Path(run)
        self.agents = [dict(agent) for agent in agents]
        self.stages = [agent["stage"] for agent in self.agents]
        self.stage = self.stages[0]
        self.start = list(self.agents[0]["start"])
        self.goal = list(goal)
        self.yaw = float(self.agents[0].get("yaw", 0.0))
        self.status = "starting"
        self.error = ""
        self.frame = None
        self.frame_seq = 0
        self.info = {
            "time": 0.0,
            "distance": float(np.linalg.norm(np.asarray(goal) - self.start)),
            "agent_count": len(self.agents),
            "success_count": 0,
        }
        self.visual_model = None
        self.visual_state = None
        self.started = time.perf_counter()
        self.finished = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        # Camera is viewer state only. It never changes the MuJoCo world or
        # the policy observation, so moving the view cannot move the target.
        self.camera = {"azimuth": 135.0, "elevation": -45.0, "distance": 1.6}
        self.camera_home = dict(self.camera)
        self.thread = threading.Thread(
            target=self._run, name=f"microduck-play-{session_id}", daemon=True
        )

    def start_session(self):
        self.thread.start()
        return self

    def _publish(self, env):
        with self.lock:
            camera = dict(self.camera)
        env.set_camera(**camera)
        frame = env.render()
        if frame is None:
            return
        stream = io.BytesIO()
        imageio.imwrite(stream, frame, format="jpeg", quality=82)
        with self.lock:
            self.frame = stream.getvalue()
            self.frame_seq += 1

    def _run(self):
        env = None
        try:
            manifest = read_json(self.run / "run.json")
            config = NavigationConfig(**manifest["config"])
            # The policy was trained on 0.35–0.65 m routes.  The audience
            # playground deliberately runs a larger, 20 s world so a hand
            # placed scene is a visible generalization test.  This does not
            # alter the recorded training configuration or its evaluation.
            config = replace(
                config,
                episode_seconds=max(PLAYGROUND_EPISODE_SECONDS, config.episode_seconds),
                world_limit=max(PLAYGROUND_WORLD_LIMIT, config.world_limit),
                boundary_radius=max(3.5, config.boundary_radius),
                floor_half_extent=max(4.0, config.floor_half_extent),
            )
            torch.set_num_threads(1)
            models = []
            envs = []
            observations = []
            agent_infos = []
            for agent in self.agents:
                if self.stop_event.is_set():
                    with self.lock:
                        self.status = "stopped"
                        self.finished = time.perf_counter()
                    return
                stage = agent["stage"]
                policy = (self.run / stage / "policy.zip").resolve()
                if not policy.is_file() or not policy.is_relative_to(self.run.resolve()):
                    raise FileNotFoundError(f"Selected checkpoint policy is not ready: {stage}")
                model = PPO.load(policy, device="cpu")
                agent_env = NavigationEnv(config, render_mode="rgb_array")
                obs, agent_info = agent_env.reset(
                    seed=42,
                    options={
                        "start": agent["start"],
                        "goal": self.goal,
                        "yaw": float(agent.get("yaw", 0.0)),
                    },
                )
                models.append(model)
                envs.append(agent_env)
                observations.append(obs)
                agent_infos.append(dict(agent_info))
            env = envs[0]
            with self.lock:
                self.visual_model = env.visual_model_payload()
                self.visual_state = {
                    "agents": [
                        {
                            **agent_env.visual_state(),
                            "id": self.agents[index]["id"],
                            "stage": self.agents[index]["stage"],
                        }
                        for index, agent_env in enumerate(envs)
                    ],
                    "goal": self.goal,
                    "start": self.start,
                    "yaw": self.yaw,
                }
                self.camera_home = {
                    "azimuth": env.camera_azimuth,
                    "elevation": env.camera_elevation,
                    "distance": env.camera_distance,
                }
                self.camera = dict(self.camera_home)
                self.status = "running"
                self.info = {
                    "time": 0.0,
                    "distance": float(np.linalg.norm(np.asarray(self.goal) - self.start)),
                    "agents": [
                        {
                            **agent_info,
                            "id": agent["id"],
                            "stage": agent["stage"],
                            "is_success": False,
                        }
                        for agent, agent_info in zip(self.agents, agent_infos, strict=True)
                    ],
                    "agent_count": len(self.stages),
                    "success_count": 0,
                }
            self._publish(env)
            reached = [False] * len(envs)
            finished = [False] * len(envs)
            for frame_index in range(round(config.episode_seconds / config.control_interval)):
                if self.stop_event.is_set():
                    break
                loop_started = time.perf_counter()
                for index, (model, agent_env) in enumerate(zip(models, envs, strict=True)):
                    if finished[index]:
                        continue
                    if reached[index]:
                        agent_infos[index] = {
                            **agent_infos[index],
                            "time": (frame_index + 1) * config.control_interval,
                            "is_success": True,
                            "failure_reason": "",
                            "reward": 0.0,
                        }
                        continue
                    action, _ = model.predict(observations[index], deterministic=True)
                    observations[index], reward, terminated, truncated, agent_infos[index] = agent_env.step(action)
                    agent_infos[index] = {
                        **agent_infos[index],
                        "id": self.agents[index]["id"],
                        "stage": self.agents[index]["stage"],
                        "reward": float(reward),
                    }
                    if agent_infos[index].get("is_success"):
                        reached[index] = True
                    elif terminated or truncated:
                        finished[index] = True
                # The browser's interactive mesh is driven by visual-state.
                # JPEG evidence is only a fallback, so encode it at 2 Hz
                # instead of making MuJoCo/ImageIO work on every control step.
                if frame_index % PLAYGROUND_FRAME_EVERY == 0:
                    self._publish(env)
                with self.lock:
                    self.info = {
                        "time": (frame_index + 1) * config.control_interval,
                        "distance": min(float(item.get("distance", 0.0)) for item in agent_infos),
                        "agents": [dict(item) for item in agent_infos],
                        "agent_count": len(envs),
                        "success_count": sum(reached),
                        "is_success": all(reached),
                    }
                    if frame_index % PLAYGROUND_STATE_EVERY == 0:
                        self.visual_state = {
                            "agents": [
                                {
                                    **agent_env.visual_state(),
                                    "id": self.agents[index]["id"],
                                    "stage": self.agents[index]["stage"],
                                }
                                for index, agent_env in enumerate(envs)
                            ],
                            "goal": self.goal,
                            "start": self.start,
                            "yaw": self.yaw,
                        }
                if all(reached[index] or finished[index] for index in range(len(envs))):
                    break
                time.sleep(max(0, config.control_interval - (time.perf_counter() - loop_started)))
            with self.lock:
                self.status = "stopped" if self.stop_event.is_set() else "ready"
                self.finished = time.perf_counter()
        except Exception as error:
            with self.lock:
                self.status = "failed"
                self.error = f"{type(error).__name__}: {error}"
                self.finished = time.perf_counter()
        finally:
            for agent_env in locals().get("envs", []):
                agent_env.close()
            # Release policy modules and MuJoCo buffers as soon as a session
            # ends. The viewer may keep a short-lived result record, but it
            # must not keep the heavyweight runtime objects alive.
            for name in ("models", "envs", "observations", "agent_infos"):
                value = locals().get(name)
                if isinstance(value, list):
                    value.clear()
            gc.collect()

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "stages": list(self.stages),
                "agents": [dict(agent) for agent in self.agents],
                "start": self.start,
                "goal": self.goal,
                "yaw": self.yaw,
                "frame_seq": self.frame_seq,
                "info": dict(self.info),
                "camera": dict(self.camera),
                "visual_ready": self.visual_model is not None,
                "error": self.error,
                "elapsed_seconds": (self.finished or time.perf_counter()) - self.started,
            }

    def frame_bytes(self):
        with self.lock:
            return self.frame

    def visual_model_payload(self):
        with self.lock:
            return self.visual_model

    def visual_state_payload(self):
        with self.lock:
            return self.visual_state

    def update_camera(self, *, azimuth=None, elevation=None, distance=None, reset=False):
        """Update viewer camera state; values are clamped by NavigationEnv."""
        with self.lock:
            if reset:
                self.camera = dict(self.camera_home)
            for key, value in (
                ("azimuth", azimuth),
                ("elevation", elevation),
                ("distance", distance),
            ):
                if value is None:
                    continue
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"camera {key} must be finite")
                self.camera[key] = value
            return dict(self.camera)

    def request_stop(self):
        self.stop_event.set()


def create_session(run, agents, goal):
    session_id = time.strftime("play-%H%M%S-") + uuid.uuid4().hex[:5]
    return PlaygroundSession(session_id, run, agents, goal).start_session()
