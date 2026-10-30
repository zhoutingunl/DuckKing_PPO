"""In-process realtime MuJoCo sessions for the local audience playground."""

from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from .navigation import NavigationConfig, NavigationEnv
from .navigation_artifacts import read_json


class PlaygroundSession:
    def __init__(self, session_id, run, stage, start, goal, yaw):
        self.id = session_id
        self.run = Path(run)
        self.stage = stage
        self.start = list(start)
        self.goal = list(goal)
        self.yaw = float(yaw)
        self.status = "starting"
        self.error = ""
        self.frame = None
        self.frame_seq = 0
        self.info = {"time": 0.0, "distance": float(np.linalg.norm(np.asarray(goal) - start))}
        self.started = time.perf_counter()
        self.finished = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name=f"microduck-play-{session_id}", daemon=True
        )

    def start_session(self):
        self.thread.start()
        return self

    def _publish(self, env):
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
            policy = (self.run / self.stage / "policy.zip").resolve()
            if not policy.is_file() or not policy.is_relative_to(self.run.resolve()):
                raise FileNotFoundError("Selected checkpoint policy is not ready")
            torch.set_num_threads(1)
            model = PPO.load(policy, device="cpu")
            env = NavigationEnv(config, render_mode="rgb_array")
            obs, info = env.reset(
                seed=42,
                options={"start": self.start, "goal": self.goal, "yaw": self.yaw},
            )
            with self.lock:
                self.status = "running"
                self.info = dict(info)
            self._publish(env)
            for _ in range(round(config.episode_seconds / config.control_interval)):
                loop_started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                self._publish(env)
                with self.lock:
                    self.info = {**info, "reward": float(reward)}
                if terminated or truncated:
                    break
                time.sleep(max(0, config.control_interval - (time.perf_counter() - loop_started)))
            with self.lock:
                self.status = "ready"
                self.finished = time.perf_counter()
        except Exception as error:
            with self.lock:
                self.status = "failed"
                self.error = f"{type(error).__name__}: {error}"
                self.finished = time.perf_counter()
        finally:
            if env is not None:
                env.close()

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "start": self.start,
                "goal": self.goal,
                "yaw": self.yaw,
                "frame_seq": self.frame_seq,
                "info": dict(self.info),
                "error": self.error,
                "elapsed_seconds": (self.finished or time.perf_counter()) - self.started,
            }

    def frame_bytes(self):
        with self.lock:
            return self.frame


def create_session(run, stage, start, goal, yaw):
    session_id = time.strftime("play-%H%M%S-") + uuid.uuid4().hex[:5]
    return PlaygroundSession(session_id, run, stage, start, goal, yaw).start_session()
