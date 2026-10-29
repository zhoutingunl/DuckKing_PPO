"""Stable-Baselines3 callbacks for curriculum and compact metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Increase turntable difficulty after successful rollout windows."""

    def __init__(self, start: float = 0.15, increment: float = 0.08, window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.level = float(start)
        self.increment = float(increment)
        self.window = int(window)
        self._scores: list[float] = []
        self.history: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        self._set_level(self.level)

    def _set_level(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))
        envs = self.training_env.envs if hasattr(self.training_env, "envs") else [self.training_env]
        for env in envs:
            target = env
            while hasattr(target, "env"):
                target = target.env
            if hasattr(target, "set_curriculum"):
                target.set_curriculum(self.level)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos, strict=False):
            if done:
                self._scores.append(float(info.get("episode", {}).get("r", 0.0)))
        if len(self._scores) >= self.window:
            mean_score = float(np.mean(self._scores[-self.window :]))
            self.history.append({"timesteps": float(self.num_timesteps), "difficulty": self.level, "mean_episode_reward": mean_score})
            # The threshold scales with episode length; it prevents a failed early
            # policy from immediately jumping to the hardest turntable.
            if mean_score > 0.65 * 2.25 * 12.0:
                self._set_level(self.level + self.increment)
        return True

    def save_history(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
