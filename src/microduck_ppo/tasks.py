"""Task factory shared by training and evaluation entry points."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .env import BalanceConfig, MicroDuckBalanceEnv
from .multi_agent import CoordinationConfig, MultiDuckCoordinationEnv


def make_task(
    task: str,
    *,
    robot_xml: Path | None = None,
    episode_seconds: float = 12.0,
    curriculum: float = 0.15,
    terrain: str = "turntable",
    roughness: float = 0.0,
    terrain_friction: float = 1.0,
    slope: float = 0.0,
    disturbance_enabled: bool = False,
    disturbance_seed: int | None = None,
    ducks: int = 2,
    render_mode: str | None = None,
) -> gym.Env:
    if task == "coordination":
        return MultiDuckCoordinationEnv(
            CoordinationConfig(
                ducks=ducks, episode_seconds=episode_seconds, robot_xml=robot_xml
            ),
            render_mode=render_mode,
        )
    config = BalanceConfig(
        robot_xml=robot_xml,
        episode_seconds=episode_seconds,
        curriculum=curriculum,
        terrain=terrain,
        roughness=roughness,
        terrain_friction=terrain_friction,
        slope=slope,
        disturbance_enabled=disturbance_enabled or task == "recovery",
        disturbance_seed=disturbance_seed,
    )
    return MicroDuckBalanceEnv(config, render_mode=render_mode)
