from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from microduck_ppo.env import BalanceConfig, MicroDuckBalanceEnv
from microduck_ppo.multi_agent import CoordinationConfig, MultiDuckCoordinationEnv
from microduck_ppo.tasks import make_task


def test_toy_environment_contract_and_nonuniform_platform():
    env = MicroDuckBalanceEnv(BalanceConfig(robot_xml=Path("/missing/microduck.xml"), episode_seconds=0.2))
    assert env.observation_space.shape == (59,)
    assert env.action_space.shape == (14,)
    obs, info = env.reset(seed=3)
    assert obs.shape == (59,)
    assert info["backend"] == "mujoco_toy"
    omegas = []
    for _ in range(4):
        obs, reward, terminated, truncated, metrics = env.step(np.zeros(14, dtype=np.float32))
        omegas.append(metrics["omega"])
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
    assert len(set(np.round(omegas, 5))) > 1
    env.close()


def test_env_checker_for_toy_model():
    env = MicroDuckBalanceEnv(BalanceConfig(robot_xml=Path("/missing/microduck.xml"), episode_seconds=0.2))
    check_env(env, skip_render_check=True)
    env.close()


def test_recovery_schedule_is_seeded_and_reported():
    config = BalanceConfig(
        robot_xml=Path("/missing/microduck.xml"), episode_seconds=0.4,
        disturbance_enabled=True, disturbance_seed=17, disturbance_count=2,
        disturbance_start=0.0, disturbance_interval=0.1,
    )
    first = MicroDuckBalanceEnv(config)
    _, info_a = first.reset(seed=5)
    _, info_b = first.reset(seed=99)
    assert info_a["disturbances"] == info_b["disturbances"]
    _, _, _, _, metrics = first.step(np.zeros(14, dtype=np.float32))
    assert metrics["disturbance_count"] == 2
    first.close()


def test_terrain_and_task_factory_modes():
    for terrain in ("flat", "rough", "roller"):
        env = make_task(
            "terrain", robot_xml=Path("/missing/microduck.xml"), terrain=terrain,
            roughness=0.01, episode_seconds=0.1,
        )
        obs, info = env.reset(seed=2)
        assert info["terrain"] == terrain
        assert obs.shape == (59,)
        env.close()


def test_multi_duck_coordination_contract():
    env = MultiDuckCoordinationEnv(CoordinationConfig(ducks=3, episode_seconds=0.2))
    obs, info = env.reset(seed=4)
    assert obs.shape == (29,)
    assert env.action_space.shape == (9,)
    obs, reward, terminated, truncated, metrics = env.step(np.zeros(9, dtype=np.float32))
    assert np.isfinite(obs).all() and np.isfinite(reward)
    assert "formation_error" in metrics and info["ducks"] == 3
    env.close()
