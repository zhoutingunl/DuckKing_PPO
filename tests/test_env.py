from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from microduck_ppo.env import BalanceConfig, MicroDuckBalanceEnv


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
