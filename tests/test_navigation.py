"""Validate real actuation and reproducible evaluation, without toy substitutions."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from microduck_ppo.navigation import NavigationConfig, NavigationEnv, resolve_assets
from microduck_ppo.navigation_artifacts import run_summary, write_json
from microduck_ppo.viewer import confined


@pytest.fixture
def env():
    config = NavigationConfig()
    try:
        resolve_assets(config)
    except FileNotFoundError as error:
        pytest.skip(f"Official external assets unavailable: {error}")
    result = NavigationEnv(config)
    yield result
    result.close()


def test_official_assets_are_required():
    with pytest.raises(FileNotFoundError):
        NavigationEnv(NavigationConfig(robot_xml="/missing/robot.xml"))


def test_navigation_contract_and_seeded_reset(env):
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=11)
    env.step(np.array([0.5, 0.5]))
    again, repeated = env.reset(seed=11)
    np.testing.assert_allclose(first, again, atol=1e-7)
    assert info == repeated


def test_bounded_joint_actuation_moves_robot_without_body_force(env):
    env.reset(seed=5, options={"yaw": 0, "goal": [0.55, 0]})
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
        assert not env.data.xfrc_applied.any()
        assert not info["fallen"]
        if terminated or truncated:
            break
    assert info["x"] > 0.2
    assert np.max(np.abs(env.data.ctrl[env.ids])) > 0.1
    env.reset(seed=5)
    clipped = env.step(np.array([1.0, -1.0]))[0]
    env.reset(seed=5)
    excess = env.step(np.array([9.0, -8.0]))[0]
    np.testing.assert_allclose(clipped, excess)
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0]))


def test_success_requires_continuous_hold_and_timeout_is_not_success(env):
    env.reset(seed=2, options={"yaw": 0, "goal": [0, 0]})
    for _ in range(2):
        _, _, terminated, _, info = env.step(np.zeros(2))
        assert not terminated and not info["is_success"]
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated and info["is_success"]
    env.config = replace(env.config, episode_seconds=0.2)
    env.reset(seed=2, options={"yaw": 0, "goal": [0.55, 0]})
    env.step(np.zeros(2))
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert truncated and not terminated and not info["is_success"]
    assert info["failure_reason"] == "timeout"


def test_manifest_retains_unready_and_failed_stages(tmp_path):
    write_json(
        tmp_path / "run.json",
        {
            "snapshots": [
                {"directory": "stages/0", "timesteps": 0},
                {"directory": "stages/1", "timesteps": 1},
            ]
        },
    )
    write_json(tmp_path / "stages/1/stage.json", {"status": "failed", "error": "encoder failed"})
    stages = run_summary(tmp_path)["stages"]
    assert stages[0]["status"] == "queued"
    assert stages[1]["status"] == "failed"
    assert stages[1]["error"] == "encoder failed"


def test_dashboard_paths_cannot_escape_runs(tmp_path):
    with pytest.raises(ValueError):
        confined(tmp_path, "../secret")
    (tmp_path / "link").symlink_to(Path("/tmp"))
    with pytest.raises(ValueError):
        confined(tmp_path, "link/secret")
