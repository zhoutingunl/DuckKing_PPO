"""Evaluate a trained policy and save machine-readable balance metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .tasks import make_task


def evaluate(
    model_path: Path,
    output: Path,
    episodes: int = 10,
    curriculum: float = 1.0,
    robot_xml: Path | None = None,
    episode_seconds: float = 12.0,
    task: str = "balance",
    terrain: str = "turntable",
    roughness: float = 0.0,
    terrain_friction: float = 1.0,
    disturbance: bool = False,
    ducks: int = 2,
) -> dict:
    env = make_task(
        task, robot_xml=robot_xml, curriculum=curriculum, episode_seconds=episode_seconds,
        terrain=terrain, roughness=roughness, terrain_friction=terrain_friction,
        disturbance_enabled=disturbance, ducks=ducks,
    )
    model = PPO.load(str(model_path), env=env)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        total = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
        rows.append(
            {
                "episode": episode,
                "return": total,
                "steps": steps,
                "completed": float(truncated and not terminated),
                **info,
            }
        )
    env.close()
    result = {
        "model": str(model_path),
        "episodes": episodes,
        "curriculum": curriculum,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "survival_rate": float(np.mean([r["completed"] for r in rows])),
        "success_rate": float(np.mean([r.get("success", 0.0) for r in rows])),
        "mean_goal_error": float(np.mean([r.get("goal_error", np.nan) for r in rows])),
        "mean_collisions": float(np.mean([r.get("collisions", 0.0) for r in rows])),
        "episodes_detail": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--task", choices=["balance", "recovery", "terrain", "coordination"], default="balance")
    parser.add_argument("--terrain", choices=["turntable", "flat", "rough", "roller"], default="turntable")
    parser.add_argument("--roughness", type=float, default=0.0)
    parser.add_argument("--terrain-friction", type=float, default=1.0)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--ducks", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.model, args.output, args.episodes, args.curriculum, args.robot_xml,
        args.episode_seconds, args.task, args.terrain, args.roughness,
        args.terrain_friction, args.disturbance, args.ducks,
    ), indent=2))


if __name__ == "__main__":
    main()
