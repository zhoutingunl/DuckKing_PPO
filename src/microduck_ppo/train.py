"""Train PPO and export TensorBoard-compatible logs plus curriculum metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from .callbacks import CurriculumCallback
from .tasks import make_task


def train(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    robot_xml = Path(args.robot_xml).resolve() if args.robot_xml else None
    raw_env = make_task(
        args.task, robot_xml=robot_xml, episode_seconds=args.episode_seconds,
        curriculum=args.curriculum, terrain=args.terrain, roughness=args.roughness,
        terrain_friction=args.terrain_friction, slope=args.slope,
        disturbance_enabled=args.disturbance, disturbance_seed=args.disturbance_seed,
        ducks=args.ducks,
    )
    if args.check_env:
        check_env(raw_env, warn=True)
    raw_env.close()
    env = Monitor(make_task(
        args.task, robot_xml=robot_xml, episode_seconds=args.episode_seconds,
        curriculum=args.curriculum, terrain=args.terrain, roughness=args.roughness,
        terrain_friction=args.terrain_friction, slope=args.slope,
        disturbance_enabled=args.disturbance, disturbance_seed=args.disturbance_seed,
        ducks=args.ducks,
    ))
    eval_env = Monitor(make_task(
        args.task, robot_xml=robot_xml, episode_seconds=args.episode_seconds,
        curriculum=args.curriculum, terrain=args.terrain, roughness=args.roughness,
        terrain_friction=args.terrain_friction, slope=args.slope,
        disturbance_enabled=args.disturbance, disturbance_seed=args.disturbance_seed,
        ducks=args.ducks,
    ))
    callback = CurriculumCallback(start=args.curriculum, increment=args.curriculum_step, window=args.curriculum_window)
    callbacks = CallbackList(
        [
            callback,
            CheckpointCallback(save_freq=max(1, args.checkpoint_every), save_path=str(output / "checkpoints"), name_prefix="microduck_ppo"),
            EvalCallback(eval_env, best_model_save_path=str(output / "best"), log_path=str(output / "eval"), eval_freq=max(1, args.eval_every), deterministic=True),
        ]
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(output / "tensorboard"),
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(str(output / f"microduck_{args.task}_ppo"))
    callback.save_history(output / "curriculum.json")
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_payload["backend"] = env.env.__class__.__name__
    config_payload["observation_dim"] = int(env.observation_space.shape[0])
    config_payload["action_dim"] = int(env.action_space.shape[0])
    (output / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    env.close()
    eval_env.close()
    return output / f"microduck_{args.task}_ppo.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("runs/microduck_balance"))
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--task", choices=["balance", "recovery", "terrain", "coordination"], default="balance")
    parser.add_argument("--terrain", choices=["turntable", "flat", "rough", "roller"], default="turntable")
    parser.add_argument("--roughness", type=float, default=0.0)
    parser.add_argument("--terrain-friction", type=float, default=1.0)
    parser.add_argument("--slope", type=float, default=0.0)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--disturbance-seed", type=int)
    parser.add_argument("--ducks", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--curriculum", type=float, default=0.15)
    parser.add_argument("--curriculum-step", type=float, default=0.08)
    parser.add_argument("--curriculum-window", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    print(f"saved: {train(args)}")


if __name__ == "__main__":
    main()
