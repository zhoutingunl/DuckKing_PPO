"""Local CPU PPO navigation, preserving the complete policy evolution."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .navigation import NavigationConfig, NavigationEnv, resolve_assets
from .navigation_artifacts import (
    artifact_worker,
    evaluate,
    run_summary,
    sha256,
    write_json,
)


class EvolutionCallback(BaseCallback):
    def __init__(
        self,
        run,
        manifest,
        queue,
        every,
        *,
        dense_until=32_000,
        late_every=10_240,
        target_success_rate=1.0,
        target_window=2,
    ):
        super().__init__()
        self.run, self.manifest, self.queue, self.every = run, manifest, queue, every
        self.next_snapshot = every
        self.started = time.perf_counter()
        self.returns = []
        self.last_write = 0
        self.target_success_rate = float(target_success_rate)
        self.target_window = int(target_window)
        self.dense_until = int(dense_until)
        self.late_every = int(late_every)
        self.successes = []
        self.convergence_step = None
        self.convergence_checkpoint = None
        self.convergence_streak = 0
        self.residuals = []
        self.action_deltas = []
        self.last_artifact_check = 0.0

    def snapshot(self, kind):
        updates = int(self.model._n_updates)
        step = int(self.model.num_timesteps)
        folder = self.run / "stages" / f"{step:09d}"
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / "policy.zip")
        record = {
            "timesteps": step,
            "updates": updates,
            "kind": kind,
            "directory": str(folder.relative_to(self.run)),
            "policy_sha256": sha256(folder / "policy.zip"),
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        self.manifest["snapshots"] = [
            item for item in self.manifest["snapshots"] if item.get("directory") != record["directory"]
        ]
        self.manifest["snapshots"].append(record)
        self.flush()
        self.queue.put(record)

    def flush(self):
        elapsed = time.perf_counter() - self.started
        self.manifest.update(
            timesteps=int(self.model.num_timesteps),
            updates=int(self.model._n_updates),
            elapsed_seconds=elapsed,
            steps_per_second=self.model.num_timesteps
            / max(self.manifest.get("training_seconds", elapsed), 0.001),
            recent_return=float(np.mean(self.returns[-40:])) if self.returns else None,
            recent_success_rate=(float(np.mean(self.successes[-32:])) if self.successes else None),
            recent_action_residual=(
                float(np.mean(self.residuals[-256:])) if self.residuals else None
            ),
            recent_action_delta=(
                float(np.mean(self.action_deltas[-256:])) if self.action_deltas else None
            ),
        )
        write_json(self.run / "run.json", self.manifest)
        self.last_write = time.perf_counter()

    def _on_rollout_start(self):
        # SB3 _on_step runs before PPO updates. At the next rollout start the
        # completed update is actually present in the saved policy.
        if self.num_timesteps >= self.next_snapshot:
            self.snapshot("checkpoint")
            interval = self.every if self.num_timesteps < self.dense_until else self.late_every
            self.next_snapshot = (self.num_timesteps // interval + 1) * interval

    def _check_artifact_convergence(self):
        """Stop from a completed checkpoint evaluation, rather than raw reward."""
        now = time.perf_counter()
        if now - self.last_artifact_check < 0.5 or self.convergence_step is not None:
            return
        self.last_artifact_check = now
        for snapshot in self.manifest["snapshots"]:
            if snapshot.get("kind") != "checkpoint":
                continue
            stage_path = self.run / snapshot["directory"] / "stage.json"
            if not stage_path.is_file():
                continue
            try:
                stage = json.loads(stage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            evaluation = stage.get("evaluation") or {}
            if (
                stage.get("status") == "ready"
                and evaluation.get("success_rate", 0.0) >= self.target_success_rate
            ):
                self.convergence_step = int(self.model.num_timesteps)
                self.convergence_checkpoint = int(snapshot["timesteps"])
                self.manifest["convergence"] = {
                    "timesteps": self.convergence_step,
                    "checkpoint_timesteps": int(snapshot["timesteps"]),
                    "success_rate": float(evaluation["success_rate"]),
                    "window_episodes": int(evaluation.get("episodes", 0)),
                    "early_stop": True,
                    "source": "checkpoint_validation",
                }
                return

    def _on_step(self):
        self._check_artifact_convergence()
        for info in self.locals.get("infos", []):
            if "action_residual" in info:
                self.residuals.append(float(info["action_residual"]))
            if "action_delta" in info:
                self.action_deltas.append(float(info["action_delta"]))
            if "episode" in info:
                self.returns.append(info["episode"]["r"])
                self.successes.append(bool(info.get("is_success", False)))
                window = self.successes[-32:]
                if len(window) >= 32 and np.mean(window) >= self.target_success_rate:
                    self.convergence_streak += 1
                    if self.convergence_streak >= self.target_window and self.convergence_step is None:
                        self.convergence_step = int(self.model.num_timesteps)
                        self.manifest["convergence"] = {
                            "timesteps": self.convergence_step,
                            "success_rate": float(np.mean(window)),
                            "window_episodes": len(window),
                            "streak_windows": self.convergence_streak,
                            "early_stop": True,
                        }
                else:
                    self.convergence_streak = 0
                with (self.run / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "timesteps": self.num_timesteps,
                                **info["episode"],
                                "is_success": info.get("is_success", False),
                                "failure_reason": info.get("failure_reason", ""),
                                "action_residual": info.get("action_residual"),
                                "action_delta": info.get("action_delta"),
                            }
                        )
                        + "\n"
                    )
        if time.perf_counter() - self.last_write >= 1:
            self.flush()
        return not (self.run / "STOP").exists() and self.convergence_step is None


def build_model(config, *, envs=1, n_steps=256, seed=7, threads=1):
    torch.set_num_threads(threads)
    env = make_vec_env(
        partial(NavigationEnv, config),
        n_envs=envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if envs > 1 else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if envs > 1 else {},
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=128,
        n_epochs=6,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    return model, env


def train(args):
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "run.json").exists():
        raise FileExistsError(
            "Use a new output directory to preserve the original policy and history"
        )
    if args.timesteps < 1 or args.envs < 1 or args.checkpoint_every < 1 or args.eval_episodes < 1:
        raise ValueError("Training/evaluation budgets and environment count must be positive")
    config = NavigationConfig(
        robot_xml=args.robot_xml,
        gait_policy=args.gait_policy,
        episode_seconds=args.episode_seconds,
        reward_mode=args.reward_mode,
    )
    robot, gait = resolve_assets(config)
    config = NavigationConfig(
        **{**asdict(config), "robot_xml": str(robot), "gait_policy": str(gait)}
    )
    manifest = {
        "task": "navigation",
        "status": "starting",
        "created_at": time.time(),
        "pid": os.getpid(),
        "config": asdict(config),
        "training": vars(args),
        "snapshots": [],
        "budget": args.timesteps,
        "timesteps": 0,
        "updates": 0,
        "control": "Frozen velstand.onnx 50 Hz → 14 joint actuators; PPO navigation 10 Hz",
        "assets": {"robot_sha256": sha256(robot), "gait_sha256": sha256(gait)},
        "evaluation_protocol": {
            "showcase_seed": 42,
            "validation_seeds": [2000, 2007],
            "holdout_seeds": [10000, 10000 + args.eval_episodes - 1],
            "fixed_target": [0.55, 0],
            "fixed_yaw_range": [-0.1, 0.1],
            "fixed_target_success": 0.9,
            "random_target_success": 0.6,
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "versions": {
            name: version(name)
            for name in ("mujoco", "torch", "stable-baselines3", "onnxruntime", "gymnasium")
        },
        "normalization": "No running observation/reward normalization",
        "initialization_seed": args.seed,
    }
    write_json(run / "run.json", manifest)
    model = env = worker = None
    started = time.perf_counter()
    try:
        model, env = build_model(
            config, envs=args.envs, seed=args.seed, n_steps=args.n_steps, threads=args.threads
        )
        model.tensorboard_log = str(run / "tensorboard")
        context = mp.get_context("spawn")
        queue = context.Queue()
        worker = context.Process(target=artifact_worker, args=(str(run), asdict(config), queue))
        worker.start()
        manifest["status"] = "training"
        callback = EvolutionCallback(
            run,
            manifest,
            queue,
            args.checkpoint_every,
            dense_until=args.dense_until,
            late_every=args.late_checkpoint_every,
            target_success_rate=args.convergence_success_rate,
            target_window=args.convergence_windows,
        )
        callback.model = model
        callback.snapshot("initial")
        model.learn(total_timesteps=args.timesteps, callback=callback)
        manifest["training_seconds"] = time.perf_counter() - callback.started
        stopped = (run / "STOP").exists()
        converged = callback.convergence_step is not None and not stopped
        callback.snapshot("converged" if converged else "stopped" if stopped else "final")
        if converged:
            manifest["status"] = "converged"
        manifest["status"] = "rendering"
        callback.flush()
        queue.put(None)
        # Keep the web page responsive while the separate exporter finishes.
        while worker.is_alive():
            worker.join(timeout=1)
        manifest["training_and_artifact_seconds"] = time.perf_counter() - callback.started
        if worker.exitcode != 0:
            raise RuntimeError(f"Artifact worker exited with code {worker.exitcode}")
        stages = run_summary(run)["stages"]
        failed = [s for s in stages if s["status"] != "ready"]
        if failed:
            raise RuntimeError(
                f"Checkpoint export failed: {failed[0].get('error', 'worker did not finish')}"
            )
        best = max(
            stages, key=lambda s: (s["evaluation"]["success_rate"], s["evaluation"]["mean_return"])
        )
        manifest["best_directory"] = best["directory"]
        if not stopped:
            manifest["status"] = "evaluating"
            callback.flush()
            evaluation_started = time.perf_counter()
            eval_env = NavigationEnv(config)
            try:
                selected = PPO.load(run / best["directory"] / "policy.zip", device="cpu")
                seeds = range(10000, 10000 + args.eval_episodes)
                results = {
                    "selected_directory": best["directory"],
                    "selection": "validation success rate, then validation return; never holdout",
                    "fixed": evaluate(selected, eval_env, seeds, fixed=True),
                    "random": evaluate(selected, eval_env, seeds),
                }
            finally:
                eval_env.close()
            results["passed"] = (
                args.eval_episodes >= 100
                and results["fixed"]["success_rate"] >= 0.9
                and results["random"]["success_rate"] >= 0.6
            )
            results["qualification"] = (
                "passed"
                if results["passed"]
                else "insufficient_episodes"
                if args.eval_episodes < 100
                else "below_target"
            )
            write_json(run / "acceptance.json", results)
            manifest["acceptance_seconds"] = time.perf_counter() - evaluation_started
            manifest["status"] = "passed" if results["passed"] else "unqualified"
        else:
            manifest["status"] = "stopped"
        manifest["total_wall_seconds"] = time.perf_counter() - started
        callback.flush()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(run / "run.json", manifest)
        if worker is not None and worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        raise
    finally:
        if env is not None:
            env.close()
    print(
        json.dumps(
            {
                "run": str(run),
                "status": manifest["status"],
                "seconds": manifest["total_wall_seconds"],
            }
        )
    )


def benchmark(args):
    config = NavigationConfig(
        robot_xml=args.robot_xml, gait_policy=args.gait_policy, reward_mode=args.reward_mode
    )
    results = []
    for count in args.benchmark_envs:
        started = time.perf_counter()
        model, env = build_model(
            config, envs=count, seed=args.seed, n_steps=args.n_steps, threads=args.threads
        )
        initialization = time.perf_counter() - started
        try:
            started = time.perf_counter()
            model.learn(total_timesteps=args.timesteps)
            elapsed = time.perf_counter() - started
            rate = model.num_timesteps / elapsed
            row = {
                "envs": count,
                "timesteps": model.num_timesteps,
                "initialization_seconds": initialization,
                "training_seconds": elapsed,
                "steps_per_second": rate,
                "100k_training_minutes": 100000 / rate / 60,
                "1m_training_minutes": 1000000 / rate / 60,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    write_json(Path(args.output) / "benchmark.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/navigation")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=2_560)
    parser.add_argument("--dense-until", type=int, default=32_000)
    parser.add_argument("--late-checkpoint-every", type=int, default=10_240)
    parser.add_argument("--episode-seconds", type=float, default=12)
    parser.add_argument(
        "--reward-mode",
        choices=["distance", "direct"],
        default="distance",
        help="导航目标函数：距离优先，或鼓励面向目标正向接近",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--convergence-success-rate", type=float, default=1.0,
        help="提前停止所需的最近 32 回合成功率",
    )
    parser.add_argument(
        "--convergence-windows", type=int, default=2,
        help="连续满足成功率的窗口数",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robot-xml")
    parser.add_argument("--gait-policy")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-envs", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    benchmark(args) if args.benchmark else train(args)


if __name__ == "__main__":
    main()
