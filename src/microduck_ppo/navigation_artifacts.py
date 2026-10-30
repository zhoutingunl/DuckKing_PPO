"""Atomic local run records, checkpoint evaluation, and streamed video export."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from .navigation import NavigationConfig, NavigationEnv


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(model, env, seeds, *, fixed=False):
    outcomes = []
    for seed in seeds:
        options = (
            {"goal": [0.55, 0], "yaw": float(np.random.default_rng(seed).uniform(-0.1, 0.1))}
            if fixed
            else None
        )
        obs, _ = env.reset(seed=seed, options=options)
        total = 0
        for _ in range(math.ceil(env.config.episode_seconds / env.config.control_interval)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        outcomes.append({"seed": int(seed), "return": float(total), **info})
    successes = sum(row["is_success"] for row in outcomes)
    n = len(outcomes)
    p = successes / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "episodes": n,
        "success_rate": p,
        "successes": successes,
        "success_ci95": [center - width, center + width],
        "mean_return": float(np.mean([r["return"] for r in outcomes])),
        "mean_distance": float(np.mean([r["distance"] for r in outcomes])),
        "fall_rate": sum(r["fallen"] for r in outcomes) / n,
        "timeout_rate": sum(r["failure_reason"] == "timeout" for r in outcomes) / n,
        "mean_sim_seconds": float(np.mean([r["time"] for r in outcomes])),
        "outcomes": outcomes,
    }


def render_showcase(model, config, folder, seed=42, *, deterministic=True, reset_options=None):
    env = NavigationEnv(config, render_mode="rgb_array")
    rows = []
    fps = 1 / config.control_interval
    prefix = "" if deterministic else "exploration_"
    try:
        obs, initial = env.reset(seed=seed, options=reset_options)
        # Seed action sampling separately from the environment; the exact same
        # sequence of random draws is used at every checkpoint.
        torch.manual_seed(43)
        with imageio.get_writer(
            folder / f"{prefix}rollout.mp4", fps=fps, codec="libx264", quality=7, macro_block_size=1
        ) as writer:
            frame = env.render()
            writer.append_data(frame)
            imageio.imwrite(folder / f"{prefix}poster.jpg", frame)
            rows.append({**initial, "reward": 0})
            for _ in range(math.ceil(config.episode_seconds / config.control_interval)):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                writer.append_data(env.render())
                rows.append({**info, "reward": reward, "action": action.tolist()})
                if terminated or truncated:
                    break
        write_json(folder / f"{prefix}trajectory.json", rows)
    finally:
        env.close()
    return {
        "video": str((folder / f"{prefix}rollout.mp4").relative_to(folder.parent.parent)),
        "poster": str((folder / f"{prefix}poster.jpg").relative_to(folder.parent.parent)),
        "trajectory": str((folder / f"{prefix}trajectory.json").relative_to(folder.parent.parent)),
        "seed": seed,
        "action_seed": 43,
        "deterministic": deterministic,
        "fps": fps,
        "duration_seconds": len(rows) / fps,
        "outcome": rows[-1],
    }


def artifact_worker(run_path: str, config_payload: dict, queue):
    """One independent process owns evaluation/video files, never the learner env."""
    torch.set_num_threads(1)
    run = Path(run_path)
    config = NavigationConfig(**config_payload)
    env = NavigationEnv(config)
    try:
        while True:
            job = queue.get()
            if job is None:
                break
            folder = run / job["directory"]
            record = {**job, "status": "evaluating"}
            write_json(folder / "stage.json", record)
            started = time.perf_counter()
            try:
                model = PPO.load(folder / "policy.zip", device="cpu")
                record["evaluation"] = evaluate(model, env, range(2000, 2008))
                record["showcase"] = render_showcase(model, config, folder)
                record["exploration"] = render_showcase(model, config, folder, deterministic=False)
                record["status"] = "ready"
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            record["artifact_seconds"] = time.perf_counter() - started
            write_json(folder / "stage.json", record)
    finally:
        env.close()


def run_summary(run: Path):
    payload = read_json(run / "run.json")
    payload["id"] = run.name
    stages = []
    for stage in payload.get("snapshots", []):
        path = run / stage["directory"] / "stage.json"
        stages.append(read_json(path) if path.exists() else {**stage, "status": "queued"})
    payload["stages"] = stages
    results = run / "acceptance.json"
    if results.exists():
        payload["acceptance"] = read_json(results)
    return payload
