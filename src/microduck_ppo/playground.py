"""Run one audience-selected policy in a user-placed local scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO

from .navigation import NavigationConfig
from .navigation_artifacts import render_showcase, write_json


def run_playground(
    run: Path, stage: str, output: Path, start: list[float], goal: list[float], yaw: float
):
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    stage_path = (run / stage).resolve()
    if not stage_path.is_relative_to(run.resolve()) or stage_path.parent.name != "stages":
        raise ValueError("stage must be a direct stage directory in the selected run")
    policy = stage_path / "policy.zip"
    if not policy.is_file():
        raise FileNotFoundError("Selected checkpoint policy is not ready")
    config = NavigationConfig(**manifest["config"])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "running",
        "stage": stage,
        "start": start,
        "goal": goal,
        "yaw": yaw,
        "created_at": time.time(),
    }
    write_json(output / "playground.json", record)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        model = PPO.load(policy, device="cpu")
        record["showcase"] = render_showcase(
            model,
            config,
            output,
            seed=42,
            deterministic=True,
            reset_options={"start": start, "goal": goal, "yaw": yaw},
        )
        record["status"] = "ready"
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "playground.json", record)
    except Exception as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "playground.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, nargs=2, required=True)
    parser.add_argument("--goal", type=float, nargs=2, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    run_playground(args.run, args.stage, args.output, args.start, args.goal, args.yaw)


if __name__ == "__main__":
    main()
