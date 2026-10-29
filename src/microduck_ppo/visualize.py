"""Render a trained PPO episode and plot the learned balance telemetry."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .tasks import make_task


def render_episode(
    model_path: Path,
    output_dir: Path,
    curriculum: float = 1.0,
    robot_xml: Path | None = None,
    seed: int = 42,
    episode_seconds: float = 12.0,
    task: str = "balance",
    terrain: str = "turntable",
    roughness: float = 0.0,
    terrain_friction: float = 1.0,
    disturbance: bool = False,
    ducks: int = 2,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_task(
        task, robot_xml=robot_xml, curriculum=curriculum, episode_seconds=episode_seconds,
        terrain=terrain, roughness=roughness, terrain_friction=terrain_friction,
        disturbance_enabled=disturbance, ducks=ducks, render_mode="rgb_array",
    )
    model = PPO.load(str(model_path), env=env)
    obs, _ = env.reset(seed=seed)
    frames = []
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        rows.append({"time": info.get("time", len(rows)), "reward": reward, **info})
    env.close()
    video = output_dir / "microduck_balance.mp4"
    if frames:
        imageio.mimsave(video, frames, fps=50, macro_block_size=1)
    csv_path = output_dir / "telemetry.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    times = [row["time"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    primary = "upright" if "upright" in rows[0] else "goal_error"
    axes[0].plot(times, [row[primary] for row in rows], label=primary)
    if primary == "upright":
        axes[0].axhline(0.9, color="green", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel(primary)
    axes[0].legend()
    secondary = "omega" if "omega" in rows[0] else "formation_error"
    axes[1].plot(times, [row[secondary] for row in rows], label=secondary, color="tab:orange")
    axes[1].set_ylabel(secondary)
    axes[1].legend()
    tertiary = "slip" if "slip" in rows[0] else "collisions"
    axes[2].plot(times, [row[tertiary] for row in rows], label=tertiary, color="tab:red")
    if "angular_velocity" in rows[0]:
        axes[2].plot(times, [row["angular_velocity"] for row in rows], label="body angular velocity", color="tab:purple")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("penalty terms")
    axes[2].legend()
    fig.suptitle(f"MicroDuck PPO {task}")
    fig.tight_layout()
    plot_path = output_dir / "learning_process.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return {"video": str(video), "plot": str(plot_path), "telemetry": str(csv_path), "frames": len(frames), "steps": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization"))
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--task", choices=["balance", "recovery", "terrain", "coordination"], default="balance")
    parser.add_argument("--terrain", choices=["turntable", "flat", "rough", "roller"], default="turntable")
    parser.add_argument("--roughness", type=float, default=0.0)
    parser.add_argument("--terrain-friction", type=float, default=1.0)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--ducks", type=int, default=2)
    args = parser.parse_args()
    print(render_episode(
        args.model, args.output_dir, args.curriculum, args.robot_xml, args.seed,
        args.episode_seconds, args.task, args.terrain, args.roughness,
        args.terrain_friction, args.disturbance, args.ducks,
    ))


if __name__ == "__main__":
    main()
