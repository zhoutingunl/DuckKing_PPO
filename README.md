# MicroDuck PPO Balance

这是一个独立的 PPO 实验仓库，用于研究仿真 MicroDuck 的平衡、扰动恢复、复杂地形运动和多鸭协作，并导出可视化学习过程。

实现参考 `microduck_mujoco_reinforcement_learning.md` 和 DuckKing 的 MuJoCo 控制接口：14 个策略关节、官方 `HOME` 姿态、200 Hz 物理步进、50 Hz 残差控制。策略输出 `[-1, 1]` 的关节目标残差，由现有 position actuator 执行。

## 功能

- MuJoCo 圆形转盘，倾角由 `platform_tilt_x/y` 定义。
- 非匀速角速度：`ω(t)=ω_center+ω_amplitude·sin(ω_frequency·t+phase)`，平台姿态用积分角度驱动。
- 观测包括身体重力方向、IMU 角速度、关节位置/速度、双脚接触、脚相对转盘滑移、平台角速度与倾角。
- 奖励鼓励直立和双脚接触，惩罚滑移、身体角速度、动作抖动、能量和摔倒。
- 课程学习从低速/低倾角逐步增加到目标扰动；每回合随机化方向、相位、初始姿态。
- `evaluate` 输出存活率和回合指标；`visualize` 输出 MP4、CSV 和 PNG 曲线。
- 找不到官方 MJCF 时自动使用小型 MuJoCo toy 模型，便于本地 CI 验证。真实 MicroDuck 训练请传 `--robot-xml`。
- `recovery` 任务提供固定种子的推搡/力矩事件和恢复指标。
- `terrain` 任务支持 `flat`、`rough`、低摩擦和 `roller` 模式；官方轮滑 MJCF 会自动选择被动轮模型。
- `coordination` 任务支持 2–4 鸭的集中式 PPO baseline，奖励目标到达、编队保持和避碰。

## 本地运行

需要 Python 3.10+ 和 `uv`：

```bash
uv sync
uv run microduck-train --timesteps 100000 --check-env --output runs/microduck_balance
uv run microduck-train --task recovery --disturbance --timesteps 100000 --output runs/microduck_recovery
uv run microduck-train --task terrain --terrain rough --roughness 0.01 --terrain-friction 0.55 --timesteps 100000 --output runs/microduck_rough
uv run microduck-train --task coordination --ducks 2 --timesteps 100000 --output runs/microduck_coordination
uv run microduck-evaluate runs/microduck_balance/microduck_balance_ppo.zip --episodes 10 --output outputs/evaluation.json
uv run microduck-visualize runs/microduck_balance/microduck_balance_ppo.zip --output-dir outputs/visualization
uv run microduck-visualize runs/microduck_recovery/microduck_recovery_ppo.zip --task recovery --disturbance --output-dir outputs/recovery_visualization
```

若官方模型位于 DuckKing 的 `external/microduck_rl`，训练器会自动发现；也可以显式指定：

```bash
uv run microduck-train \
  --robot-xml ../DuckKing/external/microduck_rl/src/mjlab_microduck/robot/microduck/robot_groundcontact.xml \
  --timesteps 100000
```

短验证命令：

```bash
uv run microduck-train --timesteps 1024 --n-steps 128 --eval-every 512 --checkpoint-every 512 --output /tmp/microduck-ppo-smoke
uv run pytest
```

训练日志在 `runs/.../tensorboard`，课程记录在 `curriculum.json`，模型在 `microduck_balance_ppo.zip`。可用 `tensorboard --logdir runs/microduck_balance/tensorboard` 查看 PPO 的 rollout、损失和价值函数曲线；`microduck-visualize` 则生成带物理状态曲线的仿真视频。

## 物理边界

这是仿真训练与评估代码，尚未在实体 MicroDuck 上验证。迁移硬件前应限制扭矩和动作幅度，并使用独立急停与姿态保护。
