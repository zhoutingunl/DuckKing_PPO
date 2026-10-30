# MicroDuck PPO

## 单鸭导航与全阶段可视化

当前可验证的完整闭环是 **平地单鸭导航**：PPO 从随机初始化开始学习前进与转向，冻结的 Pollen `velstand.onnx` 在 50 Hz 下驱动 14 个关节；MuJoCo 物理频率 200 Hz，导航决策 10 Hz。没有向躯干施加推进力，也没有重新训练底层步态。需要外部官方资产，缺失时明确报错。

```bash
uv sync --extra dev
uv run microduck-viewer --port 8780
```

打开 **http://127.0.0.1:8780/**，点击“开始新训练”。也可从另一终端启动：

```bash
uv run microduck-navigation --timesteps 100000 --envs 4 --output runs/navigation-example
```

训练记录完成后，页面的“观众实验”允许拖动橙色 3D 小鸭和蓝色零高度圆形地毯，设置初始朝向，并选择 0 步、约 10K、20K、30K ……或最终检查点。点击“运行这个场景”会启动一段实时的本地 MuJoCo 仿真；浏览器每 0.1 秒接收当前帧和状态，不生成观众回放 MP4。仿真结束后显示抵达、摔倒或超时，可以继续拖动位置、换策略并重复运行。

这不是把小鸭瞬移到画面上：起点和目标会传入 `NavigationEnv.reset(options=...)`，蓝色地毯是 MuJoCo 中的真实零高度圆柱几何，动作仍由选中 PPO 策略通过冻结步态和 14 个关节执行。实时会话只保存在网页服务进程内存中，服务重启后结束；网页只监听 `127.0.0.1`。训练阶段的历史 MP4 仍保存在 `stages/`，用于学习过程对照。

默认从相邻 `../DuckKing/external/` 发现官方模型和步态。其他路径可通过 `--robot-xml`、`--gait-policy` 或环境变量 `MICRODUCK_ROBOT_XML`、`MICRODUCK_GAIT_POLICY` 指定。官方资产和原策略来源为 [Pollen Robotics microduck_rl](https://github.com/pollen-robotics/microduck_rl)，控制接口参考 DuckKing `simulation.Driver`；资产不复制到本仓库。

网页会保存、展示：

- **训练前 0 步策略**、约每 1 万步检查点、最终策略和验证集最佳策略。保存检查点时记录真实 PPO 优化 epoch 次数，不把尚未更新的采样步误称为已经学过。
- 两种回放：**带探索采样**展示随机尝试，**确定性评估**展示策略均值。初始策略均值可能接近静止，这也是实际结果。所有阶段共享场景种子 42；探索采样种子 43；固定镜头、初始状态和回合上限。
- 阶段滑块、同步对照初始策略、慢放、路径、验证成功率、实时步数与最近回报。视频按实际 10 Hz 导出，失败回合不删除。
- 独立进程生成检查点评估和视频，网页每 3 秒刷新；“停止并保留记录”在当前 step 后停止并保留回放。导出失败会显示失败，不会当作训练通过。

任务成功要求持续 0.3 秒进入半径 0.10 米的目标区。训练目标距起点 0.35–0.65 米，方位和初始朝向随机；回合上限 12 秒，无障碍、单鸭、平地。每个检查点使用种子 2000–2007 的 8 回合验证集选择最佳策略；训练结束后，用另外种子 10000–10099 分别测试固定布局和随机布局各 100 回合，报告成功率、Wilson 95% 区间、摔倒率与超时率。通过标准为固定 ≥90%、随机 ≥60%；不足 100 回合仅算功能验证。最佳与最终策略分别保留，预算完成不等于达标。

每次训练使用新的输出目录，防止覆盖初始模型。`run.json` 保存配置、资产 SHA256、种子和阶段记录；`episodes.jsonl` 保存回合记录；`stages/*/` 保存策略、两种 MP4、轨迹和评估；`acceptance.json` 保存独立验收。TensorBoard 包含 PPO 更新指标：

```bash
uv run tensorboard --logdir runs/navigation-example/tensorboard
uv run microduck-navigation --benchmark --benchmark-envs 1 2 4 --timesteps 4096 --output outputs/navigation-benchmark
uv run python -m pytest -q
uv run ruff check src tests
```

本机 i7-9750H、6 核 12 线程、32GB、无 GPU 的实测见 [本地验证记录](docs/navigation-validation.json)。10 万步含阶段视频和 200 回合验收约 3 分钟；100 万步规划 25–40 分钟，属于吞吐外推，不是达标保证。长时间运行、CPU 负载、模型接触复杂度、视频数量都会改变耗时。

所有训练、评估、浏览器验证在本地执行，网页仅监听 `127.0.0.1`，无 CDN。`runs/` 与视频在本地保留，不自动上传。当前未实现导航断点续训、端到端视觉、夹取和运输；它们是后续阶段。

## 原有平衡、地形与协作原型

**以下属于早期实验原型，已有 smoke 回放不能证明策略收敛。** 多鸭原型使用躯干外力推进，不等价于关节行走；转盘运动、接触与课程学习仍需修正和独立物理验证，不作为导航验收成果。

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
- `recovery` 任务提供固定种子的推搡/力矩事件；尚无完整恢复成功率基准。
- `terrain` 任务支持 `flat`、`rough`、低摩擦和 `roller` 模式；官方轮滑 MJCF 会自动选择被动轮模型。
- `coordination` 任务是 2–4 鸭的集中式 PPO 接口原型，当前动作施加到躯干外力，尚未接入关节步态。

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
