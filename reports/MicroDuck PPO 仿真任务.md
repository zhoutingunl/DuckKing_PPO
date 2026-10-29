# MicroDuck PPO 下一步任务路线图

当前项目最值得继续做的不是再加一个孤立动作，而是把已有的倾斜非匀速转盘环境扩展成一套可复现实验基准：先测 MicroDuck 受到推搡、转速突变和低摩擦影响时能否恢复，再把同一套观测、奖励分解、评估和视频管线复用于行走、踢球、抓取和多鸭协作。这样每一项工作都会同时产生控制能力、可比较的数据和可视化结果。MicroDuck 上游已经用 PPO 覆盖速度行走、站起、坐站、啄地、踢球、前滚翻、轮滑、坡道和原地旋转等任务，因此我们的主要机会是把这些任务接入 DuckKing 的真实接触场景，建立统一的 PPO 基线与 holdout 评估，而不是重新建一套机器人模型（[microduck_rl README](https://github.com/pollen-robotics/microduck_rl/blob/develop/README.md)）。当前 DuckKing_PPO 的 59 维观测和残差 position-actuator 控制适合 CPU 原型；上游共享的 61 维策略契约、BAM 执行器和 backlash 模型应作为后续兼容目标，而不是在下一步一次性引入全部复杂度。

## 扰动恢复能最快形成可靠的实验基准

第一项建议实现“push recovery benchmark”。在当前转盘环境增加可复现的外力和外力矩事件：随机方向、脉冲持续时间、作用部位、施力时间，以及平台角度/角速度的突变。每回合把扰动种子和事件序列写入 `info`，将“站住”拆成恢复成功、恢复时间、最大倾角、恢复后的脚底接触率、最大可承受冲量和失败原因。已有环境已经拥有身体姿态、IMU 角速度、脚接触、脚底滑移和摔倒判定，因此这个任务主要新增扰动调度、恢复指标和评估脚本。腿式机器人研究也把受控扰动下的恢复率作为核心轴线，而不是只看 episode reward（[dynamic fall recovery](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048123/)，[robust legged dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063321/)）。

应同时保留三个对照：零扰动官方/现有控制器、未训练 PPO、训练后的 PPO。评估环境独立于训练环境，使用固定的确定性推理和 holdout seeds；这是 Stable-Baselines3 对可比较评估的建议（[SB3 evaluation](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)）。最终报告画“扰动幅值—恢复率”曲线，并给出平均值、5 分位稳健分数和典型失败视频。这个任务的研究价值在于它能回答当前策略到底学会了抗扰动，还是只在某一条转盘轨迹上保持直立。

## 速度跟踪、地形和轮滑会把平衡扩展成运动控制

第二阶段是平地速度跟踪。沿用 DuckKing 的 13 维 command 结构，随机采样前进、侧向和偏航速度；观测加入相对目标速度，奖励身体速度/偏航跟踪、直立、脚底接触和低滑移，惩罚动作变化、能量代理和跌倒。上游已经有 `Mjlab-Velocity-{Flat,Rough}-MicroDuck` 与 `VelStand` 任务，DuckKing 也加载 `velstand.onnx`，因此可直接用官方策略做 baseline 和轨迹示范（[microduck_rl README](https://github.com/pollen-robotics/microduck_rl/blob/develop/README.md)，[DuckKing simulation.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/simulation.py)）。这项工作能验证当前残差动作是否足够支撑连续运动，并为后面的地形和多鸭导航提供通用控制器。

随后增加低摩擦、斜坡、台阶和随机高度场。MuJoCo 的接触模型支持切向、扭转和滚动摩擦，因此可以在同一个引擎中构造牵引力、滑移和坡面实验（[MuJoCo computation](https://mujoco.readthedocs.io/en/latest/computation/)）。坡道/轮滑是成本较低的可视化任务：上游已经有 passive-wheel 模型、roller crouch、roller slope 和 spin 任务，DuckKing 也有轮滑场景和轮滑下蹲策略（[microduck_rl README](https://github.com/pollen-robotics/microduck_rl/blob/develop/README.md)，[DuckKing skills_model.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/skills_model.py)）。坡度、摩擦、初始位置应采用课程学习；粗糙地形研究支持用本体感觉和地形随机化提升泛化，但相关公开结果主要来自四足机器人，不能直接当作 MicroDuck 的成功保证（[challenging terrain locomotion](https://arxiv.org/abs/2010.11251)）。

## 踢球、抓取和多鸭协作提供更强的任务演示

第三阶段可以做 kick-to-target，而不是直接从稀疏奖励开始做复杂运球。随机生成球和目标点，在观测中加入身体坐标系下的球相对位置、速度和目标方向；奖励选定脚接触、球沿目标方向的位移、距离误差、踢后恢复站立，惩罚误触和球丢失。DuckKing 已经有自由球、选定脚接触和实际球位移验证，因此主要工作是目标采样、奖励和 holdout 评估（[DuckKing simulation.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/simulation.py)）。之后再做拦截和运球，并将目标速度、脚接触时序和失球作为明确指标。

第四阶段是 beak pickup-and-carry。DuckKing 的 carry 场景已有实验性嘴部、上下接触垫、自由物体、起点和终点，并测量双侧夹持、抬升、相对滑移、搬运距离和稳定释放（[DuckKing grasp_model.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/grasp_model.py)，[DuckKing grasp.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/grasp.py)）。可以先用脚本策略提供示范，再训练残差 PPO；奖励应按接触、抬升、目标进度和释放状态分阶段发放，并对进度奖励限速，防止策略通过碰撞或拖动物体刷分。由于嘴部机构被项目标为 experimental，这一项适合做仿真能力展示，不应被解释为真实夹持能力验证。

第五阶段是两到四只鸭的编队、避碰和协作搬运。DuckKing 已有 duo、四鸭 stage、发现、碰撞、A* 绕行、推搡和滚翻恢复的真实接触场景（[DuckKing stage.py](https://github.com/zhoutingunl/DuckKing/blob/dba47d4/src/duckking/stage.py)）。第一步可用集中式 critic PPO 学相对位置/速度和目标点；再比较参数共享的分布式策略、中心化 PPO 和 MAPPO。奖励包括到达进度、编队误差、最小鸭间距离、碰撞次数和同步误差。多智能体安全控制已有 Safe Multi-Agent MuJoCo 等基准，但这些方法不能替代 MicroDuck 自身的碰撞和载荷验证（[Safe multi-agent RL](https://www.sciencedirect.com/science/article/pii/S0004370223000516)）。

## 真实感、稳定性和评估应作为贯穿式工程线

每个任务都应包含无随机化 baseline、分组随机化和完整随机化三组实验。优先随机化与硬件直接相关的量：脚/平台摩擦、质量与质心、关节零位、PD 增益或扭矩代理、控制延迟、IMU 噪声/偏置、接触柔顺性和转盘扰动。legged_gym 将摩擦、质量、推搡、PD、动作延迟、零位和质心作为独立配置项，提供了可借鉴的参数组织方式（[legged_gym config](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)）；但其数值范围来自其他机器人，不应直接复制。系统辨识研究也指出，手工设定过宽的随机化区间可能降低效率，应该先用状态预测误差缩小参数分布（[SPI-Active](https://proceedings.mlr.press/v305/sobanbabu25a.html)）。在没有 MicroDuck 硬件日志时，所有随机化范围都应标为项目假设。

当前最重要的代码基础设施不是新网络，而是统一环境接口和指标。建议把 `terrain`、`platform_profile`、`disturbance_schedule`、`payload`、`sensor_noise` 和 `actuator_delay` 作为 resettable 配置，通过 `info` 回传实际参数；提供统一的 episode recorder，至少记录生存时长、成功标志、直立余弦、峰值角速度、平均/最大滑移、接触率、动作变化、能量代理、任务误差、失败原因和安全接管次数。每项训练都应有 deterministic holdout JSON、遥测 CSV 和带事件标记的视频。上游任务开发同样要求 smoke test、NaN 检查、观测一致性、ONNX 导出和域随机化检查（[microduck_rl AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)）。

安全策略应在单独一层实现：对关节目标变化率、姿态、边缘距离和接触风险设硬限幅；越界时由恢复控制器接管，并报告 override 频率和恢复结果。安全强化学习工作把恢复策略和约束违例作为可测量结果，适合转化为 MicroDuck 的 simulator-only benchmark（[Safe RL for legged locomotion](https://research.google/pubs/safe-reinforcement-learning-for-legged-locomotion/)）。视觉控制、RMA 在线适应、真实系统辨识和 soft/deformable object 应放在后面，因为它们需要相机观测、更多并行环境或硬件数据；RMA 的成功案例是四足机器人，不能直接证明 MicroDuck 的硬件迁移效果（[RMA](https://arxiv.org/abs/2107.04034)）。

## 共同实施顺序和每项交付物

| 阶段 | 任务 | 现有资产复用 | 新增工作 | 难度 | 价值 |
|---|---|---|---|---|---|
| 1 | 推搡/转速突变恢复 benchmark | 当前转盘、姿态/接触/滑移观测、摔倒判定 | 扰动调度、恢复统计、holdout evaluator | 低—中 | 很高 |
| 2 | 平地速度/偏航跟踪 | 14 关节、HOME、command、velstand baseline | 速度命令、跟踪奖励、步态指标 | 低—中 | 高 |
| 3 | 斜坡、低摩擦、粗糙地形与轮滑 | groundcontact、rollers、roller policy | 地形生成、摩擦 curriculum、滑移指标 | 中 | 很高 |
| 4 | kick-to-target 与运球 | ball scene、脚球接触、球位移验证 | 目标采样、物体相对观测、稀疏奖励 | 中 | 高 |
| 5 | 站起/跌倒恢复与安全 shield | sitstand、standup、姿态测量 | 分层策略、动作限幅、恢复接管 | 中—高 | 很高 |
| 6 | 嘴部拾取、搬运、放置 | carry/grasp 的实验性嘴部和 verifier | 接触 curriculum、释放奖励、负载随机化 | 中—高 | 很高 |
| 7 | 多鸭编队、避碰、协作搬运 | duo/stage/table 的多鸭接触 | centralized critic/MAPPO、关系观测 | 高 | 很高 |
| 8 | 视觉、在线适应、sim-to-real | 当前渲染和状态记录 | 相机策略、系统辨识、RMA、硬件数据 | 很高 | 研究型 |

我们可以按阶段 1 开始一起实现。第一轮的最小交付应是 `disturbance_schedule`、恢复指标和 `evaluate_recovery.py`：在静止地面、倾斜转盘、非匀速转盘三类条件下，对 5 个扰动方向和多个幅值做固定种子评估；同时保留未训练 PPO、当前 PPO 和官方/脚本控制器对照。第二轮加入 grouped domain randomization，第三轮再做速度跟踪和坡道，避免把训练困难、指标不明的多个变量一次叠加。

## Conclusion

最适合共同完成的路线是“先恢复，再运动，再接触，再协作”：推搡恢复能最快验证策略是否真正理解姿态与接触；速度跟踪和坡道把它扩展成运动控制；踢球、抓取和搬运提供可见的任务成功；多鸭协作则把单体控制升级为关系决策。每一项都应与官方 ONNX 或脚本策略做固定条件对照，并在随机 holdout 上报告成功率、5 分位稳健性、能耗代理和失败视频。

当前不应把仿真结果写成实机能力。DuckKing_PPO 仍使用理想 MuJoCo position actuator 和自定义 59 维观测；上游 61 维 contract、BAM/backlash 执行器、真实传感器噪声和硬件限位属于后续仿真真实性工程。先把扰动恢复 benchmark 做扎实，项目就会从“能训练一个站立策略”升级为“能测量 MicroDuck 在不同动态条件下的鲁棒控制能力”。
