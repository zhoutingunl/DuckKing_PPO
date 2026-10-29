# Sim to real and benchmark directions

## What sim to real methods and benchmarks are practical for MicroDuck PPO?

### Takeaway
The next layer after the current tilted turntable task should focus on robustness and measurement: calibrated dynamics randomization, disturbances, latency/noise, and standardized held-out evaluation. These are implementable in the existing MuJoCo environment before any hardware work. Isaac Lab and legged_gym provide reusable design patterns, but moving this small robot to those GPU frameworks is optional; the current MuJoCo + SB3 stack can implement the same ideas.

### Cited Findings
- NVIDIA Isaac Lab explicitly lists domain randomization, actuator models, multi-frequency sensor simulation, data collection pipelines, and modular environments among its robot-learning capabilities; this supports adding actuator/sensor randomization and batch evaluation as first-class environment options. [Isaac Lab overview](https://docs.nvidia.com/isaaclab/isaac-lab) and [Isaac Lab research description](https://research.nvidia.com/publication/2025-09_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning)
- The upstream `legged_gym` configuration randomizes contact friction over `[0.5, 1.25]`, can randomize base mass, and applies periodic pushes by changing base planar velocity; its reward terms include tracking, vertical/roll-pitch angular velocity penalties, torque, acceleration, action-rate, collisions, and contact-force limits. [legged_gym base configuration](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py) and [legged_gym robot implementation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- A later legged-gym configuration exposes actuator-relevant randomization knobs—PD stiffness/damping, torque scale, motor zero offsets, and action delay—alongside friction, mass, center-of-mass, and push disturbances. [Example legged-gym configuration](https://github.com/engineai-robotics/engineai_legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- A systematic study of dynamics randomization for Laikago reports that direct sim-to-real transfer can sometimes work without randomization, warning that randomization should be measured and tuned rather than assumed to help every robot/task. [Dynamics Randomization Revisited](https://www.pair.toronto.edu/understanding-dr/)
- SPI-Active (CoRL 2025) argues that heuristic domain-randomization ranges can be overly conservative and introduces sampling-based system identification with active exploration to fit physical parameters from state-prediction errors in contact-rich legged systems. [PMLR paper](https://proceedings.mlr.press/v305/sobanbabu25a.html) and [official implementation](https://github.com/LeCAR-Lab/SPI-Active)
- Safe Reinforcement Learning for Legged Locomotion combines a learned policy with a safe recovery policy and reports simulated catwalk, two-leg balance, and pacing tasks; the design is directly relevant to MicroDuck because the current task is a two-leg balance controller with a safety-critical fall boundary. [Google Research publication](https://research.google/pubs/safe-reinforcement-learning-for-legged-locomotion/) and [Google Research summary](https://research.google/blog/learning-locomotion-skills-safely-in-the-real-world/?hl=en_GB&version=v1)
- The Safe Multi-Agent RL paper introduces Safe Multi-Agent MuJoCo, Safe MARobotsuite, and Safe MAIsaac Gym benchmarks for cooperative multi-robot safety constraints. A four-duck extension can use the same constrained multi-agent framing after single-duck robustness is stable. [Safe multi-agent reinforcement learning for multi-robot control](https://www.sciencedirect.com/science/article/pii/S0004370223000516)
- Stable-Baselines3's evaluation helper recommends a separate evaluation environment, deterministic prediction, and explicit accounting for wrappers that alter reward or episode length. [SB3 evaluation helper](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html) and [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### Inferences
- For MicroDuck, randomize parameters with a clear causal relation to hardware: foot/platform friction, body and link mass, center of mass, actuator gain/torque, joint zero offsets, control delay, IMU noise/bias, and solver/contact softness. Keep a no-randomization baseline so the effect of each group can be quantified.
- Use a two-stage protocol: first perform a low-cost offline system-identification sweep over plausible parameter distributions; then train PPO under a narrowed distribution centered on the fitted parameters. This follows SPI-Active's motivation while avoiding immediate dependence on real robot data.
- Move from fixed episode success to a benchmark matrix: nominal, low/high friction, platform tilt, spin acceleration, initial pose, impulse direction, sensor noise, and delay. Each policy should be tested on held-out random seeds and parameter combinations.
- The present PPO implementation can keep MuJoCo CPU and add vectorized environments later. Isaac Lab becomes valuable if thousands of parallel environments or camera observations become necessary, but it is not required for the next experiments.

### Gaps
- No public MicroDuck hardware logs, actuator identification data, or confirmed real-world friction/latency ranges were available in the local repository or the sources above. Any numeric randomization interval beyond existing simulation values must be treated as a tunable hypothesis.
- The cited legged-gym ranges are for other robots; they should not be copied directly to MicroDuck without sensitivity analysis.

## Which concrete PPO tasks should we implement next?

### Takeaway
The most useful sequence is: (1) disturbance-recovery benchmark, (2) uneven/low-friction terrain locomotion, (3) domain-randomized robust standing and walking, (4) safe policy shielding, (5) fall recovery, and then (6) multi-duck coordination. Ball kicking and carrying can follow as manipulation benchmarks, but they require more sparse-reward and object-contact design.

### Cited Findings
- A dynamic fall-recovery study applies external forces/torques, low foot friction, and randomized dynamics, and frames recovery as a separate skill rather than treating every fall as immediate episode termination. [Dynamic Fall Recovery Control for Legged Robots via RL](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048123/)
- Research on robust legged dynamics defines recovery rate under disturbances and reports a 91.0% gait-averaged recovery rate for one policy at disturbances up to twice body weight, demonstrating that recovery percentage and disturbance magnitude are useful benchmark axes. [Neural dynamics of robust legged robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063321/)
- A locomotion benchmark comparing MPC and RL in MuJoCo evaluates disturbance rejection, energy efficiency, and terrain adaptability, which are suitable dimensions for a MicroDuck benchmark report. [Benchmarking MPC and RL-based control for legged locomotion in MuJoCo](https://arxiv.org/abs/2501.16590)
- Safe RL for legged locomotion explicitly treats safety constraints and recovery activation as measurable outcomes instead of reporting reward alone. [Google Research safe RL publication](https://research.google/pubs/safe-reinforcement-learning-for-legged-locomotion/)
- Stable-Baselines3 recommends separate test environments and deterministic inference for meaningful policy comparisons. [SB3 evaluation helper](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)

### Inferences
- **Disturbance recovery (highest priority):** add scheduled impulses and short force/torque pulses to trunk or platform, with curriculum over magnitude and direction. Metrics: survival rate, recovery time, peak tilt, maximum impulse tolerated, foot-contact restoration, and energy/action variation. Existing `MicroDuckBalanceEnv` already has fall checks, IMU, contact, and turntable state, so this is a small environment extension.
- **Terrain/traction robustness:** swap the turntable top for patches with randomized friction, slopes, compliance, and small height steps. Train either stand-still or commanded velocity. Metrics: distance, velocity tracking error, slip ratio, foot stumble count, falls per minute, and cost of transport. MuJoCo supports tangential, torsional, and rolling friction in its general contact model, enabling this without changing engines. [MuJoCo computation docs](https://mujoco.readthedocs.io/en/latest/computation/)
- **Robust standing/walking under domain randomization:** add action delay, observation noise, motor gain/zero offsets, mass/COM and contact friction randomization. Compare policies trained with no randomization, grouped randomization, and full randomization. Report mean and 5th-percentile returns across held-out parameter sets; this avoids declaring success from one nominal run.
- **Safe PPO / shielded control:** keep the PPO residual policy but clamp joint target rate, torque proxy, trunk tilt, and contact-force/edge risk. A recovery controller can override the learned action when a safety condition is violated. Metrics: task return subject to zero hard-safety violations, override frequency, recovery success, and time spent near limits.
- **Fall recovery:** train a separate policy starting from kneeling, side-lying, or high-tilt states to return to a verified standing pose. Use hierarchical switching: balance policy for upright states, recovery policy for fallen states. Avoid claiming real hardware readiness until joint limits, self-collision, and actuator authority are checked.
- **Multi-duck coordination:** instantiate two to four MicroDucks and train MAPPO or centralized-critic PPO for formation, collision avoidance, synchronized walking, or cooperative pushing/carrying. Add pairwise distance and collision constraints. The Safe Multi-Agent MuJoCo literature provides the benchmark framing, while DuckKing already has a four-duck MuJoCo stage.
- **Manipulation tasks:** kick-to-target, carry-and-place, and beak grasping are natural sparse-reward tasks because DuckKing already contains ball and grasp scenes. Use curriculum from scripted placement to target selection, then learn residual corrections around the existing official kick/grasp policies. Metrics should include object displacement/placement tolerance and contact success, not only episode reward.

### Gaps
- No direct evidence establishes that PPO is the best algorithm for every proposed task. PPO is a practical baseline; SAC/TD3, MAPPO, or hierarchical RL should be treated as comparison algorithms after the environment and metrics are stable.
- Fall recovery may require additional self-collision-safe poses and actuator authority that the current 14-joint residual interface does not guarantee.

## How should the project measure progress and choose the next implementation?

### Takeaway
Progress should be reported as a reproducible benchmark table and videos, not a single training curve. The next implementation should be a disturbance-recovery benchmark layered on the existing turntable environment, followed by a separate robust-walking environment once recovery is measured.

### Cited Findings
- SB3's evaluation API supports deterministic multi-episode evaluation and warns that reward/episode-length wrappers affect reported results; this supports fixing wrapper configuration and seeds in benchmark scripts. [SB3 evaluation helper](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)
- The MuJoCo documentation describes contact modeling with tangential, torsional, and rolling friction cones, so contact/friction metrics must be recorded explicitly when comparing policies. [MuJoCo computation](https://mujoco.readthedocs.io/en/latest/computation/)
- Legged-gym exposes reward terms for tracking, angular velocity, torque, acceleration, action rate, collisions, and contact-force limits, providing a concrete template for decomposed MicroDuck telemetry. [legged_gym configuration](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- The MuJoCo MPC/RL benchmark identifies disturbance rejection, energy efficiency, and terrain adaptability as evaluation dimensions rather than using reward alone. [MuJoCo locomotion benchmark](https://arxiv.org/abs/2501.16590)
- Robust locomotion work reports recovery rate under controlled disturbance magnitude, which supports plotting a disturbance-response curve rather than choosing one arbitrary push level. [Neural dynamics of robust legged robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063321/)

### Inferences
- Define a fixed benchmark suite with at least: 100 held-out seeds per condition; nominal and randomized physics; tilt-only, spin-only, combined tilt/spin; impulse magnitudes; friction levels; sensor delay/noise; and initial tilt bins.
- Record per-episode: survival duration, binary success, upright cosine, peak body angular speed, mean/max foot slip, contact fraction, target tracking error, action rate, torque/energy proxy, fall cause, and safety overrides.
- Publish three aggregate views: mean with confidence interval, 5th-percentile robust score, and worst-case failure examples. Include an ablation table for reward terms and randomization groups.
- Keep videos synchronized with telemetry: platform angle/omega, body tilt, foot contacts, action norm, reward decomposition, and event markers for disturbances. This makes the visual learning process scientifically auditable.
- Recommended implementation order for this repository:
  1. Add deterministic disturbance schedules and benchmark/evaluation JSON.
  2. Add grouped domain randomization and held-out parameter tests.
  3. Add uneven/low-friction terrain and commanded walking.
  4. Add safety shield and fall-recovery policy.
  5. Add multi-duck centralized-critic coordination.
  6. Add sparse manipulation tasks (kick, carry, place).

### Gaps
- Confidence intervals, sample counts, and thresholds are recommendations for this project, not universal standards; the report should label them as project protocol decisions.
- Real-world acceptance criteria cannot be finalized without MicroDuck actuator limits, sensor rates, emergency-stop behavior, and measured hardware logs.
