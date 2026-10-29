# PPO and MuJoCo task candidates

## What task families are technically established for a MicroDuck PPO roadmap?

### Takeaway
PPO is a practical baseline for continuous-control robotics because it supports minibatch policy updates and has been evaluated on simulated locomotion; the original PPO paper explicitly includes simulated robotic locomotion experiments. [PPO paper](https://arxiv.org/abs/1707.06347) A MicroDuck roadmap should therefore begin with proprioceptive control tasks that can reuse the existing 14-joint position-actuator interface, then add task-specific contacts, curriculum, and randomization.

### Cited Findings
- PPO alternates environment sampling with multiple minibatch epochs on a clipped surrogate objective; the authors report experiments including simulated robotic locomotion. [PPO paper](https://arxiv.org/abs/1707.06347)
- Gymnasium's custom-environment guidance requires explicit observation/action spaces and the `reset`/`step` API, making it a suitable interface for each independent MicroDuck task. [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
- MuJoCo models expose bodies, joints, actuators, contacts, sensors, and simulation stepping, which are the primitives needed to define proprioceptive observations and contact-based rewards. [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/overview.html)
- The current DuckKing repository already contains MuJoCo scenes and ONNX policies for walking, sit/stand, kicking, rolling, rollers, grasping, multiple ducks, table scenes, and stage contacts. [DuckKing repository](https://github.com/zhoutingunl/DuckKing)

### Inferences
- A single generic PPO environment should not be stretched across all behaviors. It is better to expose task-specific Gymnasium environments that share a common MicroDuck state extractor, action scaling, episode recorder, and evaluation schema.
- Existing official policies can serve as demonstrations, warm starts, or safety baselines while PPO learns residual corrections.

### Gaps
- I did not find a public MicroDuck-specific PPO benchmark with standardized success thresholds. Thresholds should be defined locally from MuJoCo telemetry and the intended hardware limits.

## Which near-term tasks reuse current assets with low implementation cost?

### Takeaway
The highest-return follow-on tasks are disturbance recovery, velocity tracking, sit-to-stand, and kick/ball control. They need new reset distributions and reward terms but can use the existing model, joint order, contacts, and ONNX controllers already present in DuckKing. [DuckKing repository](https://github.com/zhoutingunl/DuckKing)

### Cited Findings
- DuckKing's current simulation includes a `Driver` using 14 policy joints, HOME pose residual targets, IMU angular velocity, gravity direction, joint positions/velocities, previous action, and velocity commands; this is an immediately reusable PPO observation/action contract. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)
- The repository has real MuJoCo ball-foot contact checks and measured ball displacement for kick actions. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)
- The repository has measured sit/stand verification from floor/table contacts, uprightness, body height, and knee angles. [DuckKing posture source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/posture.py)
- DeepMimic demonstrates that a physics-based policy can combine motion imitation with a task objective and learn recoveries under perturbations. [DeepMimic paper](https://arxiv.org/abs/1804.02717)

### Inferences
- **Push recovery (difficulty: low to medium; value: very high):** randomize impulse direction, magnitude, and timing; reward uprightness, recovery time, foot contacts, and low post-recovery angular velocity; terminate on fall. Reuse the current balance environment and add an external force schedule. This is the best next experiment after the rotating-disk task because it isolates recovery robustness.
- **Velocity command tracking (low to medium; value: high):** sample planar velocity/yaw commands; reward body velocity tracking, heading tracking, uprightness, foot clearance, and energy; penalize falls and action jerk. Reuse DuckKing's existing `command` vector and official `velstand` policy as a baseline. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)
- **Sit-to-stand / stand-to-sit residual control (low; value: medium):** condition the policy on a discrete posture target and reward target height, uprightness, knee configuration, and support contacts. The existing measured posture verifier gives objective success labels. [DuckKing posture source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/posture.py)
- **Kick accuracy (medium; value: high for a visible demo):** include ball relative position/velocity and target direction in observations; reward selected-foot contact, target-line error, ball speed/distance, and post-kick standing. Existing contact and displacement measurements provide evaluation signals. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)

### Gaps
- Current public DuckKing documentation does not publish actuator torque limits or a validated reward weighting for these tasks. Those values require local MuJoCo inspection and sensitivity sweeps.

## Which tasks add locomotion or terrain research value?

### Takeaway
Terrain locomotion is a strong medium-term target, but it should follow stable standing and push recovery. Research on deep RL walking and challenging terrain supports proprioceptive feedback, terrain variation, and carefully designed locomotion rewards; these ideas transfer to a two-legged MicroDuck only after its narrow support polygon and low actuator authority are respected. [Learning to Walk via Deep RL](https://arxiv.org/abs/1812.11103) [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)

### Cited Findings
- Deep RL locomotion work identifies sample efficiency and hyperparameter sensitivity as major practical difficulties when transferring simulated policies to robots. [Learning to Walk via Deep RL](https://arxiv.org/abs/1812.11103)
- Challenging-terrain locomotion research uses proprioceptive feedback and terrain generation to obtain robust behavior and reports zero-shot simulation-to-natural-environment generalization for a quadruped. [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)
- Isaac Gym's platform couples GPU physics and policy training and reports orders-of-magnitude speedups for large-scale robot-learning workloads, which is relevant if many terrain environments become the bottleneck. [Isaac Gym paper](https://arxiv.org/abs/2108.10470)

### Inferences
- **Flat-ground gait learning (medium; value: high):** first train speed/yaw tracking on a fixed floor, with phase or previous-action features and a conservative action residual. Evaluate commanded velocity error, falls per meter, energy per meter, and foot slip.
- **Stepping stones / uneven heightfields (medium to high; value: very high):** randomize heightfields, friction, and gaps; add foot clearance and contact timing; use privileged terrain information only for a teacher, then distill to proprioceptive observations if sim-to-real is a goal.
- **Stairs and ramps (high; value: high):** curriculum from low slopes to stairs; reward progress, torso uprightness, foot placement, and no collisions. This is a good benchmark for recovery and gait adaptation but needs careful collision geometry and many parallel environments.
- CPU MuJoCo is adequate for one or a few visible MicroDuck episodes; high-throughput terrain sweeps may justify a vectorized/GPU simulator later rather than prematurely changing the current renderer.

### Gaps
- The cited challenging-terrain result is quadrupedal, not bipedal. It supports the methodology (proprioception, terrain variation, curriculum) but does not establish that the same terrain difficulty is achievable by MicroDuck.

## Which dynamic and manipulation tasks can use demonstrations?

### Takeaway
Motion imitation is the most direct way to turn DuckKing's authored skills and official ONNX policies into PPO tasks. DeepMimic specifically combines reference motion with task objectives and perturbation recovery, making it suitable for rolling, dancing, sit/stand transitions, and kick sequences. [DeepMimic paper](https://arxiv.org/abs/1804.02717)

### Cited Findings
- DeepMimic learns physics-based controllers from example motion clips while adding goal objectives and recoveries; it covers keyframed motions, flips/spins, and retargeted motions. [DeepMimic paper](https://arxiv.org/abs/1804.02717)
- DuckKing already exposes authored policies for `roulade`, `ground_pick`, `roller`, `roller_crouch`, and `sitstand`, plus a grasp scene. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)
- DuckKing's stage/table modules verify actual contact and posture outcomes rather than treating elapsed time as success. [DuckKing stage source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/stage.py)

### Inferences
- **Skill imitation with recovery (medium; value: high):** record joint targets and body poses from official ONNX policies; train PPO with pose/velocity imitation plus a task term and randomized pushes. Start with a single skill such as crouch-to-stand or forward roll recovery.
- **Beak grasp and carry (medium to high; value: very high demo value):** include object pose/velocity and beak geometry/contact state; reward stable grasp, object transport, release placement, and standing. Use the existing grasp scene and contact verification as the environment scaffold.
- **Roller control (medium; value: medium):** keep passive wheels and learn steering/speed/crouch transitions; reward forward progress, heading, wheel contact, and stability. Existing roller model and `roller`/`roller_crouch` policies reduce model work. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/skills_model.py)
- **Dance/choreography imitation (medium; value: high for product/demo):** use reference trajectories and beat phase; reward pose/style plus contact safety. This is better as a constrained imitation experiment than a free-form task.

### Gaps
- No public training dataset or explicit reference trajectory format is published for the official MicroDuck ONNX policies; extracting trajectories from MuJoCo rollouts is a practical local solution.

## What multi-agent and adaptation tasks are worth building?

### Takeaway
Multi-duck coordination and online adaptation are valuable after single-duck control is stable. Relational policies and adaptation modules have established research precedents, but they require stronger environment abstractions and more careful evaluation than a single-agent PPO baseline. [Relational inductive biases paper](https://arxiv.org/abs/1806.01261) [RMA paper](https://arxiv.org/abs/2107.04034)

### Cited Findings
- Relational inductive-bias work argues that structured representations and computations help combinatorial generalization, a rationale for entity/duck-centric policies instead of flat concatenation of all agents. [Relational inductive biases](https://arxiv.org/abs/1806.01261)
- RMA separates a base policy from an adaptation module and trains in simulation to adapt to unseen terrain, payload, wear, and other conditions. [RMA paper](https://arxiv.org/abs/2107.04034)
- DuckKing already has duo, four-duck stage, chorus, collision, discovery, and table scenes with measured positions/contact outcomes. [DuckKing stage source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/stage.py)

### Inferences
- **Formation and collision avoidance (high; value: high):** centralized PPO can use relative positions/velocities and goals; decentralized policies should use shared parameters plus a permutation-invariant entity encoder. Reward progress, separation, formation error, and no collisions.
- **Cooperative carrying/transport (high; value: very high):** two ducks coordinate around an object; reward object progress and orientation while penalizing asymmetric loads and collisions. Existing four-duck contact utilities can support evaluation, but the object and grasp geometry need extension.
- **Online friction/payload adaptation (high; value: very high for sim-to-real):** train an encoder from short history of proprioception and platform/object response; condition the policy on its latent estimate. Start with friction, mass, actuator delay, and turntable profile randomization. RMA gives a clear architecture reference. [RMA paper](https://arxiv.org/abs/2107.04034)
- **Partial-observability/recurrent PPO (medium; value: high):** hide platform angular velocity or object mass and provide history/GRU state; evaluate performance under unseen disturbance profiles. This tests whether the duck can infer hidden dynamics rather than rely on privileged simulator state.

### Gaps
- RMA's reported hardware transfer is for a quadruped A1, so the adaptation architecture is a precedent, not evidence of MicroDuck transfer performance. [RMA paper](https://arxiv.org/abs/2107.04034)

## What should we implement together next?

### Takeaway
A staged sequence maximizes learning value while preserving visible demos: (1) push recovery and evaluation suite, (2) flat-ground velocity tracking, (3) kick accuracy, (4) terrain curriculum, (5) beak grasp/carry, (6) online adaptation, and only then multi-duck cooperation. The first two can share most of the current environment code; the latter tasks need new entity/object interfaces.

### Cited Findings
- PPO is suitable as a common baseline for continuous-control task variants. [PPO paper](https://arxiv.org/abs/1707.06347)
- Motion imitation plus task objectives is an established way to train dynamic skills with recoveries. [DeepMimic paper](https://arxiv.org/abs/1804.02717)
- Domain randomization is a documented simulation-to-real strategy; the original paper trains a detector entirely in simulation and transfers it to real imagery by exposing the model to broad randomized variation. [Domain Randomization paper](https://arxiv.org/abs/1703.06907)
- RMA provides a concrete route from randomization to rapid online adaptation. [RMA paper](https://arxiv.org/abs/2107.04034)

### Inferences
- **Recommended next implementation: push-recovery benchmark.** It needs only an external impulse scheduler, a disturbance seed in `info`, recovery-time metrics, and evaluation scripts. It directly complements the current rotating-platform task and produces a scientifically interpretable robustness curve.
- **Second implementation: velocity tracking with a shared residual controller.** It reuses the existing command vector and turns the project from “balance under one perturbation” into a general locomotion controller.
- **Third implementation: kick-to-target.** It gives an immediate, visual, contact-grounded task and exercises object-relative observations, asymmetric leg control, and post-action recovery.
- Use one common metric schema across tasks: episode return, survival/time-to-fall, uprightness, foot contact ratio, foot slip, action jerk, energy proxy, task success, and seed confidence intervals. This follows the current DuckKing emphasis on measured contacts and posture verification. [DuckKing simulation source](https://github.com/zhoutingunl/DuckKing/blob/main/src/duckking/simulation.py)
- Keep real-hardware transfer out of the first implementation loop. Domain randomization and adaptation are valuable, but they should be evaluated in held-out simulator conditions before any hardware claim. [Domain Randomization paper](https://arxiv.org/abs/1703.06907)

### Gaps
- Exact implementation time and PPO sample budgets depend on local CPU/GPU speed, official asset availability, and whether training uses the toy fallback or the full MicroDuck MJCF. No public benchmark provides a reliable MicroDuck-specific estimate.
