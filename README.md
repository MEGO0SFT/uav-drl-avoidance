# UAV Autonomous Obstacle Avoidance in Unknown Environments via Deep Reinforcement Learning

**English** | [简体中文](README.zh-CN.md)

A research codebase for LiDAR-based navigation in unknown, static three-dimensional environments. The current implementation uses AirSim, a block-sparse probabilistic map and a curriculum-trained PPO policy, with an additional hybrid safety shield.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[Release catalogue](MODELS.md) · [Architecture](docs/ARCHITECTURE.md) · [Running experiments](docs/REPRODUCING.md) · [Roadmap](ROADMAP.md)

[Mathematical Foundations and Implementation](docs/MATHEMATICS.md): coordinates and observations, BSHM updates, rewards and curriculum, network dimensions, and shield optimisation.

## Status and public materials

The AirSim implementation is the Stage-1 baseline. The legacy reference is named origin_26-0364; main contains maintenance and documentation changes.

The repository retains source code, scene assets, configuration notes, historical summary tables and final figures. Retained results are historical reports, not results revalidated by this checkout.

Policy checkpoints and complete simulator packages are distributed separately. See [models](MODELS.md) and the [simulator guide](sim/airsim/ENVIRONMENT.md). The Isaac Lab, PX4 SITL and ROS 2 work described below is planned development, not an implemented replacement for AirSim.

## Task and system

The task is single-drone goal navigation through previously unmapped static obstacles. The policy receives a local representation derived from LiDAR and a task/kinematic vector. It has no preloaded global obstacle map or absolute-position observation; relative goal information and simulator ground-truth kinematics are available.

Ten drone instances are configured for parallel sampling. Inter-drone collisions and LiDAR occlusion are disabled by the [isolation patch](sim/airsim/ISOLATION_PATCH.md); this is not a multi-agent coordination experiment.

| Component | Current implementation |
|---|---|
| Simulation | AirSim 1.7.0 / Unreal Engine 4.27 scene; RPC to a data hub and UDP between hub and trainer |
| Mapping | BSHM: 64 m blocks, 0.25 m voxels, occupancy probabilities, temporal maintenance and bounded block storage |
| Observation | 18 × 36 heading-frame distance image plus an 8-dimensional task/kinematic vector |
| Policy | CNN/MLP dual-branch feature extractor and feed-forward PPO |
| Action | World-NED velocity increment, each axis in ±3.75 m/s; resulting speed capped at 7.5 m/s |
| Optional shield | Proximity repulsion, adaptive-horizon VO, local RRT* fallback, policy blending and distance-dependent speed limits |

    LiDAR → world-frame BSHM → heading-frame distance image
                                    + task/kinematic vector
                       → dual-branch PPO → velocity proposal
                       → optional shield → AirSim velocity command

The map provides temporal context; the final policy does not contain an LSTM or GRU. Observation heading follows horizontal velocity, while actions remain in world coordinates. The shield is a heuristic correction mechanism, not a demonstrated formal collision-avoidance guarantee.

### Scene

![Training scene](sim/airsim/figures/training_scene.png)

The documented training scene is approximately 250 × 250 × 50 m and contains roughly 70 obstacle actors. Its layout is stored in [FlyingExampleMap.umap](sim/airsim/project/Content/FlyingCPP/Maps/FlyingExampleMap.umap). The complete runnable simulator and AirSim plugin dependencies are not contained in the scene-only Git checkout.

## Historical results

The following values are retained summaries. Training, deterministic evaluation and generalisation use different protocols and should not be compared as one continuous performance series.

| Protocol | Reported value |
|---|---|
| Curriculum PPO, training rolling success metric | 0.810 |
| No-curriculum PPO, training rolling success metric | approximately 0.764 |
| SAC comparison, training rolling success metric | peak 0.699; final 0.373 |
| Unseen corridor experiment | reported success 68.2% |
| Nominal PPO training budget | approximately 6 million environment steps |

The current training script specifies 10 environments and 2560 rollout steps per environment. Its configuration has not been fully reconciled with the original run metadata; it should not be treated as a verified exact reconstruction of the historical run.

### Safety-shield extension

The summary below comes from [ablation_summary.json](benchmarks/ablation_summary.json). These shield experiments are repository extensions, separate from the journal article's stated scope.

| Configuration | Episodes | Success | Historical failure field* | Return | Successful-episode time |
|---|---:|---:|---:|---:|---:|
| Pure policy | 309 | 90.94% | 9.06% | 245.80 ± 78.36 | 22.31 ± 4.26 s |
| + proximity repulsion | 307 | 94.46% | 5.54% | 261.44 ± 81.76 | 24.38 ± 4.62 s |
| + adaptive VO | 305 | 93.11% | 6.89% | 258.29 ± 85.50 | 24.36 ± 4.47 s |
| + RRT* fallback | 311 | 95.82% | 4.18% | 263.84 ± 71.64 | 24.65 ± 5.27 s |
| Full system | 525 | 96.95% | 3.05% | 288.96 ± 68.27 | 24.70 ± 5.28 s |

*The historical JSON key is collision_rate, but the evaluator computes this field as 100 minus success rate. If timeouts occur, it is a failure rate rather than a collision-only rate. The experiments add modules cumulatively and do not independently identify each module's causal contribution.

![Training convergence](benchmarks/figures/ppo_training_convergence.png)

![Safety-shield ablation](RL_scripts/body-axis_v2/scripts/figures/ablation_combined.svg)

![Example trajectory](RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_shield_ep1.png)

These figures are retained historical outputs. The trajectory illustrates one episode and is not a population-level estimate. All existing final image, SVG and PDF panels remain in the repository.

## Code map

| Path | Purpose |
|---|---|
| [RL_scripts/body-axis_v2/scripts](RL_scripts/body-axis_v2/scripts/) | Main PPO training implementation, shield extension and evaluation tools |
| [methods/AirSim_ReinforceLearning/BSHM.py](methods/AirSim_ReinforceLearning/BSHM.py) | Block-sparse probabilistic mapping |
| [RL_scripts/README.md](RL_scripts/README.md) | Historical experiment directories and entry points |
| [sim/airsim](sim/airsim/) | Sensor settings, scene assets and isolation-patch documentation |
| [benchmarks](benchmarks/) | Retained historical summaries and final training figures |
| [paper](paper/) | Citation metadata and the article/repository scope distinction |
| [scripts](scripts/) | Repository checks and a local TensorBoard export utility |

## Running the AirSim baseline

The provided simulator packages target Windows x64. The training environment export targets Linux/WSL2 with Python 3.11.14, PyTorch 2.6.0+cu124, Stable-Baselines3 2.7.0 and airsim 1.7.0. The [environment file](environment.yml) is a platform-specific record, not a cross-platform installation guarantee.

    conda env create -f environment.yml
    conda activate airsim-gym-ubuntu2404

Before running, obtain the required simulator/checkpoints, install [settings.json](sim/airsim/settings.json), verify isolation, and configure RPC/UDP addresses and drone counts for your actual topology. Example entry points are:

    python RL_scripts/body-axis_v2/scripts/airsim_datamanager.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py

The last two commands are alternatives: curriculum and no-curriculum training. See the [experiment guide](docs/REPRODUCING.md) before evaluation: the default evaluation entry disables the shield, some auxiliary imports still use historical module names, and its printed success statistics use a rolling window. New runs are not expected to exactly reproduce the historical summaries.

## Development priorities and contributions

1. **Isaac Lab migration is the immediate priority.** Contributions should cover the training environment, PX4 SITL / ROS 2 communication, training-method iteration and sensor redesign, rather than only replacing the simulator API.
2. **Reward and network research follows the first priority.** The current architecture has limited performance. After the migration foundation is established, investigate the effect of architecture on small end-to-end policies, further develop DRL, and explore vision-language-action (VLA) and composed-agent approaches to broaden policy capabilities.
3. **Sim2Real.** Study control frequency versus actuator capability, latency, disturbances and generalisation through staged simulation and real-world validation.
4. **Other vehicle morphologies.** Structures beyond conventional UAV configurations are a long-term research direction, with no near-term implementation commitment.

Detailed milestones and contribution expectations are in [ROADMAP.md](ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md). None of these research directions is a claim of an existing capability or a guaranteed performance outcome.

## Upstream projects and documentation

| Component | Original project and documentation | Role in this repository |
|---|---|---|
| Microsoft AirSim | [Source repository](https://github.com/microsoft/AirSim) · [Official documentation](https://microsoft.github.io/AirSim/) | Simulation, sensors and velocity-control interfaces |
| Stable-Baselines3 | [Source repository](https://github.com/DLR-RM/stable-baselines3) · [Official documentation](https://stable-baselines3.readthedocs.io/) | PPO and training infrastructure |
| SB3 Contrib | [Source repository](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | Historical experiment dependency; the delivered policy is not RecurrentPPO |
| Gymnasium | [Source repository](https://github.com/Farama-Foundation/Gymnasium) · [Official documentation](https://gymnasium.farama.org/) | Environment interfaces and space definitions |
| Open3D | [Source repository](https://github.com/isl-org/Open3D) | Point-cloud processing and visualisation |
| Numba | [Project website](https://numba.pydata.org/) | Numerical acceleration |
| SciPy | [Project website](https://scipy.org/) | Numerical utilities and shield linear programming |

## Paper and attribution

石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕. 基于渐进课程学习的无人机自主避障研究. 系统仿真学报, 2026. [DOI: 10.16182/j.issn1004731x.joss.26-0364](https://doi.org/10.16182/j.issn1004731x.joss.26-0364).

See [publication metadata](paper/README.md) for scope and BibTeX. Stage 1 was developed at Sun Yat-sen University's School of Aeronautics and Astronautics. The code builds on [Microsoft AirSim](https://github.com/microsoft/AirSim), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [Open3D](https://github.com/isl-org/Open3D), [Numba](https://numba.pydata.org/) and [SciPy](https://scipy.org/); upstream components retain their own licences.

[MIT](LICENSE) © 2025–2026 Shifutian (石辅天).
