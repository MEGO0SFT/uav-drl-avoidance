# Running and Evaluating the AirSim Baseline

**English** | [简体中文](REPRODUCING.zh-CN.md)

[← README](../README.md) · [Architecture](ARCHITECTURE.md) · [Simulator](../sim/airsim/ENVIRONMENT.md)

## Scope of reproducibility

This version provides source code, configuration notes, historical summary tables and final figures. The checkout cannot reconstruct all original training curves or per-episode statistics.

The commands below are starting points for a new experiment, not a guarantee of historical numerical reproduction. External simulator/checkpoint artefacts and environment-specific configuration are required. The existing CI checks static repository properties rather than training or flight behavior.

## 1. Runtime prerequisites

[environment.yml](../environment.yml) records a Linux/WSL2 environment: Python 3.11.14, torch 2.6.0+cu124, Stable-Baselines3 2.7.0, sb3-contrib 2.7.1, airsim 1.7.0, numpy 2.0.1 and Open3D 0.19.0. The release simulator packages target Windows x64 / Unreal Engine 4.27. Configure the deployment topology explicitly rather than assuming all roles run in one OS instance.

    conda env create -f environment.yml
    conda activate airsim-gym-ubuntu2404

[requirements-legacy.txt](../requirements-legacy.txt) is a pip dependency reference, not a substitute for matching the simulator, Python, PyTorch/CUDA and platform. Version compatibility should be tested in the target environment.

## 2. Network topology

    AirSim ← TCP RPC 41451 → data hub ← UDP → training environments

| Socket or field | Purpose |
|---|---|
| Hub airsim_ip | Address reachable from the hub for AirSim RPC |
| Hub trainmanager_ip | Destination address of the trainer's UDP state sockets |
| Environment datamanager_ip | Hub address used for commands |
| Environment trainmanager_ip | Local interface used to bind the trainer's state socket |
| Hub UDP 5812 | Bound source socket for outgoing state messages |
| Hub UDP 5813 | Receives commands from the training environments |
| Trainer UDP 2000 + index | One state receiver per drone, zero-based index |

Current entry points contain hard-coded addresses. The two trainer-side fields do not in general identify the same host: one is a remote destination and one is a local bind address. The hub's get_real_ip() also chooses its local interface, so inspect that selection for the actual routing environment.

Loopback is valid only when both communicating endpoints share the relevant network namespace. Across Windows/WSL or multiple hosts, use reachable addresses and verify bidirectional UDP and RPC connectivity. Routing and firewall requirements depend on the setup; a TCP port proxy is not a general solution for UDP forwarding.

Every worker's vehicle name must exist in the simulator and be served by the hub. Training or evaluation may use a subset of those vehicles. The current full trainer and hub request ten drones; the default evaluation uses one, while the easy hub retains a historical count of eight.

## 3. Simulator and sensors

Obtain a runnable package as described in [ENVIRONMENT.md](../sim/airsim/ENVIRONMENT.md). Install [settings.json](../sim/airsim/settings.json) in the simulator's AirSim settings directory and verify the [isolation patch](../sim/airsim/ISOLATION_PATCH.md) at runtime.

The supplied settings define UAV01–UAV10 and a 32-channel LiDAR at 6 rotations/s, 8000 points/s, 80 m range, vertical ±30° and horizontal 360°. The policy consumes LiDAR-derived geometry plus task/kinematic state. The older methods/Sensors1.json is not the current sensor specification.

Inter-drone isolation is part of the independent sampling protocol. Preserve environment collision responses while removing inter-drone collisions and LiDAR occlusion.

## 4. Training

After configuring networking and starting the simulator:

    python RL_scripts/body-axis_v2/scripts/airsim_datamanager.py

Then select one trainer:

    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py

The first starts curriculum progress at zero and registers TrainProgresserCal. The second fixes progress at one. The behavior difference is reward scheduling, not scene difficulty.

Current code specifies ten environments × 2560 steps per rollout, or 25,600 transitions. Curriculum and entropy scheduling use a 5.12-million-step horizon; learning rate and clipping use the PPO learn progress over approximately 6 million steps.

Before process-data removal, the historical PPO logs showed a 2048-step reporting cadence, which does not match the current ten-environment count. The original run configuration and step semantics have not been fully reconciled. Do not present the current parameter list as a verified exact historical configuration.

The reported 0.810 / approximately 0.764 training success metrics average per-environment rolling windows of up to 100 episodes and include training-time exploration. They are not fixed-test-set success rates.

New events and checkpoints are generated locally. Their paths are ignored by Git and should remain outside published source history.

## 5. Evaluation and ablation

Obtain the algorithm-appropriate checkpoint from the [model catalogue](../MODELS.md). The evaluation entry expects the PPO archive under RL_scripts/body-axis_v2/results/SFT_avoid_modelv3.zip.

    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py

Read the configuration first: make_env() currently sets safe_rule=False. Enable it explicitly for a shield experiment and keep the policy/scenario protocol comparable.

Known implementation limits:

- The entry evaluates 525 episodes in one environment but stores only the most recent 300 success flags; its printed overall success rate is a rolling statistic. Successful-episode times have a separate 300-entry window.
- The retained 509/525 result is a historical summary, not a value independently reproduced by that current printing path.
- ablation_experiment.py and some plotting tools still import the historical RecurrentPPO_SB3_AirSim_model name. Adapt those imports to the delivered module before running.
- The historical collision_rate field is computed as 100 minus success rate; it includes timeouts when present.
- Success-rate printing, reward collection and failure diagnosis may use different windows. Define one episode-level accounting protocol for new experiments.

These observations do not establish that historical summary values are wrong; they identify gaps in reproducing them with the present code.

## 6. Figures and local exports

[benchmarks](../benchmarks/README.md) contains retained summaries and final figures. All existing final images, SVGs and PDFs are preserved. Some figure-generation tools need new local recordings.

scripts/export_tb_curves.py remains a local utility for event files you generate. Keep generated exports in an ignored local directory.

## 7. What new experiments should record

Record simulator/build versions, policy/checkpoint identity, configuration, seeds, scenario sampling, decision/control timing, coordinate conventions and separate episode outcomes. Check transition timing: the asynchronous environments can return an observation collected before the supplied action is sent.

Measure effective loop frequency end to end. Neither the hub's sleep calls nor the 6 Hz LiDAR setting alone determine the policy rate. The old approximate 5.5 Hz observation is a historical description, not a result verified for a new deployment.

The currently instantiated EvalCallback is not included in the trainer's callback list. Do not assume a best_model output is generated; inspect actual checkpoint behavior.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for public-data and validation expectations.
