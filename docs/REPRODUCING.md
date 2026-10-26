# Reproducing the Stage-1 results

**English** | [简体中文](REPRODUCING.zh-CN.md)

[← Back to README](../README.md) · [Architecture](ARCHITECTURE.md) · [Simulator environment](../sim/airsim/ENVIRONMENT.md)

This guide rebuilds every number in the paper from the code and the committed logs.

## 0. Verified platform

| Item | Value |
|---|---|
| OS | **WSL2 · Ubuntu 24.04** — simulator, data-exchange hub and trainer all in the *same* WSL2 instance |
| GPU | NVIDIA **RTX 4070 SUPER** (passthrough into WSL2) |
| CUDA | driver **12.0**; PyTorch wheels `2.6.0+cu124` |
| Python | 3.11.14 (miniforge / conda) |
| Environment | [`../environment.yml`](../environment.yml) — the exact export, authoritative |
| Pins that matter | `airsim 1.7.0`, `stable-baselines3 2.7.0`, `sb3-contrib 2.7.1`, `numpy 2.0.1`, `numba 0.62.1`, `open3d 0.19.0`, `gymnasium 1.2.2` |

```bash
conda env create -f environment.yml
conda activate airsim-gym-ubuntu2404
nvidia-smi        # should list the GPU inside WSL2
```

`requirements-legacy.txt` is a pip-only fallback; note that `airsim` must be **1.7.0** (the 1.8
client does not match the simulator build) and `numpy` must be **2.x** (numba 0.62 breaks on 1.x).

---

## 1. Topology

Stage 1 uses three roles. In the published setup all three ran inside one WSL2 instance; they may
also be split across a LAN.


Three processes, three links:

```
AirSim (UE4)  <--TCP RPC 41451-->  airsim_datamanager  <--UDP-->  trainer
                                    one process,          SubprocVecEnv
                                    all drones, 20 Hz     N parallel envs
```

| Link | Direction | Transport | Endpoint |
|---|---|---|---|
| simulation | `airsim_datamanager` → AirSim | TCP RPC | `<airsim_ip>:41451` |
| state | `airsim_datamanager` → trainer | UDP | the hub sends from `5812`; each trainer binds `2000 + drone_index` |
| commands | trainer → `airsim_datamanager` | UDP | the trainer sends to `5813`; the hub listens there |

Only `airsim_datamanager` talks to the simulator — the trainer sees the world purely through the
UDP state stream, which is what makes the two-machine split in §1 work at all.

| Setting | Where | Value in this repo | Meaning |
|---|---|---|---|
| `airsim_ip` | `airsim_datamanager.py:380` | `172.21.224.1` | host running the **AirSim** executable (RPC port `41451`) |
| `trainmanager_ip` | `airsim_datamanager.py:380` | `172.21.236.121` | host running the trainer; the NIC that receives the UDP state stream |
| `datamanager_ip` / `trainmanager_ip` | `make_env()` in `PPO_SB3_AirSim*.py`, `train_sac.py`, `train_ddpg.py`, `PPO_SB3_AirSim_model.py` | `172.21.236.121` | host running `airsim_datamanager.py` (both fields, same machine) |
| `datamanager_port` | `AirSimDroneEnv.__init__` (default `5812`) | hub listens on `5812` (state up) / `5813` (commands down) |
| trainer port | `2000 + uav_id` | one per drone, so `UAV01`→2000, `UAV02`→2001, … |

> All three addresses are hard-coded in this repository and must be replaced with your own,
> otherwise the trainer blocks forever waiting for observations. The original experiments used
> **two machines**: AirSim at `172.21.224.1`, and the hub plus trainer at `172.21.236.121`.
> A single-machine setup is fine — then all three fields take that one address.
>
> **Checklist for the IP configuration**
>
> 1. Inside WSL2, `ip -4 addr show eth0`; on Windows, `ipconfig`. Use an address that *every*
>    party can reach — do not fall back to `127.0.0.1`, because the UDP sockets bind to the given
>    interface and loopback binding breaks cross-process traffic.
> 2. Drone count must agree in three places: `range(10)` in `make_env()`, `uav_number=10` in
>    `airsim_datamanager.py`, and the number of drones actually launched — **10** in this project.
>    (`airsim_datamanager_easy.py` still carries `uav_number=8` from an earlier run.)
> 3. Per-drone control ports `2000 + index` must not collide with anything else on the trainer.
> 4. WSL2 is behind a NAT adapter by default; reaching it from the Windows host requires a port
>    proxy (`netsh interface portproxy add v4tov4 ...`). If the simulator also lives in the same
>    WSL2 instance, make all three parties bind the *same* interface — otherwise you get one-way
>    traffic (receives but never sends).
> 5. Windows Defender blocks newly listened UDP ports; allow them on first run.

---

## 2. Simulator setup

1. **Scene.** 250 × 250 × 50 m field, ~70 randomly placed primitives (boxes,
   cylinders, cones, spheres, pyramids), ~12.5 % spatial occupancy; start region
   in the centre, goal region at the perimeter.
2. **Simulator config.** Install [`../sim/airsim/settings.json`](../sim/airsim/settings.json) as
   AirSim's `settings.json` (`~/Documents/AirSim/settings.json`, or
   `%USERPROFILE%\Documents\AirSim\settings.json` on Windows). It declares **10** `SimpleFlight`
   drones `UAV01…UAV10` and the LiDAR actually used: **32 channels, 6 Hz, 8000 pts/s, 80 m range,
   ±30° vertical, 360° horizontal**, `DataFrame: SensorLocalFrame`.
   > `methods/Sensors1.json` describes a 16-channel / 25 m / 5 Hz sensor — that is a **stale
   > template** and does not match the published results. Use `sim/airsim/settings.json`.
   Only the LiDAR is consumed by the policy; the `CameraDefaults` entries were evaluated and
   dropped.
3. **Multi-drone isolation patch (required, already applied).** All ten vehicles spawn on the same
   point (`X=0, Y=0, Z=-0.5`), and stock AirSim lets drones collide with and occlude one another, which
   pollutes a *single-agent* observation with other agents' geometry. The patch lives in a single
   blueprint asset:

   ```
   Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
   ```

   and makes two changes:
   - disable the collision channel between drone meshes (keeps collision with environment geometry);
   - make the LiDAR ray-cast skip `Pawn` objects so lasers pass through drones.

   It is **already applied in both release archives**, and the same bytes are committed at
   [`../sim/airsim/project/Content/BP_FlyingPawn.uasset`](../sim/airsim/project/Content/BP_FlyingPawn.uasset)
   (sha256 `0f4f91c1…`). Keep that file as-is; swapping in the stock AirSim pawn makes the
   observation distribution shift and PPO will not converge.
   Per-component settings and the reason each cell matters:
   [`../sim/airsim/ISOLATION_PATCH.md`](../sim/airsim/ISOLATION_PATCH.md).

---

## 3. Launch order

```bash
# (1) simulator machine
#     install sim/airsim/settings.json as AirSim's settings.json, apply the UE blueprint
#     patches from §2, then start the scene

# (2) hub machine — ONE process serves all N drones
cd RL_scripts/body-axis_v2/scripts
python airsim_datamanager.py          # set uav_number to match your fleet

# (3) trainer machine — pick the group (see §4)
python PPO_SB3_AirSim_CL.py           # curriculum group
# or
python PPO_SB3_AirSim.py              # control group
```

Training writes TensorBoard events to `logs/` and checkpoints to
`models/<timestamp>/`. Both are git-ignored (checkpoints) or committed (logs).

---

## 4. The two paper groups — a one-line difference

`PPO_SB3_AirSim.py` and `PPO_SB3_AirSim_CL.py` are byte-identical except for two
lines, so that the ablation isolates **only** the progressive reward-shaping
curriculum:

| | control (`PPO_SB3_AirSim.py`) | curriculum (`PPO_SB3_AirSim_CL.py`) |
|---|---|---|
| `self.training_progress` initial value | `1.0` | `0.0` |
| `CallbackList` | without `TrainProgresserCal` | **with** `TrainProgresserCal` |

Both are trained with `model.learn(total_timesteps=6_000_000)` and a schedule
horizon of `total_timesteps = 1_024_000 × 5 = 5.12 M`, so the curriculum and the
entropy/clip schedules reach their terminal values at 5.12 M and are then held
constant — that is the "curriculum ends at 5 M, training totals 6 M" protocol in
the paper.

> **Batch-size note.** `n_steps=2560` is *per environment* in Stable-Baselines3, so
> one update consumes `n_envs × n_steps = 10 × 2560 = 25 600` transitions, and the
> 5.12 M schedule horizon is exactly 200 such rollouts.

Expected outcome (from the committed logs):

| Run | final `train/success_rate` | final `rollout/ep_rew_mean` |
|---|---:|---:|
| curriculum | **0.810** | 199.70 |
| control | 0.764 | — |

---

## 5. Evaluation

```bash
cd RL_scripts/body-axis_v2/scripts
python PPO_SB3_AirSim_model.py        # loads ../results/SFT_avoid_modelv3.zip
```

This runs `evaluate_policy(deterministic=True)` over the Monte-Carlo episode
count and prints success rate, mean/std of successful-episode duration, then dumps
a per-episode failure report. Set `safe_rule=False` in `make_env()` to measure the
**bare policy**, `safe_rule=True` for the **full system**.

---

## 6. Reproducing the ablation table

```bash
python ablation_experiment.py         # 5 shield configurations × ~300 episodes each
```

Results land in `scripts/ablation_results/*.json`; the published summary is
archived at [`benchmarks/ablation_summary.json`](../benchmarks/ablation_summary.json).

| Configuration | Episodes | Success |
|---|---:|---:|
| Pure DRL | 309 | 90.94 % |
| + proximity repulsion | 307 | 94.46 % |
| + adaptive VO-τ | 305 | 93.11 % |
| + RRT\*-Safe | 311 | 95.82 % |
| Full system | 525 | 96.95 % |

`ablation_experiment.py` toggles the shield layers by subclassing
`AirSimDroneEnv` and overriding `safe_ruler`; the distribution-shift guard and
the target-proximity gate stay on in every group.

---

## 7. Reproducing the figures without re-training

The training curves are already committed as decimated CSV:

```bash
python scripts/export_tb_curves.py RL_scripts/body-axis_v2/logs --out benchmarks/tb_curves
```

Each CSV carries `step`, `train/success_rate`, `rollout/ep_rew_mean`,
`rollout/ep_len_mean`, `train/approx_kl`, `train/value_loss`, `train/std`,
`train/ent_coef` — enough to redraw every training figure in the paper. The raw
`events.out.tfevents.*` files are committed too, if you prefer TensorBoard:

```bash
tensorboard --logdir RL_scripts/body-axis_v2/logs
```

Trajectory figures come from `trajectory_visualizer.py` /
`shield_trajectory_visualizer.py`, which read the committed `.npz` files under
`scripts/figures/`; the Monte-Carlo per-step trajectories are in
`body-axis_v2_plot/scripts/trajectories_mc.csv`.

---

## 8. Things that will bite you

- **Effective control rate is ≈5.5 Hz, not 20 Hz.** `step()` sleeps to a 0.05 s
  floor, but the loop is gated by the 5 Hz LiDAR rotation and the UDP round trip.
  A successful episode is ~117–125 steps ≈ 22–24 s of wall clock.
- **The observation frame and the action frame differ on purpose.** The 18×36
  range image is rotated into the *velocity-heading* frame, while Δv is added to
  the *world-frame* velocity vector. Putting the action in the body frame made the
  drone barely move during early exploration, so this asymmetry was chosen
  deliberately.
- **`Expect_yaw_Cal()` is vestigial.** The yaw command actually sent is a
  per-episode random constant; the rate profile in that function is never called.
- **`EvalCallback` is constructed but never registered**, so no `best_model/` is
  produced. Use `CheckpointCallback` output or the logs instead.
- **The `RecurrentPPO_*` filenames are historical.** In `body-axis`,
  `world-axis-v2` and `airsim-only-with-lidar` they still carry the old name; the
  policy is a feed-forward PPO everywhere. Only `body-axis_v2` was renamed to
  `PPO_SB3_AirSim*.py`.
