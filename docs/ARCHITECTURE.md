# System Architecture

**English** | [简体中文](ARCHITECTURE.zh-CN.md)

[← README](../README.md) · [Experiment guide](REPRODUCING.md)

See [Mathematical Foundations and Implementation](MATHEMATICS.md) for coordinate transforms, occupancy updates, complete reward expressions, network dimensions and shield optimisation formulas with parameter explanations.

This document describes the delivered AirSim implementation. It separates code behavior from design intent; it is not a proof of safety or an exact reconstruction of every historical run. The journal article and later repository extensions have different [scope](../paper/README.md).

## 1. Task and interfaces

A quadrotor navigates through unknown static obstacles toward a supplied goal. The formulation is partially observable: a local LiDAR-derived map provides temporal context, but no recurrent neural state is used. Kinematics come from AirSim ground truth; noisy onboard state estimation and real flight are outside the validated scope of this baseline.

The data hub is the only process that calls AirSim RPC. Training environments receive state over UDP and send velocity commands back to the hub. Multiple drone instances sample independent single-drone tasks with inter-drone interactions disabled.

## 2. Observation: 656 dimensions

[vehicle_manager.py](../RL_scripts/body-axis_v2/scripts/vehicle_manager.py) constructs a flattened 18 × 36 distance image and the following task vector:

| Indices | Quantity |
|---|---|
| 0:3 | Unit relative goal direction in world NED coordinates |
| 3 | clip(1 − goal_distance / 150, −1, 1) |
| 4:7 | Unit world-frame velocity; zero when speed is zero |
| 7 | Speed divided by 7.5 m/s |

The map query uses an axis-aligned box extending 2 × stat_range from the vehicle on each axis: ±40 m with stat_range = 20 m. Occupied voxel centres are binned by elevation and azimuth at 10° resolution. Per-bin nearest distance is clipped to 20 m and normalised.

Azimuth is rotated by the horizontal velocity heading, rounded to a 10° increment in the projection. This is a velocity-heading frame, despite historical body-frame names. A value of 1 means no retained occupied point in that direction or a saturated distance; it does not establish observed free space.

The distance image uses the heading frame while goal/velocity vectors and actions use world coordinates. That asymmetry is an implementation choice, not a coordinate-invariance guarantee.

## 3. Action and execution

The policy proposes a three-dimensional velocity increment:

    a ∈ [−3.75, 3.75]³ m/s
    v_proposed = v_current + a
    if ||v_proposed|| > 7.5:
        v_proposed *= 7.5 / ||v_proposed||

The optional shield modifies this velocity before the hub calls AirSim's velocity-control API. This is not motor, thrust or attitude control. The increment is applied per decision step and should not be interpreted as a frequency-independent acceleration. The yaw command is a per-episode sampled setting; the legacy Expect_yaw_Cal function is not the active command source.

Training step() reads an observation and computes reward before sending the supplied action, then returns that observation. The evaluation version changes some ordering but also returns a pre-command observation. These asynchronous interfaces require timestamp/transition alignment to be checked before a platform port.

## 4. Reward and curriculum

See [PPO_SB3_AirSim_CL.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py). Let p be training progress clipped to [0,1]. The curriculum group uses p = min(step / 5.12 million, 1); the control fixes p = 1.

    r_dense = r_obstacle + c(p)·r_smooth
              + (1 − k_avoid)·r_forward + r_distance + r_survival
    c(p) = 0                          when p ≤ 0.25
           (p − 0.25) / 0.75         otherwise
    r_survival = −0.1 − 0.1p

### Obstacle and forward terms

The obstacle function uses normalised warning distance D = 0.2. Cells within 2D enter its danger mask. Their distance contribution uses f(d) = 1/e − exp(1 − d/D), weighted by directional sector averages:

- Horizontal weights: 1.75, 1.50, 1.25, 1.00.
- Elevation weights: 1.25, 1.00, 0.90.
- A minimum-distance trend term penalises approach more strongly than it rewards retreat.
- The combined danger reward is clipped to [−4,0.5]; when no cell is in the mask, a separate positive-distance expression is used.

Angles are inferred from grid indices in this reward function. Binning and clipping prevent interpreting it as a globally smooth geometric objective.

    k_avoid = exp(−3·minimum_distance_ratio) · clip(exp(W_dir − 1.5), 0, 1)
    r_forward = clip(dot(goal_direction, velocity_direction)·speed/150·50,
                     −1−p, 1+p)

The gating is intended to reduce forward incentive near hazards. It does not prove the absence of reward exploitation or undesirable local optima. The distance term rewards a new best normalised distance improvement exceeding 0.01 and penalises retreat under a low-urgency condition.

### Smoothness and terminal terms

The smoothness function uses up to ten velocity samples, with nominal dt = 0.1 s. It penalises excessive scalar-speed acceleration, jerk and changes in velocity direction through a product of exponentials, clipped to [−0.5,0.5]. Its time interval is an implementation assumption, not a measured timestamp difference.

Reaching within 5 m adds 50. A collision triggers a −80 termination penalty only after the first 3 seconds of the episode; reaching 600 steps incurs −80 and truncation. The order of these checks matters if multiple conditions hold.

This is a reward-shaping curriculum, not a curriculum that increases obstacle count or scene difficulty.

## 5. Network and current training configuration

The dual-branch extractor contains:

| Branch | Layers |
|---|---|
| Distance image | Conv 1→16→32→64, 5×5 kernels, batch normalisation and ELU; pooling after the first two convolutions; flatten 2304→1024→256 with final LayerNorm |
| Task vector | 8→32→64→96, ELU and final LayerNorm |
| Fusion | Concatenate to 352; 352→512→512→256 with ELU and final LayerNorm |

Actor and critic have separate MLP heads [512,256,256,256,128], using ELU, with the feature extractor shared by the delivered PPO configuration. No LSTM/GRU is present.

The current script specifies ten SubprocVecEnv workers, n_steps = 2560 per environment, batch_size = 256, n_epochs = 4, gamma = 0.99 and GAE lambda = 0.95. This yields 25,600 transitions per rollout in that configuration.

The curriculum and entropy callbacks reach their endpoints at 5.12 million steps. Learning rate (1e−4→1e−6) and clip range (0.2→0.1) use PPO's training-progress schedule for the learn budget of approximately 6 million steps. These are different scheduling mechanisms. The delivered settings and historical run metadata are not fully reconciled.

## 6. BSHM mapping

[BSHM.py](../methods/AirSim_ReinforceLearning/BSHM.py) stores blocks and integer-keyed voxel probabilities in hash maps. The legacy OctreeMapManager name is an alias; Open3D octrees are used for rendering, not occupancy traversal.

With 64 m blocks and depth parameter 8, voxel side length is 0.25 m. The implementation stores probabilities, converts them to odds, applies capped multiplicative Bayesian updates, and converts back. It does not store a log-odds field.

The active perception path downsamples LiDAR endpoints at 0.5 m and updates obstacle occupancy; it does not ray-trace free-space clearing. Background maintenance runs at a nominal 4 Hz, decays occupied probabilities, deletes observations older than 8 seconds, and bounds block count with an LRU policy (default 2048).

Manager/block locks protect shared state. Non-reentrant locks must not be recursively acquired; doing so can deadlock rather than raise an explicit error.

## 7. Hybrid shield extension

The [evaluation environment](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py) applies the following operations when safe_rule is enabled:

| Operation | Role |
|---|---|
| Proximity repulsion | Push away from occupied directions within distance ratio 0.25, or 5 m |
| Adaptive VO horizon | Scale prediction time by speed/proximity, clipped to 0.5–4 s |
| VO correction | Build velocity constraints; use an L1 linear-program correction, with geometric fallback |
| Local RRT* | Plan continuously in the background at nominal 2 Hz with a horizon up to 8 m, shortened near the goal; blend a recent result only when correction strongly slows the policy |
| Policy blending | Reduce large opposing corrections by interpolation; blend toward the policy near the goal |
| Distance modulation | Begin limiting speed below ratio 0.35 (7 m); hard cap 1.5 m/s below ratio 0.1 (2 m) |

Obstacles extracted for VO have zero assigned velocity. This is not dynamic-obstacle tracking.

Interpolation with coefficient 0.6 does not mathematically guarantee a correction angle below 90°. Post-processing, RRT* mixing and speed normalisation also do not re-establish every original VO constraint. The implementation has no end-to-end formal collision-avoidance proof, no demonstrated global-optimality preservation and no real-flight guarantee.

## 8. Scope limits

Stage 1 addresses static obstacles with simulator-provided state. Unknown space can appear open in the observation. Control-loop frequency and latency depend on the complete running stack; sleep intervals alone do not measure them.

The double-layer map maintenance and hybrid shield are repository extensions outside the article's stated scope. Isaac Lab migration, PX4 SITL / ROS 2 integration, later reward/network research, Sim2Real and alternative morphologies are described in the [development plan](../ROADMAP.md).
