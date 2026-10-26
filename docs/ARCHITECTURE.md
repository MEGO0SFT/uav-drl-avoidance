# Architecture

[← Back to README](../README.md) · [Reproducing the results](REPRODUCING.md) · [Simulator environment](../sim/airsim/ENVIRONMENT.md)

Technical description of the Stage-1 system. Formulas are transcribed from the
implementation, not from the paper, so they match the code exactly. File and
function names are given so each claim is one jump away from the source.

---

## 1. Problem statement

A quadrotor must traverse 100–150 m of an *a priori* unknown, cluttered 3-D
environment and reach a goal it has never mapped, using only a mechanical
spinning LiDAR and its own kinematic state.

Modelled as a **model-free POMDP** with continuous state and action spaces. Four
deliberate restrictions keep the policy scene-independent:

- the policy never sees absolute position, only **relative goal direction**;
- the map observation is **rotated into the velocity-heading frame**, so "ahead"
  is always the same image region regardless of where the drone is pointing;
- the LiDAR pose is resolved through the drone's attitude, so **attitude itself is
  not part of the observation**;
- goals are sampled at random within a reachability band.

---

## 2. Observation — 656-D

### 2.1 Task vector `s_task` ∈ ℝ⁸
Constructed in `vehicle_manager.py::_Lidar_RLstat_update` (global NED frame):

| index | content | expression |
|---|---|---|
| `0:3` | goal direction | `(P_tgt − P_uav) / ‖P_tgt − P_uav‖` |
| `3` | task progress | `clip(1 − ‖P_tgt − P_uav‖ / L_max, −1, 1)`, `L_max = 150 m` |
| `4:7` | velocity direction | `v / ‖v‖` |
| `7` | normalised speed | `‖v‖ / v_max`, `v_max = 7.5 m/s` |

### 2.2 Environment image `s_env` ∈ ℝ⁶⁴⁸
`vehicle_manager.py::APF_SeqCalculate3D` bins every occupied voxel inside
`± stat_range` (`stat_range = 20 m`) into an elevation × azimuth grid,
**Δα = Δβ = 10°**, giving 18 × 36 cells. Each cell stores the **normalised
nearest range** in that direction (`1.0` = empty/far, `0` = touching):

```
alpha_world = atan2(y, x)
beta        = atan2(z, sqrt(x² + y²))
yaw         = degrees(atan2(v_y, v_x))          # ← velocity heading, not attitude
alpha_body  = wrap(alpha_world − yaw)           # azimuth in the heading frame
T_env(h, w) = clip(min_range_in_cell, 0, stat_range) / stat_range
```

Storing *range* rather than *occupancy* makes the image a dense, continuous
signal with a usable gradient, which matters because the reward is a direct
function of it.

> **Frame asymmetry, by design.** The observation is expressed in the
> velocity-heading frame, while the action (below) is added to the world-frame
> velocity vector. Placing the action in the body frame was tried first and made
> the drone barely move during early exploration — before the policy has learned
> the rotation, random actions largely cancel. The asymmetry is learnable because
> `s_task` carries `v_dir` (world frame), which is precisely the alignment basis
> of `s_env`; the policy can therefore recover the heading↔world rotation from the
> observation alone. The cost is that the policy performs a rotation internally
> that could be supplied analytically.

---

## 3. Action — 3-D continuous

```
a ∈ [−3.75, 3.75]³            # velocity increment Δv, m/s
v_des = v_current + a         # world NED
v_des ← v_des · 7.5/‖v_des‖   if ‖v_des‖ > 7.5
command = [vx, vy, vz, yaw]   # executed by a velocity-tracking PID cascade
```

Per-axis bound 3.75 m/s is comparable to `v_max`, giving a maximum three-axis
increment of √3·3.75 ≈ 6.5 m/s — enough for aggressive avoidance — while an
increment formulation means the policy only learns a *correction* to the current
motion instead of planning a velocity from scratch.

---

## 4. Reward

`r = r_dense + r_sparse`, with `r_dense = r_obs + r_smooth + (1 − k_avoid)·r_fwd + r_dist + r_surv`.

### 4.1 Obstacle term `r_obs`
`Avoid_Reward_Optimized_BodyFrame`, alert threshold `D = 0.2` (≈4 m, i.e. half a
reaction time at `v_max`).

- Alert set `𝒟 = {(i,j) : T_env(i,j) ≤ 2D}` (≤ 8 m).
- **`𝒟 = ∅`** → `r_obs = 1 − exp(−(d̂_min − 2D)/D)`.
- **`𝒟 ≠ ∅`** → directional-danger weight from π/6 sectors, weighted means
  multiplied:
  `W_dir = w̄_h · w̄_v` with `w_h = [1.75, 1.50, 1.25, 1.00]` (4 azimuth sectors,
  decreasing from directly ahead) and `w_v = [1.25, 1.00, 0.90]` (3 elevation
  sectors, decreasing away from the horizontal plane — horizontal obstacles are
  the dominant collision threat).
  - range penalty `f(d) = 1/e − exp(1 − d/D)`, which is 0 exactly at `d = 2D`
    (continuous gradient, no penalty beyond the alert radius);
  - `R_dist = W_dir · mean_{𝒟} f(clip(d, 0, 2D))`;
  - trend term over consecutive steps, `Δd = d̂_min(t) − clip(d̂_min(t−1), 0, 2D)`:
    `R_change = max(Δd/D, −1)` if closing, `min(Δd/2D, 0.5)` if opening — the
    closing penalty is twice as steep as the opening reward, so a policy cannot
    farm `R_change` by oscillating;
  - `r_obs = clip(R_dist + R_change, −4, 0.5)`.

`W_dir` is the paper's main reward-design claim: it makes the reward gradient
*direction-sensitive* rather than purely distance-based, so the same obstacle is
weighted differently depending on whether it lies ahead, to the side, or behind.

### 4.2 Urgency coefficient `k_avoid`
```
k_avoid = exp(−3·d̂_min) · clip(exp(W_dir − 1.5), 0, 1)   ∈ [0, 1]
```
It gates the forward reward: as the drone closes on a frontal cluster,
`(1 − k_avoid) → 0`, so the drone must first escape the danger state before it can
earn progress reward again. This is what removes the "keep pushing forward into
the obstacle" attractor of a plain forward-reward shaping.

### 4.3 Smoothness term `r_smooth`
`v_clip_reward_calc`, over the last `min(10, |history|)` velocity samples:

```
a_max(v) = 4.0 + 6.0·v/10        j_max(v) = 8.0 + 4.0·v/10
P_a = percentile75( max(|a| − a_max(v), 0) )     # sparse, tail-focused
P_j = max( max(|j| − j_max(v), 0) )              # peak-focused
P_ω = mean( ω² ),  ω = arcsin(‖v̂_t × v̂_t+1‖)/Δt
r_smooth = clip( exp(−0.1 P_a)·exp(−0.05 P_j)·exp(−0.05 P_ω) − 0.5, −0.5, 0.5 )
```

The multiplicative coupling means violating *any one* kinematic budget collapses
the whole term — the policy cannot trade a smooth acceleration profile against
violent heading changes. `r_smooth = 0.5` when all budgets are respected.

The adaptive limits `a_max(v)`, `j_max(v)` grow with speed so that high-speed
emergency manoeuvres remain inside the feasible set instead of being punished.

### 4.4 Progress, survival, sparse
- `r_fwd = clip( (d_tgt·v_dir)·v_norm·v_max / L_max · 50, −(1+p), +(1+p) )`,
  weighted by `(1 − k_avoid)`.
- `r_dist`: `+1` when normalized distance beats the episode best by > 0.01,
  `−1` when it increases while `k_avoid < 0.25`, else `0`. Rewards *new* progress
  only, so slow creeping cannot farm it.
- `r_surv = −0.1 − 0.1·p` (−0.1 → −0.2): a small constant cost that discourages
  loitering.
- `r_sparse`: `+50` on goal (`< 5 m`), `−80` on collision, `−80` on timeout
  (`≥ 600` steps). The failure penalty exceeds the success reward on purpose — it
  raises the expected cost of risk-taking near a near-success state.

---

## 5. Network — Dual-Branch Extractor (DBE)

`OctomapCNN` in `PPO_SB3_AirSim.py`. The two observation halves are heterogeneous
(dense 2-D range image vs. a 8-D task vector), so they are encoded separately:

```
range image (1×18×36)                       task vector (8)
  Conv 1→16, k5, pad2 + BN + ELU + MaxPool2     Linear 8→32 + ELU
  Conv 16→32, k5, pad2 + BN + ELU + MaxPool2    Linear 32→64 + ELU
  Conv 32→64, k5, pad2 + BN + ELU               Linear 64→96 + ELU + LayerNorm
  Flatten 2304
  Linear 2304→1024 + ELU
  Linear 1024→256 + ELU + LayerNorm
                    cat -> 352
                        Linear 352→512 + ELU
                        Linear 512→512 + ELU
                        Linear 512→256 + ELU + LayerNorm   → 256-D feature
```

Actor and critic are separate 5-layer MLPs `[512, 256, 256, 256, 128]` with ELU,
sharing only the extractor. Weights are Kaiming-normal (fan-in, ReLU).

**PPO configuration:** `n_steps=2560` per env, `batch_size=256`, `n_epochs=4`,
`γ=0.99`, `GAE λ=0.95`, `clip 0.2→0.1` (linear), `lr 1e-4→1e-6` (linear),
`ent_coef 0.05→0.005` (linear), `max_grad_norm = e`, initial action σ = 1.0.
Ten parallel environments (`SubprocVecEnv`) give a 25 600-transition rollout per
update; 6 M total steps.

---

## 6. Map — Block Sparse Hash Map (BSHM)

`methods/AirSim_ReinforceLearning/BSHM.py`. Note the module is imported under the
historical alias `om_o3d` and its classes are aliased `OctreeMapManager` /
`OctreeProbabilisticMap`, but **no octree traversal happens at runtime** — the
Open3D octree member exists only for rendering.

- **Two-level structure.** The world is tiled into cubic blocks of
  `s_b = 64 m`; each block is a hash map from integer voxel index to float
  occupancy probability, with voxel size `64/2^8 = 0.25 m`.
- **Bayesian update** in log-odds, so no history needs to be retained per voxel.
- **Insert only where observed** — sparse by construction.
- **Dual-layer temporal decay**, running on a background thread at
  4 Hz and fully decoupled from the control loop:
  - voxel level: `p ← max(p − p·λ, 0.5)` with `λ = 0.015`, plus unconditional
    eviction of voxels unobserved for `T_exp = 8 s` (handles voxels driven to very
    high confidence by repeated observation, which pure decay cannot remove
    quickly);
  - block level: LRU eviction under a hard `max_blocks = 2048` cap, preferring
    blocks with no occupied voxels.
- **Lock discipline**: manager lock before block lock, always; per-block locking
  keeps deadlock risk local. A non-reentrant block lock is used deliberately, so a
  recursive acquire fails loudly instead of deadlocking silently.

---

## 7. Hybrid safety shield

`PPO_SB3_AirSim_model.py`. The learned policy is optimistic in out-of-distribution
states; the shield is a deterministic verifier applied to its velocity command.
Layers run in a fixed order inside `safe_ruler()`:

| # | Layer | Trigger | Action |
|---|---|---|---|
| 0 | Proximity repulsion (APF-like) | any cell `T_env ≤ 0.25` (5 m) | `F = −Σ ê/d²` added to the body-frame velocity, scaled by urgency and density |
| 1 | Adaptive VO horizon | always | `τ = clip(τ₀ · clip(v/v_max) · clip(d̂_min/0.15), 0.5, 4.0)`, `τ₀ = 3 s` |
| 2 | VO-τ | velocity inside a collision cone within `τ` | minimum-norm correction via linear programming with ORCA half-share (0.5) |
| 3 | RRT\*-Safe | VO output `< 0.3·‖v_DRL‖` and `< 1.5 m/s` | blend with a 2 Hz background 150-iteration RRT\* direction, `β = 0.6` on a found path / `0.4` for a safest-direction fallback |
| 4 | Distribution-shift guard | `cos(∠v_DRL, v_safe) < 0` | clamp to `v_DRL + 0.6·(v_safe − v_DRL)` so correction never exceeds 90° |
| 5 | Target-proximity gate | `d_target < 30 m` | linearly hand control back to the policy over 30→10 m |
| 6 | Speed modulation | `d̂_min < 0.35` | linear derate to `0.2·v_max`; hard cap 1.5 m/s below `d̂_min = 0.1` |

Design intent: keep the policy's global optimality, borrow deterministic
guarantees only where the policy is demonstrably weak. The distribution-shift
guard is the load-bearing piece — without it, aggressive corrections move the
state out of the policy's training distribution and the shield becomes
self-defeating.

---

## 8. What is deliberately *not* here

- No recurrent memory. The POMDP is handled by the map (which is itself a memory)
  rather than by RNN state.
- No depth camera, no RGB. Evaluated, then dropped: the map already provides the
  geometric abstraction and LiDAR gives wider coverage per byte.
- No privileged teacher, no distillation. Added in the Stage-2 plan — see
  [`../ROADMAP.md`](../ROADMAP.md).
