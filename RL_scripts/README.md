# `RL_scripts/` — experiment lineage

[← Back to README](../README.md) · [Training-run index](body-axis_v2/logs/README.md) · [Architecture](../docs/ARCHITECTURE.md)

Five generations of the training stack, kept so the design decisions can be traced.
**Only one of them is the published implementation**; the rest are history and are
not maintained.

| Directory | Role | Map frame | Action | Curriculum |
|---|---|---|---|---|
| **`body-axis_v2/`** | ★ **published implementation** (policy + safety shield) | velocity-heading frame | world-frame Δv, ±3.75 m/s | switchable: `PPO_SB3_AirSim.py` = control, `PPO_SB3_AirSim_CL.py` = curriculum |
| `body-axis_v2_plot/` | plotting / Monte-Carlo trajectory export; training script is a byte-copy of `body-axis_v2` | same | same | same |
| `body-axis/` | earlier prototype: attitude body frame, normalised action, 15-D state | attitude body frame | ±1 scaled by `0.5·v_max` | always on |
| `world-axis-v2/` | earlier prototype: map pinned to the world frame, original `W_dir` idea | world frame (`yaw = 0`) | ±2 m/s | always on |
| `airsim-only-with-lidar/` | earliest prototype: 12-D state, 4-D action, quaternion-exact angle recovery | world frame, 0–360° | 4-D (goal-speed scale + xyz Δv) | step-count based |

## Why the published one is `body-axis_v2`

The map/observation frame went through three iterations:

1. **World frame** (`world-axis-v2`) — the 18×36 grid stays north-aligned, and the
   direction angles had to be *recovered from grid indices* to compute the
   directional-danger weight `W_dir`. Workable, but that angle recovery is an
   approximation and "which cell is ahead?" is not free.
2. **Attitude body frame** (`body-axis`) — grid rotated by the drone's yaw. The
   action then has to be rotated into that frame too, which made early exploration
   produce almost no displacement: the policy cannot move before it has learned the
   rotation, so the forward/progress rewards stay flat and there is nothing to learn
   from.
3. **Velocity-heading frame** (`body-axis_v2`) — grid rotated by the *velocity*
   direction instead. "Ahead" becomes the centre column of the image, `W_dir` falls
   out of the representation for free, and the action stays in the world frame so
   exploration works from step one.

Step 3 is what the paper reports. The asymmetry it leaves behind — heading-frame
observation, world-frame action — is deliberate; see
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §2.2.

## Naming

`body-axis_v2/scripts` was renamed `RecurrentPPO_SB3_AirSim*.py` →
`PPO_SB3_AirSim*.py`: the policy is a plain feed-forward PPO and the old name was a
leftover. The older directories keep their historical filenames on purpose, so that
a published number can always be traced back to the exact script that produced it.

## Per-experiment layout

Each generation follows the layout produced by its training script:

```
scripts/     code
models/      SB3 checkpoints, models/<timestamp>/...  (git-ignored)
logs/        TensorBoard event files                  (committed for body-axis_v2)
results/     named policy archives, evaluation data    (git-ignored)
```

`body-axis_v2/scripts/ablation_results/` holds the raw JSON behind the safety-shield
ablation; the published summary is mirrored to `../benchmarks/`.
