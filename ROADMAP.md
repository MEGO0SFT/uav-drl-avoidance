# Roadmap

**English** | [简体中文](ROADMAP.zh-CN.md)

[← Back to README](README.md) · [Architecture](docs/ARCHITECTURE.md)

Long-term plan for this repository. Stage 1 is finished and archived on branch
`origin_26-0364`; the work below is the migration to a modern simulation and
training stack.

Guiding goal: keep this a *self-contained, reproducible research engineering
project* rather than a pile of scripts — one that can be explained end to end.

---

## Stage 1 — AirSim + Stable-Baselines3 ✅ *archived*

Branch `origin_26-0364` (the paper's submission ID) is the frozen snapshot:

- distributed parallel AirSim simulator with a self-built UDP data-exchange hub
- BSHM probabilistic map with dual-layer temporal decay
- dual-branch feature extractor + PPO, curriculum-trained
- 5-layer hybrid safety shield, ablated over 5 configurations
- 51 training runs and all raw TensorBoard logs committed
- 90.94 % (policy) → 96.95 % (full system) success in the training scene; 68.2 %
  on an unseen corridor scene

---

## Stage 2 — Isaac Lab migration 🚧

**Why.** 8 CPU-simulated drones capped the throughput and forced a 48 h budget for
6 M steps; the reward function and curriculum were tuned around that scarcity.
GPU-parallel simulation changes what is worth trying.

| Task | Notes |
|---|---|
| Environment | Isaac Lab, PhysX **raycast** LiDAR (not RTX LiDAR, not cameras) to stay inside the VRAM budget |
| Scale target | 512–1024 environments on the available GPUs; measure and record the VRAM/throughput curve |
| Map representation | first port BSHM's semantics to a `torch` block-sparse voxel grid; then try Isaac Lab's native heightmap/ESDF as an alternative, and keep both for comparison |
| Policy | RSL-RL or skrl PPO; keep the DBE topology as the baseline so the two stages are comparable |
| Reward | re-run the ablation: start from a minimal (progress + terminal) reward, add each Stage-1 term back one at a time and measure its marginal contribution |
| Observation/action frames | **unify them** — action in the heading frame, and expose `sin ψ`/`cos ψ` explicitly. Solves the Stage-1 asymmetry; the "drone won't move early on" problem is handled with an action-scale curriculum (0.1 → 1.0) and an optional goal-directed PD warm start |
| Benchmark harness | one command → success rate, SPL, collision rate, time-to-goal, over a fixed scenario suite; must run in CI on a small subset |

Deliverable: a training curve that beats Stage 1's 90.94 % bare-policy baseline
under an identical evaluation protocol, plus a speedup number for the simulator.

---

## Stage 3 — PX4 SITL software-in-the-loop 🚧

**Why.** Stage 1 stops at "velocity set-point accepted by a simulator's idealised
controller". Real deployment means the policy talks to a flight controller, and
that interface is where most sim-to-real surprises live.

| Task | Notes |
|---|---|
| PX4 SITL | `simulator_mavlink` (HIL) backend — **no Gazebo needed**; Isaac Sim acts as the physics engine |
| Transport | MAVLink `HIL_SENSOR` / `HIL_ACTUATOR_CONTROLS`, or uXRCE-DDS to ROS 2 |
| Policy interface | `OFFBOARD` mode + `SET_POSITION_TARGET_LOCAL_NED` velocity set-points |
| Control stack to understand and document | cascaded PID: position → velocity → attitude → angular rate → mixer; `MPC_XY_VEL_*` / `MPC_Z_VEL_*`; EKF2 state estimation and its latency |
| Latency budget | measure end-to-end: policy inference + transport + PX4 loop + actuation; compare against the ~5.5 Hz effective rate of Stage 1 |
| Domain randomisation | mass/inertia, thrust curve, motor time constants, sensor noise, transport delay |

Deliverable: the same policy flying through the same scenario suite with PX4 in
the loop, with a quantified performance drop and an explanation of where it comes
from.

---

## Stage 4 — Safety layer as a first-class module

Stage 1's shield lives inside the environment and is toggled by subclassing for
the ablation. Reshape it into something deployable:

- standalone `safety_shield/` package with an explicit state machine and no gym
  dependency;
- a ROS 2 node wrapping it, so it can sit between any policy and any controller;
- formalise the guarantee it provides (invariant set / CBF-style argument) or,
  failing that, state precisely what it does *not* guarantee;
- re-ablate on the new stack, including leave-one-out this time, to separate
  individual contributions from the cumulative-order effect seen in Stage 1.

---

## Stage 5 — Perception and geometry

- Revisit the 18×36 range image: sparse 3-D convolution or a point-based encoder
  on the same voxel input, side by side with the Stage-1 representation.
- Directional-danger weighting `W_dir` currently recovers angles from grid indices;
  on the new stack the grid is generated in the heading frame, so the angles are
  exact. Quantify whether that changes anything.
- Dynamic obstacles. Stage 1 is static-only; treat temporal-decay map maintenance
  as the first step and add explicit motion prediction after that.

---

## Housekeeping (continuous)

- [ ] Publish policy checkpoints as GitHub Release assets and link them from `MODELS.md`
- [ ] CI: `ruff`, `scripts/repo_hygiene.py`, and a `-m "not gpu"` smoke test
- [ ] `pre-commit` hooks
- [ ] Broken-link and citation check in CI
- [ ] One short write-up per stage (a blog post is enough) so the reasoning is
      recoverable, not just the code

---

## Explicit non-goals

- Competing for benchmark SOTA. The value here is a defensible, documented,
  end-to-end stack — not a leaderboard entry.
- Supporting every simulator. AirSim is frozen as the archived reference;
  Isaac Lab is the forward path.
- Keeping the old prototypes (`body-axis`, `world-axis-v2`,
  `airsim-only-with-lidar`) working. They document the design evolution and are
  kept for that reason only.
