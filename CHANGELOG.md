# Changelog

**English** | [简体中文](CHANGELOG.zh-CN.md)

[← Back to README](README.md) · [Roadmap](ROADMAP.md)

All notable changes to this repository. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-06-04 · *Stage 1 archival snapshot*

The first public release. Corresponds to branch **`origin_26-0364`**, frozen at the
state of the project when the journal article was accepted.

### Added

- **Simulation.** Distributed parallel AirSim platform: one `airsim_datamanager`
  process serving all drones over UDP (fragmented pickles, `5812`/`5813` plus a
  per-drone `2000+id` control port), 20 Hz sensor/command loop. Multi-drone UE
  blueprint patch so that drones neither collide with nor occlude one another.
- **Mapping.** `methods/AirSim_ReinforceLearning/BSHM.py` — block-sparse
  probabilistic hash map, 64 m blocks / 0.25 m voxels, Bayesian log-odds updates,
  dual-layer temporal decay (4 Hz probability decay `λ = 0.015`, 8 s staleness
  eviction, LRU block cap of 2048).
- **Policy.** Dual-branch feature extractor (`OctomapCNN`) + PPO with separate
  actor/critic trunks; 656-D observation (18×36 heading-frame range image + 8-D
  task vector); 3-D velocity-increment action.
- **Reward.** Five dense terms (directionally weighted obstacle, multi-budget
  smoothness, gated forward, progress, survival) plus asymmetric sparse terminals,
  with the urgency coefficient `k_avoid` coupling obstacle proximity to the
  forward incentive.
- **Curriculum.** `PPO_SB3_AirSim_CL.py` — progressive reward shaping driven by a
  training-progress callback, paired one-for-one with the control group
  `PPO_SB3_AirSim.py` (the two differ by two lines).
- **Safety shield.** `PPO_SB3_AirSim_model.py` — proximity repulsion, adaptive
  horizon velocity obstacles (LP projection), RRT\*-Safe at 2 Hz, distribution-shift
  guard, target-proximity gate, distance-based speed modulation.
- **Evaluation.** Monte-Carlo driver with success rate, collision rate, return and
  episode duration; `ablation_experiment.py` for the 5-configuration shield
  ablation; `collision_diagnostics.py` for automated failure attribution.
- **Off-policy controls.** SAC and DDPG variants sharing the environment, reward,
  network topology and curriculum.
- **Data.** 51 training runs — raw TensorBoard event files committed (50 MB),
  plus decimated CSV curves in `benchmarks/tb_curves/`; ablation summary and a
  per-step failure report in `benchmarks/`.
- **Tooling.** `scripts/export_tb_curves.py` (dependency-free TensorBoard scalar
  reader), `scripts/repo_hygiene.py` (oversize/binary guard).
- **Docs.** `README.md` (+ `README.zh-CN.md` Chinese mirror with a language switch),
  `docs/ARCHITECTURE.md`, `docs/REPRODUCING.md`, `benchmarks/README.md`, `MODELS.md`,
  `ROADMAP.md`, `paper/README.md`, `CITATION.cff`.
- **Environment.** `environment.yml` — the exact conda export from the machine that produced
  the results: **WSL2 · Ubuntu 24.04 · RTX 4070 SUPER · CUDA driver 12.0**, Python 3.11.14,
  `airsim 1.7.0`, `stable-baselines3 2.7.0`, `torch 2.6.0+cu124`, `numpy 2.0.1`.
  `requirements-legacy.txt` pinned to the same versions as a pip-only fallback.
- **Simulator configuration.** `sim/airsim/settings.json` — the exact AirSim config used for the
  results: **10** `SimpleFlight` drones `UAV01…UAV10`, and the real LiDAR (**32 channels, 6 Hz,
  8000 pts/s, 80 m range**, `±30°` vertical, `360°` horizontal, `SensorLocalFrame`), plus
  `CameraDefaults` from the earlier camera study. `sim/airsim/README.md` explains what each setting
  implies — including why all ten drones spawning on one point makes the multi-drone isolation patch
  mandatory, and the coarse (~8.6°) horizontal sampling that explains the oblique-surface failure
  mode. This **supersedes `methods/Sensors1.json`**, which documents a 16-channel / 25 m / 5 Hz
  sensor and never matched the published runs.
- **Training environment.** `sim/airsim/project/` (13.6 MB) — the **scene itself**: the
  `Blocks2_AirSim170` UE4 project's `FlyingExampleMap.umap` (12.3 MB, the 250 × 250 × 50 m field with
  the ~70 obstacle actors), the mesh assets it instances, `Blocks.uproject` and `Config/`.
  `sim/airsim/figures/training_scene.png` is a view of the environment as it starts up.
  `sim/airsim/ENVIRONMENT.md` documents the whole environment: AirSim plugin **1.7.0**, **Unreal
  Engine 4.27** (matching the `EngineAssociation` in `Blocks.uproject`), the two ways to run it
  (standalone staged build vs. editor project), what the 4.69 GB unpacked archive contains, why only
  the scene is committed, and the SHA-256 of the full archive so any download can be verified.
- **Release artifacts for the simulator.** The 1.29 GB environment archive was repacked into two
  lean archives, both attached to the `v0.1.0` release with checksums:
  `airsim-blocks2-standalone-win64.zip` (**219 MB**, the packaged game — runs with no Unreal Engine
  install) and `airsim-blocks2-project-win64.zip` (**375 MB**, the editable project). All
  `Intermediate/` and `Saved/` content was removed at any nesting depth — 870 MB of build cache and
  editor scratch — with nothing runnable or editable lost. The nested case matters:
  `Plugins/AirSim/Intermediate/` alone was 78 MB of compiled objects.
- **Documentation of the deployment interface.** The IP configuration is now spelled out from the
  code as shipped: a table of the three distinct addresses (`airsim_ip` and `trainmanager_ip` in
  `airsim_datamanager.py`, plus `datamanager_ip`/`trainmanager_ip` in `make_env()`), their file and
  line, their current values, and the two-machine topology they imply. Also the
  `5812`/`5813`/`2000+index` port map, the three places the drone count must agree, and the WSL2 NAT
  and firewall pitfalls.
- **Reference links.** A "Built on / related repositories" table linking AirSim, Stable-Baselines3,
  SB3-Contrib, Open3D, Numba, SciPy and Gymnasium, with their documentation sites.

### Changed

- **Run names now state the experiment.** SB3's auto-generated TensorBoard directories
  (`PPO_98`, `PPO_99`, `SAC_3`, …) were renamed to `date_role-budget`, and the curve CSVs follow the
  same scheme so a curve and its run match by eye:
  `2026-03-18_paper-curriculum-6M`, `2026-04-13_paper-control-6M`, `2026-04-27_paper-sac-6M`
  (the three paper runs) plus `2026-03-11_pilot-5M`, `2026-02-07_pilot-5M`, `2025-12-23_pilot-4M`.
  Thirteen aborted/probe runs (< 0.5 MB each) are parked under `logs/_short-runs/` with their
  original names, and `logs/README.md` indexes every one of them, including the note that `PPO_94`
  was the interrupted first attempt at the curriculum run rather than a separate experiment.
  Checkpoint directories in `models/` were renamed to match; `scripts/export_tb_curves.py`,
  `benchmarks/README.md` and `MODELS.md` were updated accordingly.
- `RecurrentPPO_SB3_AirSim.py` → **`PPO_SB3_AirSim.py`**: the policy is a plain
  feed-forward PPO; the old name was a leftover from an earlier experiment. The
  legacy prototypes in `body-axis/`, `world-axis-v2/` and
  `airsim-only-with-lidar/` keep their historical names so that published results
  remain traceable to their training scripts.
- Pruned 119 intermediate checkpoints (6.0 GB) from the snapshot. They are
  preserved offline and were never needed to reproduce a published number; see
  `MODELS.md`.
- Restored the PPO configuration actually used for the reported experiments in
  `PPO_SB3_AirSim.py` / `PPO_SB3_AirSim_CL.py`: **`n_steps=2560` per environment
  with 10 parallel environments** (25 600-transition rollout). The SAC/DDPG controls
  were aligned to the same 10-environment fleet.

### Fixed

- Committed the training logs instead of ignoring them — they are the primary
  evidence behind every curve in the paper.

### Known limitations

- Static obstacles only; no dynamic-obstacle experiments.
- Simulated state measurement is assumed noise-free (documented assumption).
- Generalisation to an unseen corridor scene drops to 68.2 %.
- Observation frame (velocity heading) and action frame (world) differ by design;
  see `docs/ARCHITECTURE.md` §2.2.
- No sim-to-real. No hardware-in-the-loop. Both are Stage 2/3 in `ROADMAP.md`.

---

## [Unreleased]

### Added

- **A Chinese version of every document.** All fifteen documents now come in pairs — `X.md` next to
  `X.zh-CN.md` — each with a language-switch line under its title, and Chinese pages link to the
  Chinese twin of another document rather than bouncing the reader back to English. Three
  conventions are enforced by `python scripts/repo_hygiene.py --all --docs` and by a GitHub Actions
  workflow (`.github/workflows/docs.yml`): the twin exists, the switch line points at it, and no
  relative link is dead.
- **The isolation patch is now identifiable.** It is a single blueprint asset,
  `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset`, in which the drone-mesh collision channel
  is disabled and the LiDAR ray cast skips `Pawn` objects so beams pass through drones. It ships in
  both release archives and its bytes are committed at
  `sim/airsim/project/Content/BP_FlyingPawn.uasset` (sha256 `0f4f91c1…`). Previously the docs said
  only "two Unreal blueprint edits", which left readers unable to locate or verify it.
- **Direct download links** to the published `v0.1.0` release, so readers no longer have to hunt for
  the binaries: a release badge and a download pointer in both READMEs, per-asset links in
  [`MODELS.md`](MODELS.md), and per-asset links for the two simulator archives in
  [`sim/airsim/ENVIRONMENT.md`](sim/airsim/ENVIRONMENT.md).
- `MODELS.md` records that `SFT_avoid_modelv3.zip` and the curriculum run's `final_model` checkpoint
  are **byte-identical** (sha256 `f1fbdf2a…`) — only one of the two needs downloading.

### Fixed

- `methods/AirSim_ReinforceLearning/vehicle_manager.py::AirSim_traject()` was missing a comma
  between two waypoints, so the literal was parsed as a subscript —
  `[200, 100, -5][0, -100, -5]` — and the helper raised `TypeError` when called. No published number
  changes: the function only drives the scripted data-collection flights, never the PPO training or
  evaluation path. The frozen archival branch keeps the original bytes.
- `sim/airsim/ENVIRONMENT.md` claimed the isolation patch would *not* survive a rebuild from the
  project archive. That is wrong: the patched `BP_FlyingPawn.uasset` is inside the project archive,
  so a rebuild keeps it.

### Planned

- Isaac Lab port (GPU-parallel, 512–1024 environments) — `ROADMAP.md` Stage 2.
- PX4 SITL software-in-the-loop; policy → `OFFBOARD` velocity set-points —
  Stage 3.
- Safety layer extracted into a standalone package with a ROS 2 node — Stage 4.
- Observation/action frame unification with an action-scale curriculum — Stage 2.
