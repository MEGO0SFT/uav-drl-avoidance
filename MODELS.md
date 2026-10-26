# Model artefacts

[← Back to README](README.md) · [Benchmarks](benchmarks/README.md) · [Paper](paper/README.md)

Policy checkpoints are **not** stored in git (see `.gitignore`): they are binary
blobs, the largest exceed GitHub's 100 MB per-file limit, and the 119
intermediate checkpoints of the hyper-parameter sweep were pruned from the
archival snapshot. This file records what exists, what each artefact is, and how
to obtain it.

Training **logs** *are* in the repository (`RL_scripts/*/logs/`), together with
decimated CSV exports of every curve (`benchmarks/tb_curves/`).

## Published as Release assets

**Download page:** <https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0>
· checksums: `SHA256SUMS.txt` in that release.

Checkpoint directories are named after the run they belong to, matching
`RL_scripts/body-axis_v2/logs/` and `benchmarks/tb_curves/` one-to-one.

| Asset (download) | Size | Local path | What it is | Produced |
|---|---:|---|---|---|
| [`SFT_avoid_modelv3.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/SFT_avoid_modelv3.zip) | 47 MB | `body-axis_v2/results/SFT_avoid_modelv3.zip` | **Curriculum-group policy**, 6.0 M steps — the checkpoint behind **every** safety-shield number | success-rate curve "0.81", ablation table, generalisation |
| [`20260413-214351_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260413-214351_final_model.zip) | 47 MB | `body-axis_v2/models/2026-04-13_paper-control-6M/` | **No-curriculum control group**, 6.0 M steps | success-rate curve "0.764" |
| [`20260427-152015_SAC_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260427-152015_SAC_final_model.zip) | 106 MB | `body-axis_v2/models/2026-04-27_paper-sac-6M/` | SAC off-policy control, 6.1 M steps | SAC comparison |
| [`20260207-003758_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260207-003758_final_model.zip) | 47 MB | `body-axis_v2/models/2026-02-07_pilot-5M/` | Pilot run, 5.12 M steps | — |
| *(not published — historical)* | 47 MB | `world-axis-v2/models/20260311-205650/` | World-frame-map prototype | — |

> **One file, two names.** `SFT_avoid_modelv3.zip` and the curriculum run's
> `models/2026-03-18_paper-curriculum-6M/20260318-111106_final_model.zip` are
> **byte-identical** (sha256 `f1fbdf2a…`). The former name is kept because
> `PPO_SB3_AirSim_model.py` and `ablation_experiment.py` load it by that path; there is
> no point downloading both.

The simulator lives in the same release:
[`airsim-blocks2-standalone-win64.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip)
(209 MB, runs without Unreal Engine) and
[`airsim-blocks2-project-win64.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip)
(357 MB, editable in UE 4.27) — see [`sim/airsim/ENVIRONMENT.md`](sim/airsim/ENVIRONMENT.md).

Each zip is a complete Stable-Baselines3 `PPO`/`SAC` save file, loadable with
`PPO.load(path)` — the policy kwargs (`OctomapCNN`, `features_dim=256`,
`net_arch=[512,256,256,256,128]`) are stored inside the archive.

## Loading a policy

```python
from stable_baselines3 import PPO
model = PPO.load("SFT_avoid_modelv3.zip", device="cpu")
action, _ = model.predict(obs, deterministic=True)   # deployment uses deterministic=True
```

`PPO_SB3_AirSim_model.py` shows the full evaluation path, including the safety
shield, the success-rate bookkeeping and the Monte-Carlo driver.

## Pruned intermediate checkpoints

The original sweep produced 126 archives (6.0 GB), of which 119 were per-step
`CheckpointCallback` dumps. They are preserved offline by the author and can be
released on request; they are not needed to reproduce any number in the paper,
because every published figure is derived from (a) the final policies above and
(b) the training logs.
