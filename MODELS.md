# Model artefacts

[← Back to README](README.md) · [Benchmarks](benchmarks/README.md) · [Paper](paper/README.md)

Policy checkpoints are **not** stored in git (see `.gitignore`): they are binary
blobs, the largest exceed GitHub's 100 MB per-file limit, and the 119
intermediate checkpoints of the hyper-parameter sweep were pruned from the
archival snapshot. This file records what exists, what each artefact is, and how
to obtain it.

Training **logs** *are* in the repository (`RL_scripts/*/logs/`), together with
decimated CSV exports of every curve (`benchmarks/tb_curves/`).

## Kept locally / published as Release assets

Checkpoint directories are named after the run they belong to, matching
`RL_scripts/body-axis_v2/logs/` and `benchmarks/tb_curves/` one-to-one.

| File | Size | What it is | Paper figure / table |
|---|---:|---|---|
| `body-axis_v2/models/2026-03-18_paper-curriculum-6M/20260318-111106_final_model.zip` | 47 MB | **Curriculum group**, 6.0 M steps | success-rate curve, "0.81" |
| `body-axis_v2/models/2026-04-13_paper-control-6M/20260413-214351_final_model.zip` | 47 MB | **Control group**, 6.0 M steps | success-rate curve, "0.764" |
| `body-axis_v2/results/SFT_avoid_modelv3.zip` | 47 MB | Policy used for **all** safety-shield evaluation | ablation table, generalisation |
| `body-axis_v2/models/2026-04-27_paper-sac-6M/20260427-152015_SAC_final_model.zip` | 106 MB | SAC off-policy control, 6.1 M steps | SAC comparison |
| `body-axis_v2/models/2026-02-07_pilot-5M/20260207-003758_final_model.zip` | 47 MB | Pilot run, 5.12 M steps | — |
| `world-axis-v2/models/20260311-205650/20260311-205650_final_model.zip` | 47 MB | World-frame-map prototype (historical) | — |

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
