# Training runs — index

[← Back to README](../../../README.md) · [Benchmarks](../../../benchmarks/README.md) · [Experiment lineage](../../README.md)

Every TensorBoard run kept with this branch, including what it was for. The directory names state
the experiment; the SB3 auto-generated names (`PPO_98`, `SAC_3`, …) are gone except for the parked
short runs.

View them all:

```bash
tensorboard --logdir RL_scripts/body-axis_v2/logs
```

## The runs that produced the paper's numbers

| Directory | Date | Steps | `ent_coef` | Final in-training success | Peak | What it is |
|---|---|---:|---|---:|---:|---|
| **`2026-03-18_paper-curriculum-6M`** | 2026-03-18 | 6.00 M | 0.05 → 0.005 | **0.810** | 0.816 | ★ **curriculum group** — the paper's "0.81", produced by `PPO_SB3_AirSim_CL.py` |
| **`2026-04-13_paper-control-6M`** | 2026-04-13 | 6.00 M | 0.05 → 0.005 | 0.764 | 0.809 | ★ **control group** — same script with the curriculum callback removed, i.e. `PPO_SB3_AirSim.py` as shipped |
| **`2026-04-27_paper-sac-6M`** | 2026-04-27 | 6.14 M | auto | 0.373 | 0.699 | ★ **off-policy control** — SAC with identical reward, curriculum and network |

All three train to the same budget: the curriculum and entropy schedules run over
`1 024 000 × 5 = 5.12 M` steps and are then held constant, so the last ~0.9 M steps of the 6 M total
are already at terminal weights.

## Earlier full runs (pilot work, not the reported result)

| Directory | Date | Steps | `ent_coef` | Final / peak success | Notes |
|---|---|---:|---|---|---|
| `2026-03-11_pilot-5M` | 2026-03-11 | 5.12 M | 0.03 → 0 | 0.878 / 0.894 | A different configuration, and it reaches a *higher* in-training success rate than either 6 M paper run. This is why the repository keeps every run: the best checkpoint is not the same thing as the reported experiment. |
| `2026-02-07_pilot-5M` | 2026-02-07 | 5.12 M | 0.03 → 0 | 0.691 / 0.730 | Same configuration as the run above, much worse outcome — run-to-run variance with a 5 Hz-ish environment. |
| `2025-12-23_pilot-4M` | 2025-12-23 | 4.05 M | 0.03 → 0.001 | — | Earliest full-length run; useful only as a "before" reference. |

## `_short-runs/` — aborted starts and probes

Thirteen runs smaller than 0.5 MB, kept verbatim under their SB3 names:

| Directory | Size | What it was |
|---|---:|---|
| `PPO_94` | 0.41 MB | The **first attempt at the curriculum run**, interrupted at 0.995 M steps and restarted as `2026-03-18_paper-curriculum-6M`. Do not treat it as a separate experiment. |
| `PPO_93` | 0.11 MB | Longest of the startup probes, 0.26 M steps, success never left 0 |
| `PPO_87`–`PPO_92`, `PPO_95`–`PPO_97` | ≤ 0.01 MB | Smoke tests / crashes within the first few thousand steps |
| `SAC_1`, `SAC_2` | ≤ 0.01 MB | Aborted SAC starts before `2026-04-27_paper-sac-6M` |

They are included because a training log that only contains the run that worked is not a log.

## Curve CSVs

`benchmarks/tb_curves/` holds one decimated CSV per run above, with the **same name** as the run
directory, so a curve and its TensorBoard source are trivially matched. Regenerate them with:

```bash
python scripts/export_tb_curves.py RL_scripts/body-axis_v2/logs --out benchmarks/tb_curves
```
