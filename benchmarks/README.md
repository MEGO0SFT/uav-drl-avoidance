# Benchmarks

[← Back to README](../README.md) · [Architecture](../docs/ARCHITECTURE.md) · [Reproducing the results](../docs/REPRODUCING.md) · [Training-run index](../RL_scripts/body-axis_v2/logs/README.md)

Machine-readable archives of every number quoted in the paper. Text/JSON only —
no model weights (those live in Release assets, see [`../MODELS.md`](../MODELS.md)).

| File | Contents |
|---|---|
| `ablation_summary.json` | the 5-configuration safety-shield ablation, verbatim from `scripts/ablation_results/` |
| `failure_analysis_sample.txt` | per-step failure report for 16 failed episodes (the raw material behind the three failure modes in the paper) |
| `tb_curves/*.csv` | decimated training curves (success rate, return, episode length, KL, value loss, action std, entropy coefficient) for 6 runs |

## Safety-shield ablation

Identical policy (`SFT_avoid_modelv3.zip`), shield layers toggled cumulatively.
Distribution-shift guard and target-proximity gate always on.

| Configuration | Episodes | Success | Collision | Return | Success time |
|---|---:|---:|---:|---:|---:|
| Pure DRL policy | 309 | **90.94 %** | 9.06 % | 245.80 ± 78.36 | 22.31 ± 4.26 s |
| + proximity repulsion | 307 | 94.46 % | 5.54 % | 261.44 ± 81.76 | 24.38 ± 4.62 s |
| + adaptive VO-τ | 305 | 93.11 % | 6.89 % | 258.29 ± 85.50 | 24.36 ± 4.47 s |
| + RRT\*-Safe | 311 | 95.82 % | 4.18 % | 263.84 ± 71.64 | 24.65 ± 5.27 s |
| **Full system** | 525 | **96.95 %** | 3.05 % | **288.96 ± 68.27** | 24.70 ± 5.28 s |

Note the non-monotonicity at the VO-τ step (93.11 % < 94.46 %): adding the
velocity-obstacle layer on top of repulsion alone slightly regresses success,
while its interaction with RRT\*-Safe produces the largest single jump
(93.11 % → 95.82 %). The ablation is cumulative rather than leave-one-out, which
is what makes this interaction visible — a leave-one-out design would hide it.

## Training runs

Run directories are named after the experiment, not after SB3's auto-numbering, and each curve CSV
in `tb_curves/` carries the same name as its run directory. Full index, including the 13 parked
short runs: [`../RL_scripts/body-axis_v2/logs/README.md`](../RL_scripts/body-axis_v2/logs/README.md).

| Run directory (= CSV name) | Steps | `ent_coef` schedule | Final success | Peak success | Role |
|---|---|---:|---|---:|---:|---|
| **`2026-03-18_paper-curriculum-6M`** | 6.00 M | 0.05 → 0.005 | **0.810** | 0.816 | ★ curriculum group |
| **`2026-04-13_paper-control-6M`** | 6.00 M | 0.05 → 0.005 | 0.764 | 0.809 | ★ no-curriculum control |
| **`2026-04-27_paper-sac-6M`** | 6.14 M | auto | 0.373 | 0.699 | ★ off-policy control |
| `2026-03-11_pilot-5M` | 5.12 M | 0.03 → 0 | 0.878 | 0.894 | pilot |
| `2026-02-07_pilot-5M` | 5.12 M | 0.03 → 0 | 0.691 | 0.730 | pilot |
| `2025-12-23_pilot-4M` | 4.05 M | 0.03 → 0.001 | — | — | earliest full run |

`2026-03-11_pilot-5M` reaches a *higher* in-training success rate than either 6 M-step paper run. It
is a different configuration (entropy annealed to zero over 5.12 M steps), so it is not the reported
baseline — but it is why the repository keeps all runs rather than only the published pair.

## Regenerating the CSVs

```bash
python scripts/export_tb_curves.py RL_scripts/body-axis_v2/logs --out benchmarks/tb_curves
```
