# Historical Summaries and Final Figures

**English** | [简体中文](README.zh-CN.md)

[← README](../README.md) · [Architecture](../docs/ARCHITECTURE.md) · [Experiment guide](../docs/REPRODUCING.md)

This directory retains selected historical conclusion data and final figures. It is not a complete archive of the article's experimental evidence.

| Material | Scope |
|---|---|
| [ablation_summary.json](ablation_summary.json) | Aggregate results for five cumulative shield configurations |
| [classical_baselines_summary.json](classical_baselines_summary.json) | Aggregate APF/RRT* conclusions; small runs, not established as protocol-matched to the shield experiment |
| [figures](figures/) | Final training-convergence image and per-metric SVG panels |
| [Historical run labels](../RL_scripts/body-axis_v2/logs/README.md) | Text metadata for historical run labels |

The current checkout cannot recompute the original training curves, uncertainty across runs or per-episode statistics. New training/evaluation produces a new experiment.

## Shield extension: retained aggregate values

The same recorded PPO checkpoint was used with progressively enabled shield components. These are repository-extension results; the [article scope](../paper/README.md) does not include the shield experiments.

| Configuration | Episodes | Success | Historical failure field | Return | Successful-episode time |
|---|---:|---:|---:|---:|---:|
| Pure policy | 309 | 90.94% | 9.06% | 245.80 ± 78.36 | 22.31 ± 4.26 s |
| + repulsion | 307 | 94.46% | 5.54% | 261.44 ± 81.76 | 24.38 ± 4.62 s |
| + adaptive VO | 305 | 93.11% | 6.89% | 258.29 ± 85.50 | 24.36 ± 4.47 s |
| + RRT* | 311 | 95.82% | 4.18% | 263.84 ± 71.64 | 24.65 ± 5.27 s |
| Full system | 525 | 96.95% | 3.05% | 288.96 ± 68.27 | 24.70 ± 5.28 s |

Historical numeric values and JSON keys are preserved. collision_rate is calculated by the evaluator as 100 minus success rate, so it includes timeouts if present. It must not automatically be interpreted as a collision-only statistic.

The non-monotonic intermediate values describe conditional changes in this cumulative experiment. They do not independently identify individual module effects or demonstrate interaction mechanisms. Paired seeds and leave-one-out/factorial experiments would be needed for stronger attribution.

For groups where corrections are active, policy-blending and goal-proximity mechanisms are shared. With no shield correction, those mechanisms have no effect on the pure-policy group.

## Metric interpretation

Training success (curriculum 0.810, control approximately 0.764) is a rolling metric during exploration. The 90.94% / 96.95% values are historical deterministic-evaluation summaries. The 68.2% unseen-corridor value is a separate historical generalisation report. These protocols are not interchangeable.

The delivered 525-episode evaluation entry has 300-entry success/time buffers. Its final printed statistics do not automatically match the historical aggregate window. Current training parameters also remain incompletely reconciled with the old reporting cadence. These are reproduction limitations, not a retrospective recalculation of the retained tables.

Final figures remain public, including [training convergence](figures/ppo_training_convergence.png). Regenerating them requires the original run data.
