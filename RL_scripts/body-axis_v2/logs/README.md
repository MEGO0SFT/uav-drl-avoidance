# Historical Training-Run Metadata

**English** | [简体中文](README.zh-CN.md)

[← README](../../../README.md) · [Summaries](../../../benchmarks/README.md) · [Experiment directories](../../README.md)

This is a metadata index of historical run labels and recorded summary values.

| Historical label | Nominal budget | Recorded entropy setting | Historical final / peak training success |
|---|---:|---|---|
| 2026-03-18_paper-curriculum-6M | 6.00 M | 0.05 → 0.005 | 0.810 / 0.816 |
| 2026-04-13_paper-control-6M | 6.00 M | 0.05 → 0.005 | approximately 0.764 / 0.809 |
| 2026-04-27_paper-sac-6M | 6.144 M | automatic | 0.373 / 0.699 |
| 2026-03-11_pilot-5M | 5.12 M | 0.03 → 0 | 0.878 / 0.894 |
| 2026-02-07_pilot-5M | 5.12 M | 0.03 → 0 | 0.691 / 0.730 |
| 2025-12-23_pilot-4M | approximately 4.05 M | 0.03 → 0.001 | not retained in this summary |

These values are historical summaries rather than newly recomputed measurements. The metric averages per-environment rolling episode-success windows; it is not a fixed deterministic test-set result.

The two nominal 6 M PPO runs are the reported curriculum/control comparison. Pilot runs used different configurations and are not interchangeable baselines. SAC uses automatic entropy adjustment, not PPO's linear entropy schedule.

The delivered trainer parameters have not been fully reconciled with the original reporting cadence. The current repository cannot independently reconstruct the training curves, run-to-run variability or all original hyperparameters.

Interrupted starts and diagnostic short runs are not part of this index. Newly generated event files should remain local and Git-ignored. Final training figures are retained in [benchmarks/figures](../../../benchmarks/figures/).
