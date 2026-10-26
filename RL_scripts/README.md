# Experiment Directories

**English** | [简体中文](README.zh-CN.md)

[← README](../README.md) · [Architecture](../docs/ARCHITECTURE.md) · [Historical run labels](body-axis_v2/logs/README.md)

The five directories below record implementation variants and supporting evaluation work. They are not five independently validated algorithm generations.

| Directory | Role | Observation frame / action |
|---|---|---|
| body-axis_v2 | Main journal-policy implementation and later shield extension | Velocity-heading distance image; world-frame velocity increment |
| body-axis_v2_plot | Plotting and Monte-Carlo trajectory tools with related historical code | Similar representation; inspect the individual evaluation path |
| body-axis | Earlier prototype | Attitude/body-frame representation; different state/action conventions |
| world-axis-v2 | Earlier prototype | World-aligned distance image |
| airsim-only-with-lidar | Earliest prototype | Earlier state vector and four-dimensional action |

Historical names beginning with RecurrentPPO do not establish that an RNN is used. The final policy uses feed-forward PPO and a CNN/MLP extractor; map accumulation supplies temporal context.

## Main entry points

In body-axis_v2/scripts:

- PPO_SB3_AirSim_CL.py: reward-shaping curriculum group.
- PPO_SB3_AirSim.py: no-curriculum control.
- PPO_SB3_AirSim_model.py: policy evaluation and optional shield.
- airsim_datamanager.py: RPC/UDP data hub.
- vehicle_manager.py: mapping wrapper and observation construction.
- train_sac.py / train_ddpg.py: comparison implementations.
- ablation_experiment.py: configurable cumulative shield evaluation.
- collision_diagnostics.py: diagnostic recording for new runs.

These are related source implementations, not a guarantee of the exact executable/configuration used for every historical result. Some auxiliary imports retain old module names; the [experiment guide](../docs/REPRODUCING.md) lists current limitations.

## Representation evolution

The variants explore world-aligned, attitude-aligned and velocity-heading distance images. The final representation makes forward directions easier to locate in the image while retaining world-frame actions. Earlier observations about poor exploration motivated that choice; they are not a general proof that body-frame actions cannot learn.

The journal article covers the mapping/policy/curriculum work. Temporal map maintenance and the hybrid shield are additional repository features; see [paper scope](../paper/README.md).

## Local outputs

Training generates local logs and checkpoints; evaluation can generate local trajectory arrays and detailed reports. These remain local and Git-ignored.

[Historical run metadata](body-axis_v2/logs/README.md) is a text index of historical run labels. The retained aggregate shield JSONs are in scripts/ablation_results/ and [benchmarks](../benchmarks/README.md). Existing final images, SVGs and PDFs remain available.

Older prototype directories are retained for design context and are not promised to run with the current environment without adaptation.
