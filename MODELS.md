# Model and Simulator Artefacts

**English** | [简体中文](MODELS.zh-CN.md)

[← README](README.md) · [Summaries](benchmarks/README.md) · [Paper](paper/README.md)

Policy checkpoints are not tracked in Git. This catalogue records the final policy asset attached to the [v0.1.0 release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0); use the release's checksum manifest when obtaining it. Its presence here does not mean the download was revalidated during documentation maintenance.

## Released policy asset

The size is the approximate download size.

| Asset | Algorithm / purpose | Expected local use |
|---|---|---|
| [SFT_avoid_modelv3.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/SFT_avoid_modelv3.zip) | PPO curriculum checkpoint, nominal 6 M steps, approximately 47 MB | RL_scripts/body-axis_v2/results/SFT_avoid_modelv3.zip |

Verify the full checksum of the file you obtain rather than relying on names or abbreviated hashes.

## Loading requirements

Use PPO.load for the released policy archive. A saved checkpoint does not replace the Python source and compatible dependencies required to resolve its custom OctomapCNN feature extractor. Use the corresponding training implementation and confirm observation shape, action conventions and normalisation before evaluation.

The main PPO observation is 656-dimensional and the policy outputs three world-frame velocity increments. Evaluation mode and shield settings belong to the experiment configuration; the default main evaluation entry currently disables the shield.

The released weights allow a new evaluation when the required simulator and runtime are available. They do not restore the original training process. Retained summaries and figures cannot be independently reconstructed from this checkout alone.

## Simulator assets

The same release catalogues:

- [airsim-blocks2-standalone-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip): packaged Windows simulator.
- [airsim-blocks2-project-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip): editable Unreal project with AirSim dependencies.

See [the simulator guide](sim/airsim/ENVIRONMENT.md) for configuration, isolation-patch placement and licence boundaries.
