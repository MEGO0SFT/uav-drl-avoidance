# Model and Simulator Artefacts

**English** | [简体中文](MODELS.zh-CN.md)

[← README](README.md) · [Summaries](benchmarks/README.md) · [Paper](paper/README.md)

Policy checkpoints are not tracked in Git. This catalogue records the assets named in the [v0.1.0 release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0); use the release's checksum manifest when obtaining them. The catalogue does not mean those downloads were revalidated during documentation maintenance.

## Recorded policy assets

Sizes below are approximate historical download descriptions.

| Asset | Algorithm / purpose | Expected local use |
|---|---|---|
| [SFT_avoid_modelv3.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/SFT_avoid_modelv3.zip) | PPO curriculum checkpoint, nominal 6 M steps, approximately 47 MB | RL_scripts/body-axis_v2/results/SFT_avoid_modelv3.zip |
| [20260413-214351_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260413-214351_final_model.zip) | PPO no-curriculum comparison, nominal 6 M steps, approximately 47 MB | A locally managed model directory |
| [20260427-152015_SAC_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260427-152015_SAC_final_model.zip) | SAC comparison, approximately 6.144 M steps, approximately 106 MB | Load with SAC |
| [20260207-003758_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260207-003758_final_model.zip) | Historical PPO pilot, nominal 5.12 M steps, approximately 47 MB | Optional comparison, not the main reported baseline |

Earlier documentation records SFT_avoid_modelv3.zip and the curriculum run's final_model as identical files. Verify full checksums of the files actually obtained rather than relying on names or abbreviated hashes.

## Loading requirements

Use PPO.load for PPO archives and SAC.load for SAC archives. A saved checkpoint does not replace the Python source and compatible dependencies required to resolve its custom OctomapCNN feature extractor. Use the corresponding training implementation and confirm observation shape, action conventions and normalisation before evaluation.

The main PPO observation is 656-dimensional and the policy outputs three world-frame velocity increments. Evaluation mode and shield settings belong to the experiment configuration; the default main evaluation entry currently disables the shield.

Weights allow a new evaluation when the required simulator and runtime are available. They do not restore the original training process. Retained summaries and figures cannot be independently reconstructed from this checkout alone.

## Simulator assets

The same release catalogues:

- [airsim-blocks2-standalone-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip): packaged Windows simulator.
- [airsim-blocks2-project-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip): editable Unreal project with AirSim dependencies.

See [the simulator guide](sim/airsim/ENVIRONMENT.md) for configuration, isolation-patch placement and licence boundaries.

Intermediate checkpoints are not part of the public source distribution. This document makes no guarantee about the availability of privately held training records.
