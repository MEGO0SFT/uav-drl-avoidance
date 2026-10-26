# AirSim / Unreal Simulator Assets

**English** | [简体中文](ENVIRONMENT.zh-CN.md)

[← README](../../README.md) · [Settings](README.md) · [Isolation patch](ISOLATION_PATCH.md)

The Stage-1 environment derives from Microsoft's AirSim Blocks example. The project metadata targets Unreal Engine 4.27; the documented plugin/client version is AirSim 1.7.0.

Published simulator archives target Windows x64. The training environment export targets Linux/WSL2. Simulator, hub and trainer addresses must match the topology actually used; this document does not assert that all roles ran in one OS instance.

## What is tracked

[project](project/) contains scene assets, configuration, a minimal Blocks game module and a reference patched pawn. It is not the complete AirSim plugin/build dependency tree or a standalone executable.

| Local asset | Size / role |
|---|---|
| [FlyingExampleMap.umap](project/Content/FlyingCPP/Maps/FlyingExampleMap.umap) | 12,872,972 bytes; approximately 12.87 MB / 12.28 MiB; training scene |
| [BP_FlyingPawn.uasset](project/Content/BP_FlyingPawn.uasset) | 168,972 bytes / 165.0 KiB; reference isolation asset |
| [Blocks.uproject](project/Blocks.uproject) | Engine association and plugin configuration |
| Config and Source | Project configuration and module source |

The scene is described as an approximately 250×250×50 m area with roughly 70 obstacle actors. The image is a historical view and does not replace validation in the current runtime.

![Training scene](figures/training_scene.png)

## Recorded release packages

The [v0.1.0 release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0) catalogues two packages:

| Package | Historical approximate download size | Intended use |
|---|---:|---|
| [airsim-blocks2-standalone-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip) | 219.4 MB | Packaged Windows simulator, no editor installation required |
| [airsim-blocks2-project-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip) | 374.5 MB | Editable project with AirSim components; UE 4.27/toolchain required for editing or rebuilding |

These are recorded catalogue details, not files downloaded and reverified by the current source checkout. Use the release checksum manifest for the actual downloads. Compressed sizes and extracted sizes are different quantities.

## Running or editing

For a standalone run, extract the packaged build, install [settings.json](settings.json) for the simulator's user account, and run Blocks.exe. Host prerequisites and RPC connectivity still need to be checked.

For editing, use the project archive with the appropriate Unreal version and build toolchain. A scene-only clone does not provide all plugin libraries. Successful loading or an offline rebuild depends on the complete package and local dependencies; it is not guaranteed by EngineAssociation alone.

The release documentation records the patched pawn at Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset. Verify both the asset used by the simulator and the runtime isolation behavior. If the pawn is replaced, reapply or verify the [documented collision settings](ISOLATION_PATCH.md). A project-level reference copy should not be assumed to override a plugin asset automatically.

## Asset identity and behavior

The tracked reference pawn has SHA-256:

    0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2

This identifies bytes, not runtime behavior. Verify that drone meshes do not push each other, LiDAR still sees background obstacles through another drone, and collisions with world geometry remain detectable.

## Attribution and licences

AirSim/Blocks originate from Microsoft Research and retain their upstream licence. Original project-specific layouts, configuration and modifications are provided under the repository's [licence](../../LICENSE), subject to upstream rights.

Unreal Engine content, modified upstream assets and bundled third-party libraries retain their respective copyrights and licence conditions. The repository's MIT licence does not relicense every component in a simulator package.
