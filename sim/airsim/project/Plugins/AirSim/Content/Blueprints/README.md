# Reference Location of the Isolation Pawn

**English** | [简体中文](README.zh-CN.md)

[← README](../../../../../../../README.md) · [Patch guide](../../../../../ISOLATION_PATCH.md)

A complete AirSim project loads the isolation pawn from:

    Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset

This directory documents that expected path. The tracked reference asset is stored at [sim/airsim/project/Content/BP_FlyingPawn.uasset](../../../../Content/BP_FlyingPawn.uasset), not duplicated here.

The reference size is 168,972 bytes (165.0 KiB), with SHA-256:

    0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2

The patch sets drone-mesh Visibility and Pawn responses to Ignore while retaining the documented world responses. It does not add a Pawn filter to the LiDAR implementation.

When obtaining a simulator archive, verify the actual loaded plugin asset and the runtime checks in the patch guide. File identity alone does not prove runtime isolation, and a reference in project Content does not automatically replace a plugin asset. Upstream assets retain their licence and attribution.
