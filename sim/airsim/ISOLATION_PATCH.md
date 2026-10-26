# Multi-drone Isolation Patch

**English** | [简体中文](ISOLATION_PATCH.zh-CN.md)

[← README](../../README.md) · [Settings](README.md) · [Simulator assets](ENVIRONMENT.md)

The parallel training setup uses multiple drone instances in one scene as independent samples. Stock inter-drone collision and LiDAR occlusion can introduce interactions absent from that protocol. This patch changes collision responses on the drone pawn; it is not a multi-agent avoidance policy.

## Reference asset

The tracked copy is [project/Content/BP_FlyingPawn.uasset](project/Content/BP_FlyingPawn.uasset). In a complete project, verify that the loaded AirSim pawn is:

    Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset

The reference contains 168,972 bytes (165.0 KiB), SHA-256:

    0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2

A matching hash establishes file identity only. It does not prove that the running scene loaded that asset or that every relevant component has the desired response.

## Component settings

Apply and inspect the following profile on BodyMesh and Prop0–Prop3:

| Category | Setting | Reference value |
|---|---|---|
| Collision | Preset | Custom |
| Collision | Enabled | Query and Physics |
| Collision | Object Type | Pawn |
| Collision | Simulation Generates Hit Events | Enabled |
| Collision | Generate Overlap Events | Enabled |
| Collision | Phys Material Override | QuadBodyPhysicsMaterial |
| Collision | Can Character Step Up On | Yes |
| Trace responses | Visibility | Ignore |
| Trace responses | Camera | Ignore |
| Object responses | Pawn | Ignore |
| Object responses | WorldStatic, WorldDynamic, PhysicsBody, Vehicle, Destructible | Block |

![Collision settings](figures/collision_settings_BP_FlyingPawn.png)

The editor screenshot is in Chinese: 碰撞预设 = Collision Presets, 碰撞已启用 = Collision Enabled, 对象类型 = Object Type; 忽略 / 重叠 / 阻挡 mean Ignore / Overlap / Block.

## Why the two response types matter

AirSim's documented LiDAR path traces on the Visibility channel. Setting mesh Visibility responses to Ignore lets those traces pass through other drone meshes. This is a response-profile change, not an added Pawn-type filter in the LiDAR source.

Pawn=Ignore is an object-response setting used to suppress inter-drone collisions. The documented world-object responses remain Block so scene obstacles retain their intended interaction. Keep the object type and other profile settings consistent with the reference.

## Installing or editing

Either edit and save the pawn actually mounted by the complete AirSim project, or place the tracked reference asset at the exact plugin path above in a compatible complete project. Keep any project-specific changes under version control separately.

Do not assume that copying a pawn into a different Content directory overrides the plugin asset. Check the referenced asset path in the project. Replacing it with an upstream pawn requires reapplying or verifying the response profile.

Release documentation records this patch in the project package. The Git checkout alone cannot establish the contents or runtime behavior of a separately downloaded archive.

## Runtime checks

1. Spawn the configured drones together and check that they do not push one another apart.
2. Place a second drone between the first drone's LiDAR and a background obstacle. The background surface should remain visible in the point cloud.
3. Verify that collisions with walls, ground and obstacle geometry are still reported.

File size, timestamps and visual overlap alone are insufficient to establish all three behaviors. Changing this isolation protocol may materially change the observation distribution and training outcomes; convergence is not guaranteed merely by installing the patch.
