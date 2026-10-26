# Multi-drone isolation patch — exact reproduction

**English** | [简体中文](ISOLATION_PATCH.zh-CN.md)

[← Back to README](../../README.md) · [Simulator config](README.md) · [UE4 project](ENVIRONMENT.md)

Without this patch the platform cannot train a single-drone policy: all ten vehicles spawn on the
same point (`X=0, Y=0, Z=-0.5`), and stock AirSim lets them collide with and occlude one another, so
every observation is contaminated by the other nine agents.

The patch is **not** a C++ change and **not** a level edit. It is a collision-profile change on five
mesh components of one blueprint.

| | |
|---|---|
| Asset | `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` |
| Components touched | `BodyMesh`, `Prop0`, `Prop1`, `Prop2`, `Prop3` |
| Size / hash | 168 972 bytes · sha256 `0f4f91c1…` |
| Committed copy | [`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset) (identical bytes) |

> **⚠️ After editing, put the file back.** The edit must be made on
> `<project>/Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` and the **saved asset must then
> be copied back to exactly that path**, replacing the original — otherwise the game keeps loading the
> unpatched pawn. The path is part of the fix: AirSim's own copy under
> `Plugins/AirSim/Content/Blueprints/` is what the ten `SimpleFlight` vehicles instantiate, and a copy
> dropped anywhere else (for instance the project-level `Content/`) is only picked up if it shadows
> the same relative path. If you are unsure, save, close the editor, and confirm the file's mtime and
> size changed at `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset`.

![Collision settings of BodyMesh and Prop0..Prop3 in BP_FlyingPawn](figures/collision_settings_BP_FlyingPawn.png)

*The five mesh components of `BP_FlyingPawn`, Collision section — presets `Custom`, collision enabled
`Query and Physics`, object type `Pawn`, `Visibility` and `Camera` trace responses `Ignore`, and the
`Pawn` object response `Ignore`; everything else `Block`.*

## The settings

Apply these to **each of the five mesh components** (BodyMesh and the four propellers):

| Section | Setting | Value |
|---|---|---|
| Collision | Simulation Generates Hit Events | ✅ on |
| Collision | Phys Material Override | `QuadBodyPhysicsMaterial` |
| Collision | Generate Overlap Events | ✅ on |
| Collision | Can Character Step Up On | `Yes` |
| Collision | **Collision Presets** | **`Custom`** |
| Collision | Collision Enabled | **`Collision Enabled (Query and Physics)`** |
| Collision | Object Type | **`Pawn`** |
| Trace Responses | **Visibility** | **`Ignore`** |
| Trace Responses | Camera | `Ignore` |
| Object Responses | WorldStatic | `Block` |
| Object Responses | WorldDynamic | `Block` |
| Object Responses | **Pawn** | **`Ignore`** |
| Object Responses | PhysicsBody | `Block` |
| Object Responses | Vehicle | `Block` |
| Object Responses | Destructible | `Block` |

In the editor: open `BP_FlyingPawn`, select each mesh component → *Details → Collision* → expand
**Collision Presets**, set it to *Custom*, then expand *Trace Responses* / *Object Responses* and flip
the two cells marked **Ignore** above. Everything else stays at `Block`.

## Why exactly these two cells

The two goals are solved by two **different** response cells — this is the part that is easy to get
wrong.

### 1. LiDAR must not see other drones → `Visibility = Ignore`

AirSim's LiDAR is a ray cast on the **`ECC_Visibility`** channel, and it ignores only the drone that
fired the ray:

```cpp
// Plugins/AirSim/Source/UnrealSensors/UnrealLidarSensor.cpp:131
bool is_hit = UAirBlueprintLib::GetObstacle(actor_, ...,
                                            hit_result, actor_, ECC_Visibility);

// Plugins/AirSim/Source/AirBlueprintLib.cpp:615-626
trace_params.AddIgnoredActor(actor_);        // only the ego actor
if (ignore_actor != nullptr) trace_params.AddIgnoredActor(ignore_actor);
return actor->GetWorld()->LineTraceSingleByChannel(hit, start, end, collision_channel, trace_params);
```

There is no `Pawn` filter anywhere on that path. The **only** way a ray can pass through another drone
is for that drone's meshes to answer `Ignore` on `ECC_Visibility` — which is what the `Visibility`
cell does. Set it to `Block` and the LiDAR starts measuring the other nine drones instead of the
world.

### 2. Drones must not collide with each other → `Pawn = Ignore`

Body-to-body collision is an actor-vs-actor interaction, which UE resolves from the **object
response** matrix, not the trace channels. Every drone's meshes are `Object Type = Pawn`, so setting
the `Pawn` row to `Ignore` makes drone→drone overlap pass through, while `WorldStatic` /
`WorldDynamic` / `PhysicsBody` stay `Block` so the drone still hits walls, floors and the obstacle
actors normally.

## What must **not** change

- **`Object Type` must stay `Pawn`.** If it drifts to `WorldDynamic` the `Pawn` row no longer applies
  and the drones start colliding again.
- **`Collision Enabled` must stay `Query and Physics`.** `Query only` would remove the collision that
  the environment interaction and the crash detection rely on.
- **`Visibility` must stay `Ignore`.** This is the cell that makes the LiDAR transparent; it is also
  the one people "fix" by accident when they wonder why their LiDAR ignores their own drone.

## Verifying it took effect

1. Open the packaged simulator, let ten drones stand at the spawn point — they should overlap
   visually, not push each other apart.
2. Fly a second drone in front of the first and watch the first one's LiDAR points: they must appear
   *behind* the obstruction, not on its surface.
3. In PIE, `BP_FlyingPawn → BodyMesh → Collision` should read `Custom`, `Pawn`, `Visibility: Ignore`,
   `Pawn: Ignore`.

## Where the patch already is

| Location | Contains the patch |
|---|---|
| `airsim-blocks2-standalone-win64.zip` (release) | ✅ compiled in |
| `airsim-blocks2-project-win64.zip` (release) | ✅ as `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset`, so a rebuild keeps it |
| this repository | ✅ [`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset), byte-identical |

To inherit the published behaviour, keep that asset as it is. Restoring the stock pawn from
[upstream AirSim](https://github.com/microsoft/AirSim) 1.7.0 silently undoes the patch.
