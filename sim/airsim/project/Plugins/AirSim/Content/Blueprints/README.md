# The multi-drone isolation patch lives here

In the simulator archive this blueprint is

```
Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
```

That is **the file that carries the multi-drone isolation patch** — the collision channel between
drone meshes is disabled and the LiDAR ray cast is made to skip `Pawn` objects, so beams pass through
other drones and only report environment surfaces.

It is **not duplicated in this directory**, because the identical bytes are already committed one
level up at

```
sim/airsim/project/Content/BP_FlyingPawn.uasset
```

| | |
|---|---|
| size | 168 972 bytes (165.0 KB) |
| sha256 | `0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2` |

Both paths inside the archive (`Plugins/AirSim/Content/Blueprints/` and the project-level `Content/`)
hold **byte-identical** copies, so a rebuild from the project archive keeps the patch. To reuse the
published settings, keep this file exactly as it is.

Reverting to the stock AirSim pawn (which would break training) means restoring
`BP_FlyingPawn.uasset` from [upstream AirSim](https://github.com/microsoft/AirSim) 1.7.0.
