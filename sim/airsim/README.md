# AirSim simulator configuration

**English** | [简体中文](README.zh-CN.md)

[← Back to README](../../README.md) · [The UE4 project](ENVIRONMENT.md)

Two things live in this folder:

- [`settings.json`](settings.json) — the AirSim runtime config (drones + sensors), covered below.
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — the **UE4 project** these settings were used with:
  `Blocks2_AirSim170`, AirSim plugin 1.7.0, the training map `FlyingExampleMap.umap`, what is
  committed under [`project/`](project/) (the scene, 13.6 MB) and what is distributed separately
  (the 1.29 GB build).

[`settings.json`](settings.json) is the **exact** AirSim configuration used for the published
experiments. Drop it where AirSim looks for it:

| Platform | Path |
|---|---|
| Windows | `%USERPROFILE%\Documents\AirSim\settings.json` |
| Linux / WSL2 | `~/Documents/AirSim/settings.json` |

> ⚠️ **`methods/Sensors1.json` is a stale template.** It describes a 16-channel / 25 m / 5 Hz LiDAR
> and does **not** match the results. The values that produced the paper are the ones in
> `settings.json` below. Where the two disagree, this file wins.

## What is configured

| Setting | Value | Why it matters |
|---|---|---|
| `SimMode` | `Multirotor` | quadrotor mode |
| `ClockSpeed` | `1` | real time — training was never run with accelerated simulation |
| `SettingsVersion` | `1.2` | AirSim ≥ 1.5 |
| `Vehicles` | **10** × `SimpleFlight`, `UAV01`…`UAV10` | one drone per parallel environment |
| Vehicle spawn | all at `X=0, Y=0, Z=-0.5, Yaw=0` | **all ten spawn on the same point** — see the isolation patch below |
| `OriginGeopoint` | 31.40527 N, 121.48941 E, alt 0 | sets the local NED origin only; the policy works in relative NED, so this does not affect results |
| `DefaultSensors.Lidar1` | see below | the only sensor consumed by the policy |
| `CameraDefaults` | 3 image types, 256 × 160, FOV 90° | configured for the earlier camera study; **not consumed** by the final policy |

### LiDAR

| Parameter | Value |
|---|---|
| `SensorType` | `6` (AirSim LiDAR) |
| `NumberOfChannels` | **32** |
| `RotationsPerSecond` | **6** |
| `PointsPerSecond` | **8000** |
| `Range` | **80 m** |
| `VerticalFOVUpper` / `Lower` | `+30` / `-30` |
| `HorizontalFOVStart` / `End` | `0` / `360` |
| `DataFrame` | `SensorLocalFrame` |
| `DrawDebugPoints` | `false` |

Two consequences worth knowing before you compare against other work:

- **80 m range vs. a 20 m map query window.** The sensor sees much further than the policy
  ingests: `vehicle_manager.py` builds the 18×36 observation from an AABB of ± 2·`stat_range`
  (±40 m) around the drone and clips each bin at `stat_range = 20 m`. Anything beyond that is
  mapped but never observed by the network.
- **Coarse horizontal sampling.** `PointsPerSecond` is the *total* budget across all channels, so
  the implied resolution is `8000 / (32 × 6) ≈ 42` points per channel per revolution — roughly
  8.6° of azimuth between samples. Most of the trajectory is smooth enough for this, but it is why
  oblique ("grazing") surfaces are poorly resolved: a beam parallel to a box edge can slip between
  samples and the range image jumps non-linearly. That is the failure mode discussed in the paper's
  generalisation section.

`DataFrame: SensorLocalFrame` means the Python client receives points in the **sensor-local** frame;
`methods/AirSim_ReinforceLearning/LidarProcess.py::LidarDataB2G` rotates them into the global NED
frame before they enter the map.

## The multi-drone isolation patch (mandatory)

All ten vehicles spawn at the *same* point. With stock AirSim that is unusable for single-agent
learning: drones collide with each other, and the LiDAR ray casts hit other drones first, so every
observation is polluted by the other nine agents.

The patch lives in one asset:

```
Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
```

It makes two changes:

1. disable the collision channel between drone meshes (keep collision with world geometry);
2. make the LiDAR ray cast skip `Pawn` objects, so beams pass through drones and only report
   environment surfaces.

**It is already applied in everything published here** — in both release archives, and the same
bytes are committed to git at
[`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset)
(sha256 `0f4f91c1…`, 165 KB). To reuse the published settings, keep that file exactly as it is. If you
swap in the stock AirSim pawn instead, training will not converge — the reward then sees a state
distribution that no single-drone policy can model.

The exact per-component settings, and why two different response cells are needed (one for the LiDAR,
one for body-to-body collision), are written up in
[**`ISOLATION_PATCH.md`**](ISOLATION_PATCH.md). See also
[`../../docs/REPRODUCING.md`](../../docs/REPRODUCING.md) §2.
