# AirSim Settings and Sampling Protocol

**English** | [简体中文](README.zh-CN.md)

[← README](../../README.md) · [Simulator packages](ENVIRONMENT.md) · [Isolation patch](ISOLATION_PATCH.md)

[settings.json](settings.json) is the current reference configuration. It defines ten SimpleFlight vehicles, UAV01–UAV10, initially at the same position, and a shared LiDAR specification. The older methods/Sensors1.json is a historical template.

## Install the configuration

Use the settings directory read by the simulator process: typically %USERPROFILE%/Documents/AirSim/settings.json on Windows or ~/Documents/AirSim/settings.json for a Linux build. Check the actual user account and executable environment; the supplied release packages target Windows x64.

The JSON alone is insufficient for the independent-drone protocol. Verify the [isolation patch](ISOLATION_PATCH.md) in the running project so drones neither collide with nor occlude each other, while still interacting with environment geometry.

## Sensor and simulation settings

| Field | Supplied value |
|---|---|
| SimMode | Multirotor |
| ClockSpeed | 1 |
| VehicleType | SimpleFlight |
| Initial position | X=0, Y=0, Z=−0.5 |
| LiDAR channels | 32 |
| RotationsPerSecond | 6 |
| PointsPerSecond | 8000 |
| Range | 80 m |
| Vertical field of view | −30° to +30° |
| Horizontal field of view | 0° to 360° |
| DataFrame | SensorLocalFrame |
| DrawDebugPoints | false |

ClockSpeed=1 requests no configured simulation acceleration; it does not guarantee wall-clock real time under load. CameraDefaults retains camera settings from earlier work, but the final policy does not consume RGB/depth images. It also uses simulator kinematics and relative task information, so LiDAR is not its only state source.

An idealised point-budget calculation gives 8000/(32×6) ≈ 41.7 points per channel per rotation, or approximately 8.64° average azimuth spacing. This is a budget estimate, not measured uniform sampling. Sparse returns on grazing surfaces are a possible limitation, not a demonstrated sole cause of failures.

## Geometry processing

LidarProcess.py transforms sensor-local points by rotation and translation into world NED coordinates. The active path downsamples endpoints at 0.5 m before inserting them into BSHM.

The observation builder queries a ±40 m box and saturates directional distances at 20 m, producing an 18×36 image. The speed-derived heading is quantised to 10°. Empty bins do not distinguish unknown space from verified free space.

OriginGeopoint is not directly included in the final policy vector. Changes to world/coordinate configuration still require system-level checks.

## Deployment checks

Every worker's UAV name must exist in settings and be served by the data hub; workers may use a subset of the available drones. Configure AirSim RPC and bidirectional UDP for the actual network interfaces. Measure effective decision/control timing rather than inferring it from sensor rate or sleep calls. See [the experiment guide](../../docs/REPRODUCING.md).
