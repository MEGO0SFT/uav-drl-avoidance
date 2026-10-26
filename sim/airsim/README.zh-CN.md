# AirSim 仿真器配置

[**English**](README.md) | **简体中文**

[← 返回 README](../../README.zh-CN.md) · [UE4 工程](ENVIRONMENT.zh-CN.md)

这个目录里放着两样东西：

- [`settings.json`](settings.json) —— AirSim 运行时配置（无人机 + 传感器），下文逐一说明。
- [`ENVIRONMENT.md`](ENVIRONMENT.zh-CN.md) —— 上述配置所配套的 **UE4 工程**：
  `Blocks2_AirSim170`、AirSim 插件 1.7.0、训练地图 `FlyingExampleMap.umap`，
  [`project/`](project/) 下入库了什么（场景，13.6 MB），以及另外单独分发了什么
  （1.29 GB 的构建包）。

[`settings.json`](settings.json) 是发表实验所用的 **确切** AirSim 配置。请把它放到 AirSim 查找配置的位置：

| 平台 | 路径 |
|---|---|
| Windows | `%USERPROFILE%\Documents\AirSim\settings.json` |
| Linux / WSL2 | `~/Documents/AirSim/settings.json` |

> ⚠️ **`methods/Sensors1.json` 是一份过时的模板。** 它描述的是 16 线 / 25 m / 5 Hz 的 LiDAR，
> 与本文结果**并不相符**。产出论文的数值是下方 `settings.json` 中的那一组。二者冲突时，
> 以本文件为准。

## 配置了什么

| 设置项 | 取值 | 为什么重要 |
|---|---|---|
| `SimMode` | `Multirotor` | 四旋翼模式 |
| `ClockSpeed` | `1` | 实时——训练从未使用加速仿真 |
| `SettingsVersion` | `1.2` | AirSim ≥ 1.5 |
| `Vehicles` | **10** 架 `SimpleFlight`，`UAV01`…`UAV10` | 每个并行环境一架无人机 |
| 无人机生成位置 | 全部为 `X=0, Y=0, Z=-0.5, Yaw=0` | **十架全部生成在同一点** —— 见下文的隔离补丁 |
| `OriginGeopoint` | 31.40527 N, 121.48941 E, alt 0 | 仅用于设置局部 NED 原点；策略在相对 NED 坐标系下工作，因此不影响结果 |
| `DefaultSensors.Lidar1` | 见下文 | 策略唯一使用的传感器 |
| `CameraDefaults` | 3 种图像类型，256 × 160，FOV 90° | 为早期的相机研究配置；最终策略**未使用** |

### LiDAR

| 参数 | 取值 |
|---|---|
| `SensorType` | `6`（AirSim LiDAR） |
| `NumberOfChannels` | **32** |
| `RotationsPerSecond` | **6** |
| `PointsPerSecond` | **8000** |
| `Range` | **80 m** |
| `VerticalFOVUpper` / `Lower` | `+30` / `-30` |
| `HorizontalFOVStart` / `End` | `0` / `360` |
| `DataFrame` | `SensorLocalFrame` |
| `DrawDebugPoints` | `false` |

在与其他工作对比之前，有两点需要先知道：

- **80 m 量程 vs. 20 m 地图查询窗口。** 传感器的感知距离远大于策略实际摄入的范围：
  `vehicle_manager.py` 以无人机周围 ± 2·`stat_range`（±40 m）的 AABB 构建 18×36 的观测，
  并在 `stat_range = 20 m` 处对每个栅格单元做截断。超出该范围的部分虽然被建图，但从未进入
  网络的观测。
- **水平采样偏粗。** `PointsPerSecond` 是**所有**通道共享的总预算，因此隐含的分辨率为
  `8000 / (32 × 6) ≈ 42` 点每通道每圈 —— 相邻采样点之间约 8.6° 方位角。轨迹的大部分对此
  足够平滑，但也正因如此，斜掠（"grazing"）表面分辨得很差：与箱体棱边平行的光束可能从
  采样点之间溜过，使距离图像出现非线性跳变。这正是论文泛化性一节讨论的失效模式。

`DataFrame: SensorLocalFrame` 表示 Python 客户端收到的是**传感器局部**坐标系下的点；
`methods/AirSim_ReinforceLearning/LidarProcess.py::LidarDataB2G` 会在这些点进入地图之前，
先把它们旋转到全局 NED 坐标系。

## 多机隔离补丁（必做）

十架无人机全部生成在*同一点*。用原版 AirSim 时，这无法用于单智能体学习：无人机会互相碰撞，
而且 LiDAR 的射线会先打到其他无人机上，于是每一帧观测都被另外九个智能体污染。

补丁只落在一个资产文件里：

```
Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
```

它做了两处改动：

1. 关闭无人机网格体之间的碰撞通道（保留与世界几何体之间的碰撞）；
2. 让 LiDAR 射线跳过 `Pawn` 对象，使光束穿过无人机，只报告环境表面。

**此处发布的所有内容都已应用该补丁** —— 两个 release 压缩包中都有，同样的字节也已入库至
[`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset)
（sha256 `0f4f91c1…`，165 KB）。若要复用已发布的设置，请保持该文件原样。如果换回原版 AirSim
pawn，训练将无法收敛 —— 此时的奖励看到的是一个任何单机策略都无法建模的状态分布。

每个组件的精确设置，以及为什么需要两个不同的响应单元格（一个给 LiDAR，一个给机体间碰撞），
都写在 [**`ISOLATION_PATCH.md`**](ISOLATION_PATCH.zh-CN.md) 里。另见
[`../../docs/REPRODUCING.md`](../../docs/REPRODUCING.zh-CN.md) §2。
