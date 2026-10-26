# AirSim 配置与采样协议

[**English**](README.md) | **简体中文**

[← README](../../README.zh-CN.md) · [仿真器产物](ENVIRONMENT.zh-CN.md) · [隔离补丁](ISOLATION_PATCH.zh-CN.md)

[settings.json](settings.json)是当前参考配置，定义初始位置相同的十架 SimpleFlight 机体 UAV01–UAV10，以及统一的 LiDAR 规格。methods/Sensors1.json 是历史模板。

## 安装配置

配置应放入仿真器进程实际读取的目录：Windows 通常为 %USERPROFILE%/Documents/AirSim/settings.json，Linux 构建通常为 ~/Documents/AirSim/settings.json。请核对实际运行用户与可执行环境；提供的 Release 仿真器面向 Windows x64。

仅有 JSON 不足以建立独立机体采样协议，还需在运行工程中验证[隔离补丁](ISOLATION_PATCH.zh-CN.md)，使机体不互相碰撞或遮挡雷达，同时保留与环境几何的交互。

## 传感器与仿真参数

| 字段 | 当前值 |
|---|---|
| SimMode | Multirotor |
| ClockSpeed | 1 |
| VehicleType | SimpleFlight |
| 初始位置 | X=0、Y=0、Z=−0.5 |
| LiDAR 通道数 | 32 |
| RotationsPerSecond | 6 |
| PointsPerSecond | 8000 |
| Range | 80 m |
| 垂直视场 | −30° 至 +30° |
| 水平视场 | 0° 至 360° |
| DataFrame | SensorLocalFrame |
| DrawDebugPoints | false |

ClockSpeed=1 表示没有在配置中主动加速，不保证负载下按墙钟实时运行。CameraDefaults 保留早期相机研究设置，最终策略不消费 RGB／深度图像。策略还使用仿真运动状态及相对任务信息，因此 LiDAR 不是唯一状态来源。

按理想点数预算计算，8000/(32×6) ≈ 41.7 点／通道／圈，对应平均约 8.64° 方位间隔。这是预算估计，不是均匀采样实测。斜掠表面的稀疏回波可能构成限制，但不能据此认定它是失败的唯一原因。

## 几何处理

LidarProcess.py 通过旋转和平移，将传感器局部点转换为世界 NED 坐标。当前路径按 0.5 m 对端点降采样，再写入 BSHM。

观测构造器查询 ±40 m 包围盒，将各方向距离饱和于 20 m，形成 18×36 图像；速度航向量化到 10°。空栅格不能区分未知空间与已确认自由空间。

OriginGeopoint 不直接进入最终策略向量，但修改世界／坐标配置仍需系统级验证。

## 部署检查

每个 worker 的 UAV 名称必须存在于 settings 并由数据中心服务；worker 可以只使用可用机体的子集。根据实际网卡配置 AirSim RPC 和双向 UDP，端到端测量决策／控制时序，不由传感器频率或 sleep 推断。参见[实验指南](../../docs/REPRODUCING.zh-CN.md)。
