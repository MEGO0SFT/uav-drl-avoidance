# 多机隔离补丁就在这里

[**English**](README.md) | **简体中文**

[← 返回 README](../../../../../../../README.zh-CN.md) · [补丁详解](../../../../../ISOLATION_PATCH.zh-CN.md)

在仿真器压缩包里，这个 blueprint 位于

```
Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
```

**它正是承载多机隔离补丁的文件** —— 无人机网格体之间的碰撞通道被关闭，LiDAR 的射线检测被设为
跳过 `Pawn` 对象，于是光束穿过其他无人机，只报告环境表面。

它**没有**在本目录里再存一份，因为完全相同的字节已经入库到上一级目录：

```
sim/airsim/project/Content/BP_FlyingPawn.uasset
```

| | |
|---|---|
| 大小 | 168 972 bytes (165.0 KB) |
| sha256 | `0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2` |

压缩包内的两条路径（`Plugins/AirSim/Content/Blueprints/` 与工程级的 `Content/`）都持有**字节完全
相同**的副本，因此从工程压缩包重新构建后补丁依然保留。若要复用已发布的设置，请保持该文件原样。

回退到原版 AirSim pawn（那样会破坏训练）意味着要从[上游 AirSim](https://github.com/microsoft/AirSim)
1.7.0 恢复 `BP_FlyingPawn.uasset`。
