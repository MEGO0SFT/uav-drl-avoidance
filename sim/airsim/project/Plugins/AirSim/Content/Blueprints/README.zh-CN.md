# 隔离 Pawn 的参考位置

[**English**](README.md) | **简体中文**

[← README](../../../../../../../README.zh-CN.md) · [补丁说明](../../../../../ISOLATION_PATCH.zh-CN.md)

完整 AirSim 工程中的隔离 pawn 加载路径为：

    Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset

本目录说明该预期路径。入库参考资产实际位于 [sim/airsim/project/Content/BP_FlyingPawn.uasset](../../../../Content/BP_FlyingPawn.uasset)，本目录不重复保存。

参考文件大小为 168,972 字节（165.0 KiB），SHA-256：

    0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2

补丁将机体网格的 Visibility 和 Pawn 响应设为 Ignore，保留文档中的世界物体响应，不是在 LiDAR 实现中新增 Pawn 过滤。

取得仿真器归档后，应核对实际加载的插件资产并完成补丁说明中的运行检查。字节身份不等于运行隔离，工程 Content 中的参考副本也不自动覆盖插件资产。上游资产保留其许可与署名。
