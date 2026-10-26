# AirSim／Unreal 仿真器资产

[**English**](ENVIRONMENT.md) | **简体中文**

[← README](../../README.zh-CN.md) · [配置](README.zh-CN.md) · [隔离补丁](ISOLATION_PATCH.zh-CN.md)

第一阶段环境源于 Microsoft AirSim 的 Blocks 示例。工程元数据指定 Unreal Engine 4.27，文档记录的插件／客户端版本为 AirSim 1.7.0。

发布的仿真器归档面向 Windows x64，训练环境导出面向 Linux／WSL2。仿真器、数据中心和训练器地址必须符合实际拓扑；本文不声称所有角色都运行在同一操作系统实例。

## Git 中的内容

[project](project/)包含场景资产、配置、最小 Blocks 游戏模块和参考补丁 pawn，不包含完整 AirSim 插件／构建依赖树或独立可执行程序。

| 本地资产 | 大小／作用 |
|---|---|
| [FlyingExampleMap.umap](project/Content/FlyingCPP/Maps/FlyingExampleMap.umap) | 12,872,972 字节，约 12.87 MB／12.28 MiB；训练场景 |
| [BP_FlyingPawn.uasset](project/Content/BP_FlyingPawn.uasset) | 168,972 字节／165.0 KiB；隔离参考资产 |
| [Blocks.uproject](project/Blocks.uproject) | 引擎关联与插件配置 |
| Config 和 Source | 工程配置与模块源码 |

场景文档描述为约 250×250×50 m 区域、约 70 个障碍 actor。下图是历史画面，不能替代对当前运行环境的验证。

![训练场景](figures/training_scene.png)

## 历史发布产物

[v0.1.0 Release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0)列出两个包：

| 产物 | 历史近似下载大小 | 用途 |
|---|---:|---|
| [airsim-blocks2-standalone-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip) | 219.4 MB | Windows 打包仿真器，无需安装编辑器 |
| [airsim-blocks2-project-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip) | 374.5 MB | 附 AirSim 组件的可编辑工程；编辑／重建需要 UE 4.27 及工具链 |

以上是已有清单信息，不表示本次检出重新下载并核验了产物。实际下载应核对 Release 校验清单。压缩大小与解压大小是不同量，不能混算。

## 运行与编辑

独立运行时，解压打包版本，向仿真器运行用户的目录安装 [settings.json](settings.json)，运行 Blocks.exe，仍需检查宿主前提和 RPC 连通性。

编辑时使用完整工程归档、对应 Unreal 版本及构建工具链。只克隆场景源码不包含全部插件库；能否正常加载或离线重建取决于完整产物及本地依赖，不能仅由 EngineAssociation 保证。

发布说明记录补丁 pawn 位于 Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset。应核对实际加载资产及运行隔离效果。替换 pawn 后需重新应用或验证[碰撞配置](ISOLATION_PATCH.zh-CN.md)，不能假定工程级参考副本会自动覆盖插件资产。

## 资产身份与行为

当前参考 pawn 的 SHA-256 为：

    0f4f91c148c81f5ca3ad7079d4d85ade190b65cecf4b1625cc26f8045efd2dd2

哈希标识字节，不证明运行行为。需验证机体之间不推挤、LiDAR 可透过另一机体看到背景障碍，以及与世界几何的碰撞仍可检测。

## 来源与许可

AirSim／Blocks 源于 Microsoft Research，保留上游许可。本工程原创布局、配置及修改内容按仓库[许可证](../../LICENSE)发布，同时受上游权利约束。

Unreal Engine 内容、经修改的上游资产和随包第三方库保留各自版权及许可条件。仓库 MIT 许可证不意味着整个仿真器包的全部组件都被重新授权。
