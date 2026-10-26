# 训练环境 —— AirSim Blocks（UE4）工程

[**English**](ENVIRONMENT.md) | **简体中文**

[← 返回 README](../../README.zh-CN.md) · [仿真配置](README.zh-CN.md) · [复现结果](../../docs/REPRODUCING.zh-CN.md)

本仓库中的每一个数字都来自这个仿真环境。它就是**微软 AirSim 的 "Blocks" Unreal Engine 4
C++ 工程**，封装为 `Blocks2_AirSim170`：内置 AirSim 插件，并以训练场景作为默认地图。

| | |
|---|---|
| 工程名 | `Blocks2_AirSim170` |
| 上游 | [microsoft/AirSim](https://github.com/microsoft/AirSim) —— `Blocks` 示例工程（MIT） |
| AirSim 插件 | **1.7.0**（见 `Plugins/AirSim/AirSim.uplugin`） |
| Python 客户端 | `airsim==1.7.0` —— 见 [`../../environment.yml`](../../environment.yml) |
| 引擎关联 | **Unreal Engine 4.27**（`Blocks.uproject`：`"EngineAssociation": "4.27"`，已确认） |
| 平台 | **Windows**（指仿真器；训练脚本运行在 WSL2 的 Ubuntu 24.04 上） |
| 默认 / 唯一地图 | `Content/FlyingCPP/Maps/FlyingExampleMap.umap` |
| 完整压缩包 | `Blocks2_AirSim170.zip`，压缩后 **1.29 GB** / 解压后 4.69 GB |
| SHA-256 | `d65942ea5a7a40fad82ae941f0ddef0c4ad14650010eb10eab079b86106fa7f9` |
| 仿真配置 | [`settings.json`](settings.json) —— 10 架无人机 + LiDAR |

> **引擎版本：Unreal Engine 4.27。** 这与 `Blocks.uproject` 中的 `"EngineAssociation": "4.27"`
> 一致，因此用原版 4.27 安装即可直接打开该工程 —— 不会弹出重定向（retarget）提示。下文的
> 独立运行版则完全不需要引擎。

## 本仓库里有什么

`project/`（13.6 MB）存放的正是定义**场景**的那部分 —— 体积小、是二进制、且不可替代：

| 路径 | 大小 | 说明 |
|---|---:|---|
| `project/Content/FlyingCPP/Maps/FlyingExampleMap.umap` | 12.3 MB | **训练场景** —— 250 × 250 × 50 m 的场地，内含约 70 个障碍物 actor |
| `project/Content/Geometry/Meshes/1M_Cube*.uasset` | 0.17 MB | 障碍物实例化所用的立方体网格 |
| `project/Content/FlyingCPP/…`、材质、`BP_FlyingPawn.uasset` | 0.4 MB | AirSim 飞行 pawn 与基础材质 |
| `project/Blocks.uproject`、`project/Config/Default*.ini` | 4 KB | 工程描述（启用的插件）与地图/启动设置 |
| `project/Source/` | 3 KB | `Blocks` 游戏模块的桩代码 |

障碍物的布局**就在 `.umap` 内部**，表现为那三种基础网格上的 actor transform 集合（外加
Unreal 自带的圆柱／圆锥／球体图元 —— 它们属于引擎内容，所以在这里看不到）。这就是为什么一个
含约 70 个障碍物的场景只有 12 MB：它存的是变换，不是几何。

![训练场景](figures/training_scene.png)

环境启动后的样子：四面围墙的 250 × 250 × 50 m 场地，约 70 个障碍物 actor 由楼板、立柱、
圆柱、圆锥、棱锥、球体和组合体构成。该截图同样入库于
[`figures/training_scene.png`](figures/training_scene.png)。

## 哪些内容**不**在本仓库里，以及为什么

压缩包解压后为 4.69 GB，其中几乎没有场景本身：

| 压缩包中的部分 | 大小 | 处置方式 |
|---|---:|---|
| `Intermediate/` | 2.59 GB | **丢弃** —— UE 构建缓存，用 `GenerateProjectFiles` + 构建可重新生成 |
| `Saved/`（`StagedBuilds/` 除外） | 0.54 GB | **丢弃** —— 编辑器状态、自动保存、着色器烘焙产物、崩溃报告（含 5 份地图自动保存副本） |
| `Saved/StagedBuilds/WindowsNoEditor/` | 0.36 GB | **单独重新打包为独立产物** —— 可直接运行的独立版本 |
| `Plugins/AirSim/` | 0.68 GB | 保留（已去掉它自己的 `Intermediate/` 构建缓存 78 MB）—— 随工程附带的插件与依赖，使工程可离线构建 |
| `Binaries/`、`Content/`、`Config/`、`Source/`、`Blocks.uproject` | 0.52 GB | 保留在工程产物中（其中 `Content/` 还额外入库到 git） |

这就是为什么 release 提供的是**两个**压缩包，而不是一个 1.29 GB 的大包 —— 见下文。

## 如何运行

**方案 A —— 独立运行版（无需安装 Unreal Engine，下载约 200 MB）。**
解压 `airsim-blocks2-standalone-win64.zip`，把 [`settings.json`](settings.json) 放到
`%USERPROFILE%\Documents\AirSim\`，然后运行 `Blocks.exe`。这就是训练时实际使用的打包游戏
版本，因此无需任何编译。

**方案 B —— 编辑器工程（需要 Unreal Engine 4.27，下载约 375 MB）。**
解压 `airsim-blocks2-project-win64.zip`，运行 `GenerateProjectFiles.bat`，打开
`Blocks.uproject`，按 Play。这是唯一能编辑场景的方式 —— 但请注意**多机隔离补丁是一处
blueprint 改动，并不在该压缩包里**（它位于 UE 资产内部），因此刚构建好的工程在训练前仍需
应用该补丁。

无论走哪条路，训练前**多机隔离补丁都是必做的**：十架无人机生成在同一点，原版 AirSim 会让它们
互相碰撞、相互遮挡。见 [`README.md`](README.zh-CN.md#多机隔离补丁必做)、
[`ISOLATION_PATCH.md`](ISOLATION_PATCH.zh-CN.md) 与
[`../../docs/REPRODUCING.md`](../../docs/REPRODUCING.zh-CN.md) §2。

## 发布产物

环境**不在** git 里。它从原始压缩包重新打包为两个精简产物，二者都挂在
[**`v0.1.0` release**](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0)
上（`Blocks2_AirSim170.zip` 本身从未发布，因为其中 2.59 GB 是构建缓存、0.89 GB 是编辑器
临时状态）：

| 产物 | 压缩包 | 解压后 | 内容 | SHA-256 |
|---|---:|---:|---|---|
| [**`airsim-blocks2-standalone-win64.zip`**](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip) | **219.4 MB** | 356.6 MB | 打包好的独立游戏：`Blocks.exe`、`Blocks-WindowsNoEditor.pak`、引擎第三方 DLL、UE4 前置组件安装器。共 25 个文件。解压即运行。 | `3c17b77c7c39b67576a4d800d8317402eb8a9819fce1e627b5f6b3e112371a8c` |
| [**`airsim-blocks2-project-win64.zip`**](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip) | **374.5 MB** | 1.13 GB | 可编辑的 UE4 工程：`Content/`、`Config/`、`Source/`、`Binaries/`、`Plugins/AirSim`（随工程附带的 1.7.0）、`Blocks.uproject`。`Intermediate/` 与 `Saved/` 已按**任意**层级移除。共 1556 个文件。 | `137ecca71eb48be32d29963d86c258685fc0536f64691fde1678b6be45a88a4d` |

两者合计 **594 MB**，而原始压缩包为 1.29 GB —— 省下的 870 MB 完全是构建缓存与编辑器临时
状态，可运行或可编辑的内容没有任何损失。

工程压缩包有意保留了约 640 MB 的内容，而**只运行**的用户并不需要它们：原生调试符号
（`.pdb`，422 MB）和随工程附带的静态库（`.lib`，216 MB）。保留它们是为了让插件能够**离线
重新构建并调试**，无需重新下载上游 AirSim。如果你只想打开场景，独立运行版是更好的下载选择。

> **重新打包后已验证。** 工程压缩包已解压并在 Unreal Engine 4.27 中打开 —— 场景正常加载并
> 运行，没有重定向提示，也无需重新构建，因为
> `Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll`（编译好的插件，4.77 MB）与随工程附带
> 的源码放在一起。上文的截图就是从那个构建里截取的。

请从**独立运行版**开始 —— 实验实际运行的就是它，而且不需要安装 Unreal Engine。只有需要编辑
场景时，才去下载工程压缩包。

> 多机隔离补丁是一处 **blueprint 改动**，并且**已包含在工程压缩包中** —— 打过补丁的资产随包
> 分发为 `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset`（同样的字节也在工程级的
> `Content/BP_FlyingPawn.uasset`），因此重新构建后补丁依然保留。请保持该文件原样；换回原版
> AirSim pawn 会破坏训练。同样的字节也已入库至
> [`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset)
> （sha256 `0f4f91c1…`）。

原始压缩包，仅供参考 —— 若想更严谨些，可以拿它校验重新打包的结果：
`Blocks2_AirSim170.zip`，压缩后 1.29 GB / 解压后 4.69 GB，
sha256 `d65942ea5a7a40fad82ae941f0ddef0c4ad14650010eb10eab079b86106fa7f9`。

## 版权与署名

`Blocks` 与 AirSim 插件版权归 Microsoft Research 所有，采用 MIT 许可 —— 见
[microsoft/AirSim](https://github.com/microsoft/AirSim)。障碍物布局、多机隔离 blueprint 补丁
以及传感器配置均为本工程的成果，并按本仓库的 [MIT 许可证](../../LICENSE) 发布。
