# 更新日志

[**English**](CHANGELOG.md) | **简体中文**

[← 返回 README](README.zh-CN.md) · [路线图](ROADMAP.zh-CN.md)

本仓库所有值得注意的变更。格式遵循
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)；本项目使用
[Semantic Versioning](https://semver.org/spec/v2.0.0.html)。

---

## [0.1.0] — 2026-06-04 · *Stage 1 归档快照*

首个公开发布版本。对应分支 **`origin_26-0364`**，冻结于期刊文章被接收时的项目
状态。

### 新增

- **仿真。** 分布式并行 AirSim 平台：一个 `airsim_datamanager` 进程通过 UDP 为
  所有无人机提供服务（分片 pickle、`5812`/`5813` 加上每架无人机一个 `2000+id`
  控制端口），20 Hz 传感器/指令回路。多无人机 UE 蓝图补丁，使无人机之间既不互相
  碰撞、也不互相遮挡。
- **建图。** `methods/AirSim_ReinforceLearning/BSHM.py` —— 块稀疏概率哈希地图，
  64 m 分块 / 0.25 m 体素，贝叶斯 log-odds 更新，双层时间衰减（4 Hz 概率衰减
  `λ = 0.015`、8 s 陈旧数据驱逐、LRU 块上限 2048）。
- **策略。** 双分支特征提取器（`OctomapCNN`）+ 带独立 actor/critic 主干的 PPO；
  656 维观测（18×36 航向系距离图像 + 8 维任务向量）；三维速度增量动作。
- **奖励。** 五个稠密项（方向加权障碍、多预算平滑、门控前向、进度、生存）加上
  非对称稀疏终止项，其中紧迫度系数 `k_avoid` 把障碍接近度与前向激励耦合起来。
- **课程。** `PPO_SB3_AirSim_CL.py` —— 由训练进度回调驱动的渐进式奖励塑形，与
  对照组 `PPO_SB3_AirSim.py` 一一配对（两者只差两行）。
- **安全盾。** `PPO_SB3_AirSim_model.py` —— 近距斥力、自适应视野速度障碍
  （LP 投影）、2 Hz 的 RRT\*-Safe、分布偏移防护、目标接近度门限、基于距离的
  速度调制。
- **评估。** 蒙特卡洛驱动程序，给出成功率、碰撞率、回报与回合时长；
  `ablation_experiment.py` 用于 5 配置的安全盾消融；`collision_diagnostics.py`
  用于自动失败归因。
- **异策略对照。** SAC 与 DDPG 变体，共享环境、奖励、网络拓扑与课程。
- **数据。** 51 条训练运行 —— 原始 TensorBoard 事件文件已入库（50 MB），
  另有 `benchmarks/tb_curves/` 中的抽稀 CSV 曲线；消融摘要和逐 step 失败报告在
  `benchmarks/` 中。
- **工具。** `scripts/export_tb_curves.py`（无依赖的 TensorBoard 标量读取器）、
  `scripts/repo_hygiene.py`（超体积/二进制防护）。
- **文档。** `README.md`（+ 带语言切换的 `README.zh-CN.md` 中文镜像）、
  `docs/ARCHITECTURE.md`、`docs/REPRODUCING.md`、`benchmarks/README.md`、`MODELS.md`、
  `ROADMAP.md`、`paper/README.md`、`CITATION.cff`。
- **环境。** `environment.yml` —— 产出这些结果的那台机器上精确的 conda 导出：
  **WSL2 · Ubuntu 24.04 · RTX 4070 SUPER · CUDA driver 12.0**，Python 3.11.14、
  `airsim 1.7.0`、`stable-baselines3 2.7.0`、`torch 2.6.0+cu124`、`numpy 2.0.1`。
  `requirements-legacy.txt` 锁定到相同版本，作为仅用 pip 的兜底方案。
- **仿真器配置。** `sim/airsim/settings.json` —— 产出结果所用的精确 AirSim 配置：
  **10** 架 `SimpleFlight` 无人机 `UAV01…UAV10`，以及真实 LiDAR（**32 线、6 Hz、
  8000 pts/s、80 m 量程**，垂直 `±30°`，水平 `360°`，`SensorLocalFrame`），再加上
  早期相机研究留下的 `CameraDefaults`。`sim/airsim/README.md` 解释了每项设置意味着
  什么 —— 包括为什么十架无人机生成在同一点使得多机隔离补丁成为必需，以及解释了
  斜曲面失效模式的粗糙（约 8.6°）水平采样。这**取代 `methods/Sensors1.json`**，
  后者记录的是 16 线 / 25 m / 5 Hz 的传感器，且从未与已发表的运行相匹配。
- **训练环境。** `sim/airsim/project/`（13.6 MB）—— **场景本身**：`Blocks2_AirSim170`
  UE4 工程中的 `FlyingExampleMap.umap`（12.3 MB，含约 70 个障碍物 actor 的
  250 × 250 × 50 m 场地）、它实例化的网格资产、`Blocks.uproject` 和 `Config/`。
  `sim/airsim/figures/training_scene.png` 是环境启动时的视图。
  `sim/airsim/ENVIRONMENT.md` 记录了整个环境：AirSim 插件 **1.7.0**、**Unreal
  Engine 4.27**（与 `Blocks.uproject` 中的 `EngineAssociation` 一致）、运行它的
  两种方式（独立 staged 构建 vs. 编辑器工程）、4.69 GB 解包归档包含什么、为什么
  只提交场景，以及完整归档的 SHA-256，以便校验任何一次下载。
- **仿真器的 Release 产物。** 1.29 GB 的环境归档被重新打包成两个精简归档，两者都
  附在带校验和的 `v0.1.0` Release 上：`airsim-blocks2-standalone-win64.zip`
  （**219 MB**，打包好的游戏 —— 无需安装 Unreal Engine 即可运行）和
  `airsim-blocks2-project-win64.zip`（**375 MB**，可编辑工程）。任意嵌套深度下的
  所有 `Intermediate/` 和 `Saved/` 内容都被删除 —— 870 MB 的构建缓存和编辑器临时
  文件 —— 没有丢失任何可运行或可编辑的内容。嵌套的情况很重要：仅
  `Plugins/AirSim/Intermediate/` 就有 78 MB 的编译对象。
- **部署接口文档。** IP 配置现在按交付的代码逐项写明：一张表列出三个不同的地址
  （`airsim_datamanager.py` 中的 `airsim_ip` 和 `trainmanager_ip`，以及 `make_env()`
  中的 `datamanager_ip`/`trainmanager_ip`）、它们所在的文件与行号、当前取值，以及
  它们所暗示的双机拓扑。此外还有 `5812`/`5813`/`2000+index` 端口映射、无人机数量
  必须一致的三个位置，以及 WSL2 NAT 与防火墙的坑。
- **参考链接。** 一张 "Built on / related repositories" 表，链接 AirSim、
  Stable-Baselines3、SB3-Contrib、Open3D、Numba、SciPy 与 Gymnasium，并附各自
  文档站点。

### 变更

- **运行名称现在说明实验内容。** SB3 自动生成的 TensorBoard 目录
  （`PPO_98`、`PPO_99`、`SAC_3`、…）被重命名为 `date_role-budget`，曲线 CSV 沿用
  同一命名方案，因此曲线和它的运行可以一眼对上：
  `2026-03-18_paper-curriculum-6M`、`2026-04-13_paper-control-6M`、
  `2026-04-27_paper-sac-6M`（三次论文运行），加上 `2026-03-11_pilot-5M`、
  `2026-02-07_pilot-5M`、`2025-12-23_pilot-4M`。十三次中止/探测运行（每个小于
  0.5 MB）以原名停放在 `logs/_short-runs/` 下，`logs/README.md` 为它们逐一建立
  索引，包括注明 `PPO_94` 是课程运行被打断的首次尝试，而非一次独立实验。
  `models/` 中的 checkpoint 目录已重命名以保持一致；`scripts/export_tb_curves.py`、
  `benchmarks/README.md` 和 `MODELS.md` 相应更新。
- `RecurrentPPO_SB3_AirSim.py` → **`PPO_SB3_AirSim.py`**：该策略是普通的前馈
  PPO；旧名字是早期实验的遗留。`body-axis/`、`world-axis-v2/` 和
  `airsim-only-with-lidar/` 中的旧原型保留其历史名称，以便已发表结果仍可追溯到
  它们的训练脚本。
- 从快照中剔除 119 个中间 checkpoint（6.0 GB）。它们被离线保存，且从来不是复现
  任何已发表数字所必需的；见 `MODELS.md`。
- 在 `PPO_SB3_AirSim.py` / `PPO_SB3_AirSim_CL.py` 中恢复了已报告实验实际使用的
  PPO 配置：**每个环境 `n_steps=2560`，共 10 个并行环境**（25 600 步的 rollout）。
  SAC/DDPG 对照已对齐到同样的 10 环境机队。

### 修复

- 将训练日志提交入库，而不是忽略它们 —— 它们是论文中每条曲线背后的首要证据。

### 已知局限

- 仅静态障碍；没有动态障碍实验。
- 假定仿真状态测量无噪声（已记录的假设）。
- 在未见过走廊场景上的泛化下降到 68.2 %。
- 观测坐标系（速度航向）与动作坐标系（世界系）按设计不同；见
  `docs/ARCHITECTURE.md` §2.2。
- 未做 sim-to-real。未做硬件在环。二者分别是 `ROADMAP.md` 中的 Stage 2/3。

---

## [Unreleased]

### 新增

- **每份文档都有了中文版。** 现在十五份文档都是成对的 —— `X.md` 与 `X.zh-CN.md` 并存 ——
  标题下方各有一条语言切换行，中文页面里的文档互链指向对应的中文版，而不是把读者弹回英文页。
  这三条约定由 `python scripts/repo_hygiene.py --all --docs` 和 GitHub Actions 工作流
  （`.github/workflows/docs.yml`）强制检查：中文版存在、切换行指向它、且没有失效的相对链接。
- **隔离补丁现在可以被识别了。** 它是单个蓝图资产
  `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset`，其中无人机网格的碰撞通道
  被关闭，LiDAR 射线投射跳过 `Pawn` 对象，因此光束可以穿过无人机。它随两个 Release
  归档一起发布，其字节提交在 `sim/airsim/project/Content/BP_FlyingPawn.uasset`
  （sha256 `0f4f91c1…`）。此前文档只写了"两处 Unreal 蓝图修改"，让读者无法定位或
  验证它。
- **直达下载链接**，指向已发布的 `v0.1.0` Release，读者不必再四处寻找二进制文件：
  两个 README 中的 Release 徽章和下载指引、[`MODELS.zh-CN.md`](MODELS.zh-CN.md) 中的逐资产
  链接，以及 [`sim/airsim/ENVIRONMENT.zh-CN.md`](sim/airsim/ENVIRONMENT.zh-CN.md) 中两个
  仿真器归档的逐资产链接。
- `MODELS.md` 记录 `SFT_avoid_modelv3.zip` 与课程运行的 `final_model` checkpoint
  **字节完全相同**（sha256 `f1fbdf2a…`）—— 两者只需下载一个。

### 修复

- `sim/airsim/ENVIRONMENT.md` 声称隔离补丁在从工程归档重建后*不会*保留。这是错的：
  打过补丁的 `BP_FlyingPawn.uasset` 就在工程归档内部，因此重建会保留它。

### 计划

- Isaac Lab 移植（GPU 并行，512–1024 环境）—— `ROADMAP.md` Stage 2。
- PX4 SITL 软件在环；策略 → `OFFBOARD` 速度设定点 ——
  Stage 3。
- 安全层抽离为独立包，并带一个 ROS 2 节点 —— Stage 4。
- 观测/动作坐标系统一，配合动作尺度课程 —— Stage 2。
