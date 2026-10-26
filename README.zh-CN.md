# 未知环境下无人机自主避障（深度强化学习）

[**English**](README.md) | **简体中文**

**一套完整、可复现的无人机三维自主避障研究工程：分布式并行 AirSim 仿真平台 → 块稀疏概率地图 → 课程学习训练的避障策略 → 多策略混合安全决策框架。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/MEGO0SFT/uav-drl-avoidance?label=download&color=green)](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0)
[![Stage](https://img.shields.io/badge/stage-legacy%20AirSim%20%C2%B7%20archived-blue)]()
[![Next](https://img.shields.io/badge/next-Isaac%20Lab%20%2B%20PX4%20SITL-orange)](ROADMAP.md)

> **状态。** 分支 **`origin_26-0364`** 是论文录用时项目的冻结快照，作为本文的归档参照。
> `main` 继续向前演进，正在迁移到 **NVIDIA Isaac Lab + PX4 SITL**，见 [`ROADMAP.md`](ROADMAP.md)。
>
> **仓库里有什么。** 代码、文档、消融数据表、全部 51 条训练记录（原始 TensorBoard
> 事件文件，50 MB）以及抽稀后的曲线 CSV。策略权重**不在** git 里——单文件超过 GitHub
> 体积上限。
>
> 📦 **仿真器与训练好的策略：[下载 v0.1.0 Release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0)** ——
> 含免装 Unreal 即可运行的打包仿真器，以及 PPO/SAC 权重。清单见 [`MODELS.md`](MODELS.md)，
> 仿真器说明见 [`sim/airsim/ENVIRONMENT.md`](sim/airsim/ENVIRONMENT.md)。

---

## 路线图快照

| | |
|---|---|
| ✅ | Stage 1 — AirSim + SB3：分布式并行仿真、BSHM 地图、课程学习 PPO 策略、五层安全盾、完整评估与消融 |
| 🚧 | Stage 2 — 迁移 Isaac Lab（GPU 并行强化学习，512–1024 环境） |
| 🚧 | Stage 3 — PX4 SITL 软件在环（MAVLink HIL），策略输出接 OFFBOARD 速度设定点 |
| 📄 | 论文已发表于《系统仿真学报》（基于渐进课程学习的无人机自主避障研究） |

---

## 这个项目在做什么

四旋翼需要在**完全未知、障碍密集的三维环境**中穿越 100–150 m 并抵达从未测绘过的目标点，
**只依靠一台机械旋转式激光雷达和自身的运动状态**：没有全局地图、没有先验知识、观测中也不包含
特权目标坐标。

系统分为四层：

| # | 层次 | 职责 |
|---|---|---|
| 1 | **仿真** | AirSim (UE4) 分布式并行平台；自建 UDP 数据交互中心，20 Hz 收发传感器与控制；多机隔离补丁（关闭机间碰撞与激光雷达遮挡） |
| 2 | **感知** | 激光点云 → 分块稀疏哈希地图 BSHM：64 m 分块、0.25 m 体素、贝叶斯占据概率、双层时间衰减（概率衰减 + LRU 驱逐）→ 18 × 36「航向系」距离栅格观测 |
| 3 | **策略** | 双分支特征提取器 DBE —— 地图走 CNN、任务向量走 MLP —— 汇入独立主干的 PPO Actor/Critic（ELU）。输出：速度增量 Δv ∈ ±3.75 m/s。训练：渐进式奖励塑形课程学习 |
| 4 | **安全盾** | 近距斥力场 → 自适应视野 VO-τ → RRT\*-Safe → 分布偏移防护 → 基于距离的速率调制。用确定性规则校验并限幅学习策略的输出 |

端到端数据流：

```
LiDAR -> BSHM -> 18x36 range image  +  8-D task vector
      -> DBE -> PPO -> delta-v -> safety shield -> velocity PID -> AirSim
```

### 训练场景

<img src="sim/airsim/figures/training_scene.png" width="820">

四面围墙的 **250 × 250 × 50 m** 场地，内含约 **70 个障碍物 actor**，全部由 Unreal 基础几何体构成：
三个立方体网格 + 引擎自带的圆柱／圆锥／球形体实例化摆放。包含楼板与立柱（占比最高、"类建筑"的
多数）、圆柱、圆锥与棱椎、球体，以及非规则组合体（画面前景那排带棱纹的圆柱）。
障碍物集中在场地中部，四周沿围墙留出开阔带；起始点与目标点在论文所述的可行区间内随机采样。

场景本体已入库：[`sim/airsim/project/Content/FlyingCPP/Maps/FlyingExampleMap.umap`](sim/airsim/project/Content/FlyingCPP/Maps/FlyingExampleMap.umap)；
UE4 工程、打包好的独立运行版及获取方式见 [`sim/airsim/ENVIRONMENT.md`](sim/airsim/ENVIRONMENT.md)。

---

## 结果

### 混合安全盾消融
训练场景内的蒙特卡洛评估；策略权重完全相同，逐层叠加开启安全盾。

| 配置 | 幕数 | 成功率 | 碰撞率 | 累积奖励 | 成功用时 |
|---|---:|---:|---:|---:|---:|
| 纯 DRL 策略 | 309 | **90.94 %** | 9.06 % | 245.80 ± 78.36 | 22.31 ± 4.26 s |
| + 近距斥力场 | 307 | 94.46 % | 5.54 % | 261.44 ± 81.76 | 24.38 ± 4.62 s |
| + 自适应 VO-τ | 305 | 93.11 % | 6.89 % | 258.29 ± 85.50 | 24.36 ± 4.47 s |
| + RRT\*-Safe | 311 | 95.82 % | 4.18 % | 263.84 ± 71.64 | 24.65 ± 5.27 s |
| **完整系统** | 525 | **96.95 %** | 3.05 % | **288.96 ± 68.27** | 24.70 ± 5.28 s |

原始数据：[`benchmarks/ablation_summary.json`](benchmarks/ablation_summary.json)

### 策略训练与泛化
| 指标 | 数值 |
|---|---|
| 训练预算 | 600 万环境步，10 架无人机并行，约 48 h |
| 训练内成功率（最近 100 幕） | 0.81 |
| 课程组 vs. 无课程对照组（同为 600 万步） | 0.810 vs. 0.764 |
| 泛化 —— 未见过 150 × 50 × 50 m 通道场景，200 幕 | **68.2 %** |
| 离线策略对照（SAC，奖励/课程/网络完全一致） | 峰值 0.699，末值跌至 0.373 |

### 训练收敛过程

<img src="benchmarks/figures/ppo_training_convergence.png" width="580">

课程组在完整 600 万步上的五项指标（曲线平滑系数 0.9）：

| 面板 | 反映的内容 |
|---|---|
| **每幕平均奖励** | 冷启动 −120，前约 50 万步内升至 +50，之后稳步爬升至 ≈ 200 |
| **成功率**（最近 100 幕） | 0 → **0.81**；最陡的上升段在 100–200 万步之间 |
| **近似 KL 散度** | 学习期升至 ≈ 0.03，随后在 0.02 附近波动，600 万步时降至 ≈ 0——更新幅度随策略收敛而收缩 |
| **动作采样标准差** | 1.0 → 4.0，约 400 万步后饱和：探索**没有**被衰减掉，随机策略稳定在一定的采样宽度 |
| **每幕平均步长** | 早期乱撞时峰值约 340，最终稳定在 ≈ 110——直接说明最终策略是高效飞行而非游荡 |

各指标的**矢量单图**见 [`benchmarks/figures/`](benchmarks/figures/)；逐指标原始数据见
[`benchmarks/tb_curves/2026-03-18_paper-curriculum-6M.csv`](benchmarks/tb_curves/2026-03-18_paper-curriculum-6M.csv)。

### MDP 一览
| | |
|---|---|
| 观测 | 656 维 = 648 维（18 仰角 × 36 方位距离图，旋转到**航向系**）+ 8 维任务向量（目标方向、进度、速度方向、速率） |
| 动作 | 3 维连续速度增量 Δv ∈ [−3.75, 3.75]³ m/s，\|v\| 限幅 7.5 m/s |
| 奖励 | 5 个稠密项（避障、平滑、前进、进度、生存）+ 稀疏 ±50/−80；三项权重随训练进度调度 |
| 控制 | 速度跟踪 PID 级联（离线速度设定点） |

公式、代码引用与设计动机见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### 成果图

<img src="RL_scripts/body-axis_v2/scripts/figures/ablation_combined.svg" width="880">

**安全盾消融实验（A–E 组）。**
A = 纯策略 · B = +近距斥力场 · C = +自适应 VO-τ · D = +RRT\*-Safe · E = 完整系统。
左：成功 / 失败率；中：平均累积奖励；右：成功回合平均耗时。

<img src="RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_shield_ep1.png" width="640">

**单幕飞行过程（任务进度 25 % / 50 % / 75 % / 100 %），按决策层着色。**
每一段航迹按"由哪一层产生该步指令"上色，每个面板同时给出地图规模、已飞行距离与各层占比。
可以看出**安全盾是随障碍密度渐进介入**的：任务进度 25 % 时几乎全由策略自己飞（**98 %**，
速度调制仅 2 %，已飞 40 m，BSHM 31 639 体素）；到 75 % 阶段速度调制升至 **51 %**、近距斥力场 11 %、
VO-τ 3 %；抵达目标时 44 % / 9 %（已飞 173 m，BSHM 29 620 体素）。
这正是混合架构设计的意图——开阔段交给学习策略，密集区交给确定性规则层。

<details>
<summary>单面板与高分辨率文件</summary>

| 图 | 链接 |
|---|---|
| 成功 / 失败率 | [`ablation_a_success.svg`](RL_scripts/body-axis_v2/scripts/figures/ablation_a_success.svg) |
| 平均累积奖励 | [`ablation_b_reward.svg`](RL_scripts/body-axis_v2/scripts/figures/ablation_b_reward.svg) |
| 成功回合平均耗时 | [`ablation_c_time.svg`](RL_scripts/body-axis_v2/scripts/figures/ablation_c_time.svg) |
| 合并版（含各组名称） | [`ablation_results.svg`](RL_scripts/body-axis_v2/scripts/figures/ablation_results.svg) · [`.pdf`](RL_scripts/body-axis_v2/scripts/figures/ablation_results.pdf) |
| 任务进度分阶段图（无着色，4829 × 5126 px） | [`trajectory_stages_ep1.png`](RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_ep1.png) · [`.pdf`](RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_ep1.pdf) · [`.svg`](RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_ep1.svg) |
| 安全盾轨迹俯视图 | [`shield_trajectory_ep1_topview.svg`](RL_scripts/body-axis_v2/scripts/figures/shield_trajectory_ep1_topview.svg) |
| 安全盾轨迹三维视图 | [`shield_trajectory_ep1.svg`](RL_scripts/body-axis_v2/scripts/figures/shield_trajectory_ep1.svg)（2.1 MB） |

</details>

> 这些图里的坐标轴标注为中文——它们是投稿时使用的原始面板。
> 同目录下另存有矢量（`.svg`）与印刷用（`.pdf`）版本，可直接用于排版。

---

## 仓库结构

```
.
├── environment.yml                   # ★ 真实运行环境（conda，WSL2 / Ubuntu 24.04）
├── sim/
│   └── airsim/
│       ├── settings.json             # ★ 出结果时用的 AirSim 配置（10 机 + 激光雷达）
│       ├── README.md                 # 各参数的含义；为什么必须做多机隔离
│       ├── ENVIRONMENT.md            # ★ UE4 工程说明：Blocks2_AirSim170、AirSim 1.7.0、场景
│       ├── figures/
│       │   └── training_scene.png    # 训练场景实拍
│       └── project/                  # 训练场景本体（13.6 MB）：.umap、网格、工程配置
├── methods/                          # 可复用的感知与解算模块
│   ├── AirSim_ReinforceLearning/
│   │   ├── BSHM.py                   # ★ 分块稀疏哈希地图（论文的地图）
│   │   ├── LidarProcess.py           # 点云解析、机体→全局变换、降采样
│   │   ├── OctoMap_o3d.py            # 早期八叉树实现（保留作参照）
│   │   └── ...
│   ├── Sensors1.json                 # 旧版传感器模板 —— 已被 sim/airsim/ 取代
│   └── airsim_lidar.py
│
├── RL_scripts/
│   ├── body-axis_v2/                 # ★ Stage-1 最终版（策略 + 安全盾）
│   │   ├── scripts/
│   │   │   ├── PPO_SB3_AirSim.py             # 策略训练入口（无课程对照组）
│   │   │   ├── PPO_SB3_AirSim_CL.py          # 课程学习组（与上一份仅差 2 行）
│   │   │   ├── PPO_SB3_AirSim_model.py       # 完整系统含安全盾 + 评估入口
│   │   │   ├── vehicle_manager.py            # 观测构造、BSHM 封装
│   │   │   ├── airsim_datamanager*.py        # UDP 数据交互中心
│   │   │   ├── ablation_experiment.py        # 五组安全盾消融
│   │   │   ├── collision_diagnostics.py      # 失败归因
│   │   │   ├── train_sac.py / train_ddpg.py  # 离线策略对照
│   │   │   ├── trajectory_visualizer.py, ... # 论文插图
│   │   │   └── figures/                      # 生成的图
│   │   ├── results/  models/  logs/          # 权重（忽略）/ 日志（已入库）
│   │   └── ...
│   ├── body-axis_v2_plot/            # 绘图 / 蒙特卡洛轨迹导出
│   ├── body-axis/                    # 早期原型（机体系动作）
│   ├── world-axis-v2/                # 早期原型（世界系地图 + W_dir 初版）
│   └── airsim-only-with-lidar/       # 最早原型
│
├── benchmarks/                       # 机器可读结果
│   ├── ablation_summary.json
│   ├── failure_analysis_sample.txt
│   ├── figures/                      # 训练收敛面板（PNG + 5 张矢量单图）
│   └── tb_curves/*.csv               # 抽稀后的训练曲线（6 条 run）
├── docs/                             # ARCHITECTURE.md、REPRODUCING.md
├── paper/                            # 论文元数据、DOI、章节→代码对照
├── scripts/
│   ├── export_tb_curves.py           # 无依赖的 TensorBoard 标量 → CSV
│   └── repo_hygiene.py               # 体积/二进制入库守卫
├── MODELS.md                         # 各 checkpoint 是什么、怎么取
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md / README.zh-CN.md
├── ROADMAP.md
└── requirements-legacy.txt
```

> ⚠️ **关于命名。** 只有 `body-axis_v2/scripts` 改名为 `PPO_SB3_AirSim*.py`。
> `body-axis/`、`world-axis-v2/`、`airsim-only-with-lidar/` 以及 `body-axis_v2_plot`
> 里的副本仍保留历史上的 `RecurrentPPO_*` 文件名，以保证论文里的数字能追溯到产生它的脚本。
> 所有这些策略都是前馈 `PPO("MlpPolicy")` + 自定义 `OctomapCNN`——**本项目没有 LSTM/GRU**。

---

## 运行环境

原实验环境（`environment.yml` 即该环境的原始导出）：

| 项 | 值 |
|---|---|
| 系统 | **WSL2 · Ubuntu 24.04**（仿真器、数据交互中心、训练脚本全部在同一 WSL2 实例内运行） |
| GPU | NVIDIA **RTX 4070 SUPER** |
| CUDA | 驱动 **12.0**；PyTorch 使用 `2.6.0+cu124` wheel |
| Python | 3.11.14（miniforge / conda） |
| 关键依赖 | `airsim 1.7.0`、`stable-baselines3 2.7.0`、`sb3-contrib 2.7.1`、`torch 2.6.0`、`numpy 2.0.1`、`numba 0.62.1`、`open3d 0.19.0`、`gymnasium 1.2.2` |

```bash
conda env create -f environment.yml
conda activate airsim-gym-ubuntu2404
```

不使用 conda 时可参考 `requirements-legacy.txt`（注意 `airsim` 必须是 **1.7.0**，且 numpy 必须是 2.x）。
WSL2 内的 GPU 直通需在 Windows 侧装好支持 WSL 的显卡驱动，装好后在 WSL2 里 `nvidia-smi` 应能看到显卡。

---

## 快速开始

```bash
# 1. 启动 AirSim 场景。二选一：(a) 用仓库里 sim/airsim/project/ 的工程（.umap + 配置，需 UE 4.27），
#    或 (b) 按 sim/airsim/ENVIRONMENT.md 运行已打包的独立构建（无需安装 Unreal）
#    先把 sim/airsim/settings.json 装为 AirSim 的 settings.json（10 机 + 激光雷达）
#    多机隔离补丁必须打上（见 docs/REPRODUCING.md §2）

# 2. 启动数据交互中心 —— 一个进程服务所有无人机
python RL_scripts/body-axis_v2/scripts/airsim_datamanager.py

# 3. 训练（先按下面的「IP 配置」改好地址）。两组实验只差 2 行：
python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py    # 课程学习组
python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py       # 无课程对照组

# 4. 评估完整系统（策略 + 安全盾）
python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py

# 5. 复现消融表
python RL_scripts/body-axis_v2/scripts/ablation_experiment.py

# 6. 不重新训练也能重画训练曲线
python scripts/export_tb_curves.py RL_scripts/body-axis_v2/logs --out benchmarks/tb_curves
```

### ⚠️ IP 配置（最容易卡住的地方）

**下面这些地址都是仓库里写死的，部署到自己的机器前必须全部替换**，否则训练脚本会一直阻塞在等待观测。
一共有**三个**不同地址，注意它们分散在两个文件里，很容易混：

| 字段 | 出现位置 | 仓库中的原值 | 必须指向 |
|---|---|---|---|
| `airsim_ip` | `airsim_datamanager.py:380` | `172.21.224.1` | 运行 **AirSim** 仿真器的主机（RPC 端口 `41451`） |
| `trainmanager_ip` | `airsim_datamanager.py:380` | `172.21.236.121` | 运行**训练脚本**、接收状态回传的主机 |
| `datamanager_ip` / `trainmanager_ip` | `PPO_SB3_AirSim*.py`、`train_sac.py`、`train_ddpg.py`、`PPO_SB3_AirSim_model.py` 里的 `make_env()` | `172.21.236.121` | 运行 `airsim_datamanager.py` 的主机（这两个字段同机，填相同地址） |

也就是说，原实验是**两台机器**：AirSim 在 `172.21.224.1`，数据交互中心与训练脚本同在 `172.21.236.121`。
单机部署也可以，那时三个字段填同一个地址即可。

```python
# PPO_SB3_AirSim.py —— 把这两个地址换成你自己的
def make_env(uavname):
    def _init():
        env = AirSimDroneEnv(uavname=uavname,
                             datamanager_ip='172.21.236.121',   # ← 你的数据中心主机
                             trainmanager_ip='172.21.236.121',  # ← 你的训练主机
                             start_point=np.array([0, 0, -0.5], dtype=np.float64))
        env = Monitor(env)  # 添加监控
        return env
    return _init
```

```python
# airsim_datamanager.py:380 —— 两个地址都要换成你自己的
ADM = AirSim_Data_Manager(airsim_ip='172.21.224.1',          # ← 你的 AirSim 主机
                          uav_number=10,
                          trainmanager_ip='172.21.236.121')  # ← 你的训练主机
```

要点：

1. **查自己的地址**：WSL2 内用 `ip -4 addr show eth0`；Windows 侧用 `ipconfig`。
   不要填 `127.0.0.1` 了事——UDP 端口绑定在指定网卡上，绑定到回环会导致跨进程/跨机不通。
2. **端口必须对得上**：数据交互中心绑定 `5812`（状态上行）、向 `5813` 下发指令；
   训练侧每架无人机绑定 `2000 + drone_index`（`UAV01`→2000，`UAV02`→2001 …）。
3. **无人机数量必须在三处一致**：`make_env()` 里的 `range(10)`、`airsim_datamanager.py` 里的
   `uav_number=10`、以及 AirSim 实际启动的机体数——本项目为 **10**。
   （`airsim_datamanager_easy.py` 里目前仍是早期实验的 `uav_number=8`。）
4. **WSL2 特有坑**：WSL2 默认使用 NAT 虚拟网卡，Windows 宿主访问 WSL2 需要端口转发
   （`netsh interface portproxy`）；若仿真器也跑在同一个 WSL2 实例里，请确保各方绑定的是
   **同一个网络接口**，否则会出现"能收不能发"的单向通信。
5. **防火墙**：Windows Defender 防火墙默认会拦下新监听的 UDP 端口，首次运行需放行。

更完整的拓扑、AirSim 配置与踩坑清单见 [`docs/REPRODUCING.md`](docs/REPRODUCING.md)。

---

## 依赖项目与参考

本项目直接建立在以下开源项目之上，感谢它们的作者：

| 项目 | 在本项目中的作用 | 仓库 |
|---|---|---|
| **Microsoft AirSim** | UE4 高保真无人机仿真器、激光雷达与运动学接口 | <https://github.com/microsoft/AirSim> |
| **Stable-Baselines3** | PPO / SAC / DDPG 实现与训练回调体系 | <https://github.com/DLR-RM/stable-baselines3> |
| **SB3 Contrib** | 提供 `RecurrentPPO`（历史文件名的来源）；本项目最终未使用 | <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib> |
| **Open3D** | 点云与八叉树结构、可视化 | <https://github.com/isl-org/Open3D> |
| **Numba** | 体素降采样与体素键计算加速 | <https://numba.pydata.org/> |
| **SciPy** | VO-τ 的线性规划求解（`scipy.optimize.linprog`，HiGHS） | <https://scipy.org/> |
| **Gymnasium** | 环境接口规范 | <https://github.com/Farama-Foundation/Gymnasium> |

文档：AirSim 官方文档 <https://microsoft.github.io/AirSim/> ·
Stable-Baselines3 官方文档 <https://stable-baselines3.readthedocs.io/> ·
Gymnasium 官方文档 <https://gymnasium.farama.org/>

---

## 论文

> 石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕. 基于渐进课程学习的无人机自主避障研究.
> *系统仿真学报* (*Journal of System Simulation*), 2026.
> DOI: **[10.16182/j.issn1004731x.joss.26-0364](https://doi.org/10.16182/j.issn1004731x.joss.26-0364)**

元数据与 BibTeX：[`paper/README.md`](paper/README.md) · [`CITATION.cff`](CITATION.cff)

---

## 参与贡献

欢迎提 Issue 和 PR，见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
目前最有价值的贡献是**可移植性修复**（Linux/WSL 路径、AirSim 版本钉死）与**评测脚本**改进，
这两项都能直接帮助 Stage 2 的 Isaac Lab 迁移。

## 许可证

[MIT](LICENSE) © 2025-2026 Shifutian（石辅天）
