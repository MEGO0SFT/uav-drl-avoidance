# 基于深度强化学习的无人机自主避障

[**English**](README.md) | **简体中文**

本项目研究未知静态三维环境中的激光雷达导航。当前实现基于 AirSim、分块稀疏概率地图和课程学习 PPO，并包含额外的混合安全盾。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[模型清单](MODELS.zh-CN.md) · [系统架构](docs/ARCHITECTURE.zh-CN.md) · [实验运行](docs/REPRODUCING.zh-CN.md) · [长期计划](ROADMAP.zh-CN.md)

[数学原理与实现公式](docs/MATHEMATICS.zh-CN.md)：坐标与观测、BSHM 概率更新、奖励函数与课程、网络维度、安全盾优化。

## 项目状态与公开材料

AirSim 实现是第一阶段基线，历史参照名称为 origin_26-0364；main 用于后续维护与文档更新。

当前版本保留源码、场景资产、配置说明、历史汇总表和最终图表。保留结果属于历史报告，不代表已通过当前检出版本重新验证。

策略权重和完整仿真器单独分发，见[模型清单](MODELS.zh-CN.md)与[仿真器说明](sim/airsim/ENVIRONMENT.zh-CN.md)。下述 Isaac Lab、PX4 SITL 和 ROS 2 工作属于后续计划，当前尚未形成替代 AirSim 的实现。

## 研究任务与系统

任务是在先验未知的静态障碍环境中完成单机目标导航。策略输入为 LiDAR 派生的局部几何表示和任务／运动学向量，不包含预置的全局障碍地图或绝对位置观测；相对目标信息和仿真真值运动状态可用。

配置中的十架无人机用于并行采样。[隔离补丁](sim/airsim/ISOLATION_PATCH.zh-CN.md)关闭机间碰撞和激光遮挡，因此这里研究的是单机策略，并非多智能体协同导航。

| 组成部分 | 当前实现 |
|---|---|
| 仿真 | AirSim 1.7.0 / Unreal Engine 4.27；数据中心通过 RPC 连接仿真器，通过 UDP 连接训练器 |
| 建图 | BSHM：64 m 分块、0.25 m 体素、占据概率更新、时间维护与有界块存储 |
| 观测 | 18 × 36 航向系距离图像，加 8 维任务／运动学向量 |
| 策略 | CNN／MLP 双分支特征提取器与前馈 PPO |
| 动作 | 世界 NED 系速度增量，各轴范围 ±3.75 m/s；合成速度上限 7.5 m/s |
| 可选安全盾 | 近距斥力、自适应视野 VO、局部 RRT* 退路、策略混合与距离限速 |

    LiDAR → 世界系 BSHM → 航向系距离图像
                           ＋任务／运动学向量
                   → 双分支 PPO → 速度提议
                   → 可选安全盾 → AirSim 速度指令

地图提供时间上下文，最终策略不含 LSTM 或 GRU。观测朝向依据水平速度，而动作保留在世界坐标系。安全盾是启发式修正机制，当前实现没有给出严格的无碰撞保证。

### 场景

![训练场景](sim/airsim/figures/training_scene.png)

文档记录的训练场地约为 250 × 250 × 50 m，包含约 70 个障碍物 actor。布局存放在 [FlyingExampleMap.umap](sim/airsim/project/Content/FlyingCPP/Maps/FlyingExampleMap.umap) 中。仅克隆场景资产并不包含完整可运行仿真器及 AirSim 插件依赖。

## 历史结果

以下数值作为历史摘要保留。训练指标、确定性评估与泛化实验采用不同协议，不应将其解读为同一口径下连续递进的性能。

| 协议 | 历史报告值 |
|---|---|
| 课程 PPO，训练期滚动成功率 | 0.810 |
| 无课程 PPO，训练期滚动成功率 | 约 0.764 |
| SAC 对照，训练期滚动成功率 | 峰值 0.699，末值 0.373 |
| 未见走廊泛化实验 | 报告成功率 68.2% |
| PPO 名义训练预算 | 约 600 万环境步 |

当前训练脚本配置为十个环境、每环境 rollout 2560 步。该配置尚未与原始运行元数据完整对齐，不能视为已经核实的历史实验精确重建。

### 安全盾扩展

下表保留自 [ablation_summary.json](benchmarks/ablation_summary.json)。安全盾实验是仓库扩展，与期刊论文所述范围分开记录。

| 配置 | 回合数 | 成功率 | 历史失败字段* | 累积奖励 | 成功回合用时 |
|---|---:|---:|---:|---:|---:|
| 纯策略 | 309 | 90.94% | 9.06% | 245.80 ± 78.36 | 22.31 ± 4.26 s |
| ＋近距斥力 | 307 | 94.46% | 5.54% | 261.44 ± 81.76 | 24.38 ± 4.62 s |
| ＋自适应 VO | 305 | 93.11% | 6.89% | 258.29 ± 85.50 | 24.36 ± 4.47 s |
| ＋RRT* 退路 | 311 | 95.82% | 4.18% | 263.84 ± 71.64 | 24.65 ± 5.27 s |
| 完整系统 | 525 | 96.95% | 3.05% | 288.96 ± 68.27 | 24.70 ± 5.28 s |

*历史 JSON 的字段名为 collision_rate，但评估器按“100 减成功率”计算该值。若存在超时，它表示失败率而非单独的碰撞率。该消融逐层累加模块，不能独立识别每个模块的因果贡献。

![训练收敛图](benchmarks/figures/ppo_training_convergence.png)

![安全盾消融图](RL_scripts/body-axis_v2/scripts/figures/ablation_combined.svg)

![示例轨迹](RL_scripts/body-axis_v2/scripts/figures/trajectory_stages_shield_ep1.png)

以上均为保留的历史图表。示例轨迹只展示一个回合，不能替代总体统计。现有最终图片、SVG 与 PDF 面板全部保留。

## 代码导航

| 路径 | 用途 |
|---|---|
| [RL_scripts/body-axis_v2/scripts](RL_scripts/body-axis_v2/scripts/) | PPO 主实现、安全盾扩展与评估工具 |
| [BSHM.py](methods/AirSim_ReinforceLearning/BSHM.py) | 分块稀疏概率建图 |
| [实验目录说明](RL_scripts/README.zh-CN.md) | 历史实验目录与入口 |
| [sim/airsim](sim/airsim/) | 传感器配置、场景资产与隔离补丁 |
| [benchmarks](benchmarks/) | 保留的历史摘要与最终训练图 |
| [paper](paper/) | 引用元数据及论文／仓库范围说明 |
| [scripts](scripts/) | 仓库检查与本地 TensorBoard 导出工具 |

## 运行 AirSim 基线

提供的仿真器归档面向 Windows x64；训练环境导出面向 Linux／WSL2，记录 Python 3.11.14、PyTorch 2.6.0+cu124、Stable-Baselines3 2.7.0 和 airsim 1.7.0。[环境文件](environment.yml)是特定平台的记录，不是跨平台安装保证。

    conda env create -f environment.yml
    conda activate airsim-gym-ubuntu2404

运行前需要取得仿真器／权重，安装 [settings.json](sim/airsim/settings.json)，验证多机隔离，并按实际拓扑配置 RPC／UDP 地址及机体数量。入口示例：

    python RL_scripts/body-axis_v2/scripts/airsim_datamanager.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py

后两条命令分别对应课程组和无课程对照组，应择一运行。评估前请阅读[实验指南](docs/REPRODUCING.zh-CN.md)：默认评估入口关闭安全盾，部分辅助脚本仍引用历史模块名，成功率打印采用滚动窗口。新运行不保证精确复现历史摘要。

## 开发优先级与贡献方向

1. **当前首要任务是 Isaac Lab 移植。** 欢迎完善训练环境、PX4 SITL／ROS 2 通信链路、训练方法迭代与传感器二次设计，而不仅是替换仿真器 API。
2. **第 1 项完善后开展奖励与网络研究。** 当前架构性能有限，后续关注小型端到端策略的架构影响，继续挖掘 DRL 潜力，并探索视觉—语言—动作（VLA）、组合智能体等方法，拓展策略的应用范围与能力边界。
3. **Sim2Real。** 研究控制频率与执行机构能力的匹配、链路延迟、抗干扰和泛化，通过分阶段仿真及现实实验验证。
4. **其他载具形态。** 不局限于传统无人机结构，但这是长期研究方向，短期没有实施承诺。

详细计划与贡献要求见[长期计划](ROADMAP.zh-CN.md)和[贡献指南](CONTRIBUTING.zh-CN.md)。这些方向均不代表已经具备的能力，也不承诺确定的性能提升。

## 上游项目与文档

| 组件 | 原项目与文档 | 本项目中的用途 |
|---|---|---|
| Microsoft AirSim | [源码仓库](https://github.com/microsoft/AirSim) · [官方文档](https://microsoft.github.io/AirSim/) | 仿真、传感器与速度控制接口 |
| Stable-Baselines3 | [源码仓库](https://github.com/DLR-RM/stable-baselines3) · [官方文档](https://stable-baselines3.readthedocs.io/) | PPO 与训练基础设施 |
| SB3 Contrib | [源码仓库](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | 历史实验依赖；当前交付策略不是 RecurrentPPO |
| Gymnasium | [源码仓库](https://github.com/Farama-Foundation/Gymnasium) · [官方文档](https://gymnasium.farama.org/) | 环境接口与空间定义 |
| Open3D | [源码仓库](https://github.com/isl-org/Open3D) | 点云处理与可视化 |
| Numba | [项目主页](https://numba.pydata.org/) | 数值计算加速 |
| SciPy | [项目主页](https://scipy.org/) | 数值工具与安全盾线性规划 |

## 论文与署名

石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕. 基于渐进课程学习的无人机自主避障研究. 系统仿真学报, 2026。[DOI: 10.16182/j.issn1004731x.joss.26-0364](https://doi.org/10.16182/j.issn1004731x.joss.26-0364)。

论文范围与 BibTeX 见[发表元数据](paper/README.zh-CN.md)。第一阶段在中山大学航空航天学院开展。工程使用 [Microsoft AirSim](https://github.com/microsoft/AirSim)、[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)、[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)、[Open3D](https://github.com/isl-org/Open3D)、[Numba](https://numba.pydata.org/) 和 [SciPy](https://scipy.org/)；上游组件保留各自许可证。

[MIT](LICENSE) © 2025–2026 Shifutian（石辅天）。
