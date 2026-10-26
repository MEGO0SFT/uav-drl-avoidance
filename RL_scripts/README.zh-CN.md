# 实验目录说明

[**English**](README.md) | **简体中文**

[← README](../README.zh-CN.md) · [系统架构](../docs/ARCHITECTURE.zh-CN.md) · [历史运行标签](body-axis_v2/logs/README.zh-CN.md)

以下五个目录记录不同实现及评估辅助工作，不代表五代经过独立验证的算法。

| 目录 | 角色 | 观测坐标系／动作 |
|---|---|---|
| body-axis_v2 | 论文策略主实现及后续安全盾扩展 | 速度航向系距离图；世界系速度增量 |
| body-axis_v2_plot | 绘图、蒙特卡洛轨迹工具及相关历史代码 | 表示相近，需检查各自评估路径 |
| body-axis | 早期原型 | 姿态机体系表示，状态／动作约定不同 |
| world-axis-v2 | 早期原型 | 世界系对齐距离图 |
| airsim-only-with-lidar | 最早原型 | 更早的状态向量与四维动作 |

以 RecurrentPPO 开头的历史文件名不能证明使用了循环网络。最终策略是前馈 PPO 配合 CNN／MLP 提取器，地图累积提供时间上下文。

## 主要入口

body-axis_v2/scripts 中：

- PPO_SB3_AirSim_CL.py：奖励塑形课程组。
- PPO_SB3_AirSim.py：无课程对照组。
- PPO_SB3_AirSim_model.py：策略评估及可选安全盾。
- airsim_datamanager.py：RPC／UDP 数据中心。
- vehicle_manager.py：建图封装与观测构造。
- train_sac.py／train_ddpg.py：对照实现。
- ablation_experiment.py：可配置的累积安全盾评估。
- collision_diagnostics.py：新运行的诊断记录。

这些是相关源码实现，不能保证逐一对应每项历史结果当时使用的可执行代码及配置。部分辅助脚本仍保留旧导入名，当前限制见[实验指南](../docs/REPRODUCING.zh-CN.md)。

## 表示演进

不同版本探索了世界系、姿态系及速度航向系距离图。最终表示让前向方向在图中更容易定位，同时保留世界系动作。早期实验中的探索表现是该选择的动机，不能据此推导机体系动作普遍无法学习。

期刊论文覆盖建图、策略及课程工作；地图时间维护与混合安全盾属于额外仓库功能，见[论文范围](../paper/README.zh-CN.md)。

## 本地产物

训练在本地生成日志与 checkpoint，评估可生成轨迹数组和详细报告。这些文件保留在本地并已被 Git 忽略。

[历史运行元数据](body-axis_v2/logs/README.zh-CN.md)只是历史运行标签的文本索引。安全盾汇总 JSON 保留在 scripts/ablation_results/ 和 [benchmarks](../benchmarks/README.zh-CN.md)。现有最终图片、SVG 与 PDF 均保留。

旧原型用于记录设计背景，不保证无需适配即可在当前环境运行。
