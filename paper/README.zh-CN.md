# 论文发表元数据

[**English**](README.md) | **简体中文**

[← 返回 README](../README.zh-CN.md) · [模型产物](../MODELS.zh-CN.md) · [引用元数据](../CITATION.cff)

## 期刊论文（主要引用）

> 石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕.
> **基于渐进课程学习的无人机自主避障研究**.
> *系统仿真学报* (*Journal of System Simulation*), 2026.

| | |
|---|---|
| DOI | [`10.16182/j.issn1004731x.joss.26-0364`](https://doi.org/10.16182/j.issn1004731x.joss.26-0364) |
| 稿件编号 | joss.26-0364 → 本仓库归档分支名 `origin_26-0364` |

```bibtex
@article{shi2026uavcl,
  title   = {基于渐进课程学习的无人机自主避障研究},
  author  = {石辅天 and 李嘉轩 and 汪孝迪 and 邓中行 and 史格非 and 杨雪榕},
  journal = {系统仿真学报 (Journal of System Simulation)},
  year    = {2026},
  doi     = {10.16182/j.issn1004731x.joss.26-0364},
  url     = {https://doi.org/10.16182/j.issn1004731x.joss.26-0364}
}
```

## 范围说明

期刊论文覆盖仿真器、BSHM 地图、观测 / 奖励设计，以及经课程学习训练出的策略。
本仓库还有两个组成部分**不**属于论文内容，只在此处的
[`../docs/ARCHITECTURE.zh-CN.md`](../docs/ARCHITECTURE.zh-CN.md) 中记载：
地图的双层时间衰减，以及混合安全盾及其消融实验。

## 论文各部分在本仓库中的对应位置

| 阶段 | 代码 |
|---|---|
| §1.1 分布式并行仿真器 / 数据交互中心 | `RL_scripts/body-axis_v2/scripts/airsim_datamanager*.py` |
| §1.2 传感器与假设 | `methods/Sensors1.json`, `methods/airsim_lidar.py` |
| §2.1–2.2 状态 / 动作空间 | `vehicle_manager.py` (`APF_SeqCalculate3D`, `_Lidar_RLstat_update`) |
| §2.3 网络（DBE） | `PPO_SB3_AirSim.py` 中的 `OctomapCNN` |
| §2.4 奖励函数 | `Avoid_Reward_Optimized_BodyFrame`、`v_clip_reward_calc`、`_compute_reward` |
| §3.1 渐进式课程学习 | `TrainProgresserCal` + `PPO_SB3_AirSim_CL.py` |
| §3.2 超参数 | `PPO_SB3_AirSim.py` 中的 `PPO(...)` 调用 |
| §4 结果（对照组 vs. 课程组） | `PPO_SB3_AirSim.py` vs `PPO_SB3_AirSim_CL.py`；曲线见 `benchmarks/tb_curves/` |
| BSHM 时间衰减 | `methods/AirSim_ReinforceLearning/BSHM.py` (`_decay_factor`、`_expiration_timeout`、LRU 驱逐) |
| 混合安全盾 | `RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py`（`safe_ruler` 及其相关函数） |
| 安全盾消融表 | `ablation_experiment.py` → `benchmarks/ablation_summary.json` |
| 失效归因 | `collision_diagnostics.py` → `failure_analysis.txt` |

## 版权说明

本仓库不转发出版方的期刊论文 PDF。请使用上方的 DOI 链接。如果确实需要副本、
又无法访问出版方，请开一个 issue。
