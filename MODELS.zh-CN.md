# 模型产物

[**English**](MODELS.md) | **简体中文**

[← 返回 README](README.zh-CN.md) · [基准数据](benchmarks/README.zh-CN.md) · [论文](paper/README.zh-CN.md)

策略 checkpoint **不在** git 中（见 `.gitignore`）：它们是二进制块，其中最大的
超过了 GitHub 单文件 100 MB 的上限，而且超参数扫描产生的 119 个中间 checkpoint
已从归档快照中剔除。本文件记录有哪些产物、每个产物是什么、以及如何获取它们。

训练**日志**则确实在仓库中（`RL_scripts/*/logs/`），此外还有每条曲线的抽稀 CSV
导出（`benchmarks/tb_curves/`）。

## 以 Release 附件形式发布

**下载页面：** <https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0>
· 校验和：该 Release 中的 `SHA256SUMS.txt`。

checkpoint 目录以它们所属的训练运行命名，与 `RL_scripts/body-axis_v2/logs/` 和
`benchmarks/tb_curves/` 一一对应。

| 附件（下载） | 大小 | 本地路径 | 是什么 | 产出 |
|---|---:|---|---|---|
| [`SFT_avoid_modelv3.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/SFT_avoid_modelv3.zip) | 47 MB | `body-axis_v2/results/SFT_avoid_modelv3.zip` | **课程分组策略**，6.0 M 步 —— **每一个**安全盾数字背后的 checkpoint | 成功率曲线 "0.81"、消融表、泛化 |
| [`20260413-214351_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260413-214351_final_model.zip) | 47 MB | `body-axis_v2/models/2026-04-13_paper-control-6M/` | **无课程对照组**，6.0 M 步 | 成功率曲线 "0.764" |
| [`20260427-152015_SAC_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260427-152015_SAC_final_model.zip) | 106 MB | `body-axis_v2/models/2026-04-27_paper-sac-6M/` | SAC 异策略对照，6.1 M 步 | SAC 对比 |
| [`20260207-003758_final_model.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260207-003758_final_model.zip) | 47 MB | `body-axis_v2/models/2026-02-07_pilot-5M/` | 试运行，5.12 M 步 | — |
| *（未发布 —— 历史）* | 47 MB | `world-axis-v2/models/20260311-205650/` | 世界坐标系地图原型 | — |

> **一个文件，两个名字。** `SFT_avoid_modelv3.zip` 与课程学习运行的
> `models/2026-03-18_paper-curriculum-6M/20260318-111106_final_model.zip` 是
> **字节完全相同**的（sha256 `f1fbdf2a…`）。保留前一个名字，是因为
> `PPO_SB3_AirSim_model.py` 和 `ablation_experiment.py` 按该路径加载它；两个都下载
> 没有意义。

仿真器也在同一个 Release 中：
[`airsim-blocks2-standalone-win64.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip)
（209 MB，无需 Unreal Engine 即可运行）以及
[`airsim-blocks2-project-win64.zip`](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip)
（357 MB，可在 UE 4.27 中编辑）—— 见 [`sim/airsim/ENVIRONMENT.zh-CN.md`](sim/airsim/ENVIRONMENT.zh-CN.md)。

每个 zip 都是一个完整的 Stable-Baselines3 `PPO`/`SAC` 保存文件，可用
`PPO.load(path)` 加载 —— 策略超参数（`OctomapCNN`、`features_dim=256`、
`net_arch=[512,256,256,256,128]`）都存放在归档文件内部。

## 加载策略

```python
from stable_baselines3 import PPO
model = PPO.load("SFT_avoid_modelv3.zip", device="cpu")
action, _ = model.predict(obs, deterministic=True)   # deployment uses deterministic=True
```

`PPO_SB3_AirSim_model.py` 展示了完整的评估路径，包括安全盾、成功率记录以及
蒙特卡洛驱动程序。

## 被剔除的中间 checkpoint

最初的扫描共产生 126 个归档（6.0 GB），其中 119 个是逐 step 的
`CheckpointCallback` 转储。它们由作者离线保存，可按需发布；复现论文中的任何数字
都不需要它们，因为每一张已发表的图都来自 (a) 上述最终策略和 (b) 训练日志。
