# 模型与仿真器产物

[**English**](MODELS.md) | **简体中文**

[← README](README.zh-CN.md) · [汇总材料](benchmarks/README.zh-CN.md) · [论文](paper/README.zh-CN.md)

Git 不跟踪策略 checkpoint。本清单记录 [v0.1.0 Release](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/tag/v0.1.0) 中所列的产物名称；获取时请核对该 Release 的校验清单。列入清单不表示本次文档维护重新验证了下载内容。

## 历史策略产物

下表大小为历史下载说明中的近似值。

| 文件 | 算法／用途 | 本地使用位置 |
|---|---|---|
| [SFT_avoid_modelv3.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/SFT_avoid_modelv3.zip) | 课程 PPO，名义 6 M 步，约 47 MB | RL_scripts/body-axis_v2/results/SFT_avoid_modelv3.zip |
| [20260413-214351_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260413-214351_final_model.zip) | 无课程 PPO 对照，名义 6 M 步，约 47 MB | 自行管理的本地模型目录 |
| [20260427-152015_SAC_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260427-152015_SAC_final_model.zip) | SAC 对照，约 6.144 M 步，约 106 MB | 使用 SAC 加载 |
| [20260207-003758_final_model.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/20260207-003758_final_model.zip) | 历史 PPO 预实验，名义 5.12 M 步，约 47 MB | 可选对照，不是主要报告基线 |

早期文档记录 SFT_avoid_modelv3.zip 与课程运行的 final_model 为同一文件。请对实际取得的文件核对完整校验和，不要仅依赖名称或哈希缩写。

## 加载前提

PPO 归档使用 PPO.load，SAC 归档使用 SAC.load。checkpoint 不能替代解析自定义 OctomapCNN 特征提取器所需的 Python 源码与兼容依赖。应使用相应训练实现，并在评估前核对观测形状、动作约定和归一化方式。

主要 PPO 观测为 656 维，输出为三维世界系速度增量。评估模式和安全盾开关属于实验配置；当前主评估入口默认关闭安全盾。

有权重、仿真器和运行环境，可以开展新的评估，但不能恢复原始训练过程。仅凭当前检出不能独立重建保留摘要及图表。

## 仿真器产物

同一 Release 还列出：

- [airsim-blocks2-standalone-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-standalone-win64.zip)：Windows 打包仿真器。
- [airsim-blocks2-project-win64.zip](https://github.com/MEGO0SFT/uav-drl-avoidance/releases/download/v0.1.0/airsim-blocks2-project-win64.zip)：附 AirSim 依赖的可编辑 Unreal 工程。

配置、隔离补丁位置和许可边界见[仿真器说明](sim/airsim/ENVIRONMENT.zh-CN.md)。

中间 checkpoint 不属于公开源码分发范围。本页不承诺私人训练记录的可获得性。
