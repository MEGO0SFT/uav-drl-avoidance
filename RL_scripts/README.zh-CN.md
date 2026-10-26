# `RL_scripts/` — 实验演进脉络

[**English**](README.md) | **简体中文**

[← 返回 README](../README.zh-CN.md) · [训练 run 索引](body-axis_v2/logs/README.zh-CN.md) · [架构说明](../docs/ARCHITECTURE.zh-CN.md)

训练栈共有五代，保留下来是为了让设计决策可追溯。
**其中只有一代是论文发表的实现**；其余都是历史，已不再维护。

| 目录 | 角色 | 地图坐标系 | 动作 | 课程学习 |
|---|---|---|---|---|
| **`body-axis_v2/`** | ★ **发表的实现**（策略 + 安全盾） | 速度航向系 | 世界系 Δv，±3.75 m/s | 可切换：`PPO_SB3_AirSim.py` = 对照组，`PPO_SB3_AirSim_CL.py` = 课程组 |
| `body-axis_v2_plot/` | 绘图 / 蒙特卡洛轨迹导出；训练脚本是 `body-axis_v2` 的逐字节副本 | 同上 | 同上 | 同上 |
| `body-axis/` | 早期原型：姿态机体系、归一化动作、15 维状态 | 姿态机体系 | ±1，按 `0.5·v_max` 缩放 | 始终开启 |
| `world-axis-v2/` | 早期原型：地图固定在世界系，最初的 `W_dir` 思路 | 世界系（`yaw = 0`） | ±2 m/s | 始终开启 |
| `airsim-only-with-lidar/` | 最早的原型：12 维状态、4 维动作、四元数精确角度还原 | 世界系，0–360° | 4 维（目标速度标度 + xyz Δv） | 基于步数 |

## 为什么最终用 `body-axis_v2`

地图 / 观测坐标系历经三次迭代：

1. **世界系**（`world-axis-v2`）—— 18×36 栅格始终保持正北对齐，要计算方向危险
   权重 `W_dir`，就必须*从栅格索引反推*方位角。做法能用，但这种角度反推是一种
   近似，而且“哪一侧是前方”并非随手可得。
2. **姿态机体系**（`body-axis`）—— 栅格随无人机的偏航角旋转。于是动作也必须
   旋转到该坐标系里，这使得早期探索几乎产生不了位移：策略在学会这个旋转之前
   根本动不了，前进 / 进度奖励因此一直是平的，也就没有可学的东西。
3. **速度航向系**（`body-axis_v2`）—— 改为按*速度*方向旋转栅格。“前方”于是
   就是图像的中心列，`W_dir` 从表征中自然得出、无需额外代价，而动作仍留在
   世界系，所以从第一步起探索就能正常工作。

论文报告的正是第 3 步。它留下的这种不对称 —— 航向系的观测、世界系的动作 ——
是有意为之，见 [`../docs/ARCHITECTURE.zh-CN.md`](../docs/ARCHITECTURE.zh-CN.md) §2.2。

## 命名

`body-axis_v2/scripts` 里的 `RecurrentPPO_SB3_AirSim*.py` 已改名为
`PPO_SB3_AirSim*.py`：策略就是普通的前馈 PPO，旧名字只是历史遗留。更早的几个
目录则有意保留其历史文件名，这样论文里任何一个数字都能追溯到产生它的那个确切脚本。

## 每次实验的目录布局

每一代都沿用它自己的训练脚本所产生的布局：

```
scripts/     code
models/      SB3 checkpoints, models/<timestamp>/...  (git-ignored)
logs/        TensorBoard event files                  (committed for body-axis_v2)
results/     named policy archives, evaluation data    (git-ignored)
```

`body-axis_v2/scripts/ablation_results/` 存放安全盾消融背后的原始 JSON；
发表用的汇总镜像到 `../benchmarks/`。
