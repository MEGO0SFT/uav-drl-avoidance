# 系统架构

[**English**](ARCHITECTURE.md) | **简体中文**

[← 返回 README](../README.zh-CN.md) · [复现指南](REPRODUCING.zh-CN.md) · [仿真环境](../sim/airsim/ENVIRONMENT.zh-CN.md)

Stage-1 系统的技术说明。公式直接从实现代码转录，而非取自论文，因此与代码完全一致。
文中给出文件名与函数名，使每条论断都能一步跳到源码。

---

## 1. 问题陈述

四旋翼必须在*先验*未知、障碍密集的三维环境中穿越 100–150 m，并仅依靠一台机械旋转式
LiDAR 与自身的运动学状态，抵达一个从未被测绘过的目标点。

建模为状态与动作空间均连续的**无模型 POMDP**。四项刻意的限制使策略与具体场景无关：

- 策略从不看到绝对位置，只看到**相对目标方向**；
- 地图观测被**旋转到速度朝向系**，因此无论无人机朝哪个方向，“前方”永远是同一块图像区域；
- LiDAR 位姿通过无人机姿态解算，因此**姿态本身不属于观测的一部分**；
- 目标点在可达性区间内随机采样。

---

## 2. 观测 — 656 维

### 2.1 任务向量 `s_task` ∈ ℝ⁸
在 `vehicle_manager.py::_Lidar_RLstat_update` 中构造（全局 NED 系）：

| 索引 | 内容 | 表达式 |
|---|---|---|
| `0:3` | 目标方向 | `(P_tgt − P_uav) / ‖P_tgt − P_uav‖` |
| `3` | 任务进度 | `clip(1 − ‖P_tgt − P_uav‖ / L_max, −1, 1)`, `L_max = 150 m` |
| `4:7` | 速度方向 | `v / ‖v‖` |
| `7` | 归一化速率 | `‖v‖ / v_max`, `v_max = 7.5 m/s` |

### 2.2 环境图像 `s_env` ∈ ℝ⁶⁴⁸
`vehicle_manager.py::APF_SeqCalculate3D` 将 `± stat_range`（`stat_range = 20 m`）内每个被
占据的体素分箱到「仰角 × 方位角」网格中，**Δα = Δβ = 10°**，得到 18 × 36 个栅格。每个栅格
存储该方向上的**归一化最近距离**（`1.0` = 空旷/远，`0` = 相接触）：

```
alpha_world = atan2(y, x)
beta        = atan2(z, sqrt(x² + y²))
yaw         = degrees(atan2(v_y, v_x))          # ← velocity heading, not attitude
alpha_body  = wrap(alpha_world − yaw)           # azimuth in the heading frame
T_env(h, w) = clip(min_range_in_cell, 0, stat_range) / stat_range
```

存储*距离*而非*占据*，使图像成为带可用梯度的稠密连续信号；这一点很关键，因为奖励是它的
直接函数。

> **刻意的坐标系不对称。** 观测表达在速度朝向系中，而动作（见下）叠加在世界系速度矢量上。
> 最初曾尝试把动作放在机体系，结果无人机在探索早期几乎不动——在策略学会该旋转之前，随机
> 动作基本相互抵消。这种不对称是可学习的，因为 `s_task` 携带了 `v_dir`（世界系），而它
> 正好是 `s_env` 的对齐基准；因此策略仅凭观测就能恢复「朝向 ↔ 世界」的旋转。代价是策略
> 在内部执行了一次本可由解析方式给出的旋转。

---

## 3. 动作 — 三维连续

```
a ∈ [−3.75, 3.75]³            # velocity increment Δv, m/s
v_des = v_current + a         # world NED
v_des ← v_des · 7.5/‖v_des‖   if ‖v_des‖ > 7.5
command = [vx, vy, vz, yaw]   # executed by a velocity-tracking PID cascade
```

每轴 3.75 m/s 的界与 `v_max` 相当，三轴最大增量 √3·3.75 ≈ 6.5 m/s——足以完成激进的规避——
而增量式表述意味着策略只需学习对当前运动的*修正*，而不必从零规划一个速度。

---

## 4. 奖励

`r = r_dense + r_sparse`，其中 `r_dense = r_obs + r_smooth + (1 − k_avoid)·r_fwd + r_dist + r_surv`。

### 4.1 障碍项 `r_obs`
`Avoid_Reward_Optimized_BodyFrame`，告警阈值 `D = 0.2`（≈4 m，即在 `v_max` 下约半个反应
时间）。

- 告警集合 `𝒟 = {(i,j) : T_env(i,j) ≤ 2D}`（≤ 8 m）。
- **`𝒟 = ∅`** → `r_obs = 1 − exp(−(d̂_min − 2D)/D)`。
- **`𝒟 ≠ ∅`** → 以 π/6 扇区定义的方向危险权重，加权均值相乘：
  `W_dir = w̄_h · w̄_v`，其中 `w_h = [1.75, 1.50, 1.25, 1.00]`（4 个方位扇区，自正前方起
  递减），`w_v = [1.25, 1.00, 0.90]`（3 个仰角扇区，随远离水平面递减——水平方向的障碍是
  主要碰撞威胁）。
  - 距离惩罚 `f(d) = 1/e − exp(1 − d/D)`，它在 `d = 2D` 处恰好为 0（梯度连续，超出告警
    半径后不再惩罚）；
  - `R_dist = W_dir · mean_{𝒟} f(clip(d, 0, 2D))`；
  - 跨连续步的趋势项，`Δd = d̂_min(t) − clip(d̂_min(t−1), 0, 2D)`：
    接近时为 `R_change = max(Δd/D, −1)`，远离时为 `min(Δd/2D, 0.5)`——接近惩罚的斜率是
    远离奖励的两倍，因此策略无法通过来回振荡来刷 `R_change`；
  - `r_obs = clip(R_dist + R_change, −4, 0.5)`。

`W_dir` 是论文在奖励设计上的主要主张：它让奖励梯度*对方向敏感*，而不是纯粹基于距离，从而
同一个障碍会因其位于前方、侧方还是后方而得到不同的权重。

### 4.2 紧迫度系数 `k_avoid`
```
k_avoid = exp(−3·d̂_min) · clip(exp(W_dir − 1.5), 0, 1)   ∈ [0, 1]
```
它闸控前向奖励：当无人机逼近正前方障碍簇时，`(1 − k_avoid) → 0`，因此无人机必须先脱离危险
状态，才能再次获得进度奖励。正是这一点消除了朴素前向奖励塑形中“一直向前硬顶进障碍”的
吸引子。

### 4.3 平滑项 `r_smooth`
`v_clip_reward_calc`，取最近 `min(10, |history|)` 个速度采样：

```
a_max(v) = 4.0 + 6.0·v/10        j_max(v) = 8.0 + 4.0·v/10
P_a = percentile75( max(|a| − a_max(v), 0) )     # sparse, tail-focused
P_j = max( max(|j| − j_max(v), 0) )              # peak-focused
P_ω = mean( ω² ),  ω = arcsin(‖v̂_t × v̂_t+1‖)/Δt
r_smooth = clip( exp(−0.1 P_a)·exp(−0.05 P_j)·exp(−0.05 P_ω) − 0.5, −0.5, 0.5 )
```

乘性耦合意味着只要违反*任意一项*运动学预算，整项就会崩塌——策略无法用剧烈的航向变化去
交换平滑的加速度曲线。当所有预算都被满足时 `r_smooth = 0.5`。

自适应上限 `a_max(v)`、`j_max(v)` 随速度增长，使高速下的紧急机动仍落在可行集内，而不是
被惩罚。

### 4.4 进度、存活、稀疏项
- `r_fwd = clip( (d_tgt·v_dir)·v_norm·v_max / L_max · 50, −(1+p), +(1+p) )`，
  并由 `(1 − k_avoid)` 加权。
- `r_dist`：当归一化距离比本回合最佳值改善超过 0.01 时 `+1`；当 `k_avoid < 0.25` 而距离
  增大时 `−1`；否则为 `0`。只奖励*新增*进度，因此慢速蠕行无法刷分。
- `r_surv = −0.1 − 0.1·p`（−0.1 → −0.2）：一项小的常数代价，用于抑制盘旋滞留。
- `r_sparse`：到达目标（`< 5 m`）`+50`，碰撞 `−80`，超时（`≥ 600` 步）`−80`。失败惩罚
  刻意大于成功奖励——它抬高了在“接近成功”状态下冒险的期望代价。

---

## 5. 网络 — 双分支特征提取器（DBE）

`PPO_SB3_AirSim.py` 中的 `OctomapCNN`。观测的两半是异质的（稠密二维距离图像 vs. 8 维任务
向量），因此分别编码：

```
range image (1×18×36)                       task vector (8)
  Conv 1→16, k5, pad2 + BN + ELU + MaxPool2     Linear 8→32 + ELU
  Conv 16→32, k5, pad2 + BN + ELU + MaxPool2    Linear 32→64 + ELU
  Conv 32→64, k5, pad2 + BN + ELU               Linear 64→96 + ELU + LayerNorm
  Flatten 2304
  Linear 2304→1024 + ELU
  Linear 1024→256 + ELU + LayerNorm
                    cat -> 352
                        Linear 352→512 + ELU
                        Linear 512→512 + ELU
                        Linear 512→256 + ELU + LayerNorm   → 256-D feature
```

Actor 与 Critic 是各自独立的 5 层 MLP `[512, 256, 256, 256, 128]`，激活函数为 ELU，二者
仅共享特征提取器。权重初始化采用 Kaiming-normal（fan-in，ReLU）。

**PPO 配置：** `n_steps=2560` per env，`batch_size=256`，`n_epochs=4`，
`γ=0.99`，`GAE λ=0.95`，`clip 0.2→0.1`（线性），`lr 1e-4→1e-6`（线性），
`ent_coef 0.05→0.005`（线性），`max_grad_norm = e`，初始动作 σ = 1.0。
十个并行环境（`SubprocVecEnv`）使每次更新得到 25 600 条 transition 的 rollout；总计 6 M 步。

---

## 6. 地图 — 块稀疏哈希地图（BSHM）

`methods/AirSim_ReinforceLearning/BSHM.py`。注意该模块以历史别名 `om_o3d` 导入，其类被
别名为 `OctreeMapManager` / `OctreeProbabilisticMap`，但**运行时并不发生任何八叉树遍历**
——Open3D 的 octree 成员仅用于渲染。

- **两级结构。** 世界被划分为 `s_b = 64 m` 的立方块；每个块是一个从整数体素索引到浮点
  占据概率的哈希映射，体素尺寸为 `64/2^8 = 0.25 m`。
- **贝叶斯更新**在对数几率（log-odds）下进行，因此无需为每个体素保留历史。
- **仅在观测处插入**——构造上即稀疏。
- **双层时间衰减**，运行在 4 Hz 的后台线程上，与控制回路完全解耦：
  - 体素级：`p ← max(p − p·λ, 0.5)`，其中 `λ = 0.015`，并对未被观测超过 `T_exp = 8 s` 的
    体素无条件驱逐（用于处理因反复观测而被推到极高置信度的体素，纯衰减无法快速清除它们）；
  - 块级：在硬性上限 `max_blocks = 2048` 下做 LRU 驱逐，优先选择不含占据体素的块。
- **加锁纪律**：始终先取管理器锁，再取块锁；按块加锁使死锁风险局部化。块锁刻意采用不可
  重入锁，因此递归获取会显式失败，而不是静默死锁。

---

## 7. 混合安全盾

`PPO_SB3_AirSim_model.py`。学习策略在分布外状态下会过于乐观；安全盾是对其速度指令施加的
确定性校验器。各层在 `safe_ruler()` 内按固定顺序执行：

| # | 层 | 触发条件 | 动作 |
|---|---|---|---|
| 0 | 近距斥力（类 APF） | 任一栅格 `T_env ≤ 0.25`（5 m） | `F = −Σ ê/d²` 叠加到机体系速度上，按紧迫度与密度缩放 |
| 1 | 自适应 VO 视野 | 始终生效 | `τ = clip(τ₀ · clip(v/v_max) · clip(d̂_min/0.15), 0.5, 4.0)`，`τ₀ = 3 s` |
| 2 | VO-τ | `τ` 内速度落入碰撞锥 | 通过线性规划做最小范数修正，采用 ORCA 半分担（0.5） |
| 3 | RRT\*-Safe | VO 输出 `< 0.3·‖v_DRL‖` 且 `< 1.5 m/s` | 与一个 2 Hz 后台、150 次迭代的 RRT\* 方向混合，找到路径时 `β = 0.6`，回退到最安全方向时 `0.4` |
| 4 | 分布偏移防护 | `cos(∠v_DRL, v_safe) < 0` | 钳制到 `v_DRL + 0.6·(v_safe − v_DRL)`，使修正量永不超过 90° |
| 5 | 目标邻近闸门 | `d_target < 30 m` | 在 30→10 m 区间内线性地把控制权交还策略 |
| 6 | 速度调制 | `d̂_min < 0.35` | 线性降额至 `0.2·v_max`；`d̂_min = 0.1` 以下硬性限速 1.5 m/s |

设计意图：保留策略的全局最优性，只在策略被证明薄弱的环节借用确定性保证。分布偏移防护是
承重件——没有它，激进修正会把状态推出策略的训练分布，安全盾反而自我拆台。

---

## 8. 刻意*不*包含的东西

- 没有循环记忆。POMDP 由地图（它本身即一种记忆）处理，而非由 RNN 状态处理。
- 没有深度相机，没有 RGB。经过评估后被舍弃：地图已提供几何抽象，而 LiDAR 每字节给出的
  覆盖范围更广。
- 没有特权教师，没有蒸馏。已在 Stage-2 计划中加入——见
  [`../ROADMAP.md`](../ROADMAP.zh-CN.md)。
