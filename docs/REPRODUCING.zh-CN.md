# 复现 Stage-1 结果

[**English**](REPRODUCING.md) | **简体中文**

[← 返回 README](../README.zh-CN.md) · [系统架构](ARCHITECTURE.zh-CN.md) · [仿真环境](../sim/airsim/ENVIRONMENT.zh-CN.md)

本指南用代码与已提交的日志重建论文中的每一个数字。

## 0. 已验证平台

| 项目 | 值 |
|---|---|
| 操作系统 | **WSL2 · Ubuntu 24.04** —— 仿真器、数据交互中心与训练器全部位于*同一个* WSL2 实例中 |
| GPU | NVIDIA **RTX 4070 SUPER**（透传进 WSL2） |
| CUDA | 驱动 **12.0**；PyTorch wheel `2.6.0+cu124` |
| Python | 3.11.14（miniforge / conda） |
| 环境 | [`../environment.yml`](../environment.yml) —— 精确导出文件，为权威依据 |
| 关键版本锁定 | `airsim 1.7.0`、`stable-baselines3 2.7.0`、`sb3-contrib 2.7.1`、`numpy 2.0.1`、`numba 0.62.1`、`open3d 0.19.0`、`gymnasium 1.2.2` |

```bash
conda env create -f environment.yml
conda activate airsim-gym-ubuntu2404
nvidia-smi        # should list the GPU inside WSL2
```

`requirements-legacy.txt` 是仅用 pip 的备用方案；注意 `airsim` 必须是 **1.7.0**（1.8 客户端
与仿真器构建不匹配），`numpy` 必须是 **2.x**（numba 0.62 在 1.x 上会崩溃）。

---

## 1. 拓扑

Stage 1 使用三个角色。在已发表的配置中，三者都运行在同一个 WSL2 实例内；它们也可以分布在
同一局域网中的多台机器上。


三个进程，三条链路：

```
AirSim (UE4)  <--TCP RPC 41451-->  airsim_datamanager  <--UDP-->  trainer
                                    one process,          SubprocVecEnv
                                    all drones, 20 Hz     N parallel envs
```

| 链路 | 方向 | 传输 | 端点 |
|---|---|---|---|
| 仿真 | `airsim_datamanager` → AirSim | TCP RPC | `<airsim_ip>:41451` |
| 状态 | `airsim_datamanager` → trainer | UDP | 中心从 `5812` 发送；每个训练器绑定 `2000 + drone_index` |
| 指令 | trainer → `airsim_datamanager` | UDP | 训练器发送到 `5813`；中心在该端口监听 |

只有 `airsim_datamanager` 与仿真器通信——训练器完全通过 UDP 状态流感知世界，这正是 §1 中
的双机拆分得以成立的原因。

| 配置项 | 位置 | 本仓库中的取值 | 含义 |
|---|---|---|---|
| `airsim_ip` | `airsim_datamanager.py:380` | `172.21.224.1` | 运行 **AirSim** 可执行程序的主机（RPC 端口 `41451`） |
| `trainmanager_ip` | `airsim_datamanager.py:380` | `172.21.236.121` | 运行训练器的主机；即接收 UDP 状态流的网卡 |
| `datamanager_ip` / `trainmanager_ip` | `make_env()` in `PPO_SB3_AirSim*.py`、`train_sac.py`、`train_ddpg.py`、`PPO_SB3_AirSim_model.py` | `172.21.236.121` | 运行 `airsim_datamanager.py` 的主机（两个字段指向同一台机器） |
| `datamanager_port` | `AirSimDroneEnv.__init__`（默认 `5812`） | 中心在 `5812`（状态上行）/ `5813`（指令下行）监听 |
| 训练器端口 | `2000 + uav_id` | 每架无人机一个，因此 `UAV01`→2000、`UAV02`→2001、…… |

> 三个地址在本仓库中都是硬编码的，必须替换为你自己的地址，否则训练器会永久阻塞等待观测。
> 原始实验使用**两台机器**：AirSim 在 `172.21.224.1`，数据交互中心与训练器在
> `172.21.236.121`。单机配置也可以——此时三个字段都填那一个地址。
>
> **IP 配置检查清单**
>
> 1. 在 WSL2 内执行 `ip -4 addr show eth0`；在 Windows 上执行 `ipconfig`。请选用*每一方*都
>    能访问的地址——不要退回到 `127.0.0.1`，因为 UDP socket 会绑定到给定接口，回环绑定会
>    中断跨进程流量。
> 2. 无人机数量必须在三处一致：`make_env()` 中的 `range(10)`、`airsim_datamanager.py` 中的
>    `uav_number=10`，以及实际启动的无人机数量——本项目为 **10**。
>    （`airsim_datamanager_easy.py` 仍保留着早期运行留下的 `uav_number=8`。）
> 3. 每架无人机的控制端口 `2000 + index` 不得与训练器上的其他任何东西冲突。
> 4. WSL2 默认位于 NAT 适配器之后；从 Windows 宿主访问它需要端口代理
>    （`netsh interface portproxy add v4tov4 ...`）。如果仿真器也运行在同一个 WSL2 实例中，
>    请让三方都绑定*同一个*接口——否则你会得到单向流量（能收但永远发不出去）。
> 5. Windows Defender 会拦截新监听的 UDP 端口；首次运行时需放行。

---

## 2. 仿真器配置

1. **场景。** 250 × 250 × 50 m 场地，约 70 个随机放置的基本体（长方体、圆柱、圆锥、球、
   棱锥），空间占据率约 12.5 %；起点区域位于中心，目标区域位于外围。
2. **仿真器配置。** 把 [`../sim/airsim/settings.json`](../sim/airsim/settings.json) 安装为
   AirSim 的 `settings.json`（`~/Documents/AirSim/settings.json`，Windows 上为
   `%USERPROFILE%\Documents\AirSim\settings.json`）。它声明了 **10** 架 `SimpleFlight`
   无人机 `UAV01…UAV10`，以及实际使用的 LiDAR：**32 线、6 Hz、8000 pts/s、80 m 量程、
   垂直 ±30°、水平 360°**，`DataFrame: SensorLocalFrame`。
   > `methods/Sensors1.json` 描述的是一台 16 线 / 25 m / 5 Hz 的传感器——那是一个**过时的
   > 模板**，与已发表结果不符。请使用 `sim/airsim/settings.json`。
   策略只消费 LiDAR 数据；`CameraDefaults` 中的条目经评估后被舍弃。
3. **多机隔离补丁（必需，已应用）。** 全部十架载具都生成在同一个点（`X=0, Y=0, Z=-0.5`），
   而原版 AirSim 允许无人机之间相互碰撞与遮挡，这会把其他智能体的几何信息污染进*单智能体*
   观测中。该补丁位于单个 blueprint 资产中：

   ```
   Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset
   ```

   它做了两处改动：
   - 禁用无人机网格之间的碰撞通道（保留与环境几何的碰撞）；
   - 让 LiDAR 射线投射跳过 `Pawn` 对象，使激光穿过无人机。

   它**已在两个 release 压缩包中应用**，同样的字节也提交在
   [`../sim/airsim/project/Content/BP_FlyingPawn.uasset`](../sim/airsim/project/Content/BP_FlyingPawn.uasset)
   （sha256 `0f4f91c1…`）。请保持该文件原样；换成原版 AirSim pawn 会使观测分布发生偏移，
   PPO 将无法收敛。
   各组件配置以及每个配置项为何重要，见
   [`../sim/airsim/ISOLATION_PATCH.md`](../sim/airsim/ISOLATION_PATCH.zh-CN.md)。

---

## 3. 启动顺序

```bash
# (1) simulator machine
#     install sim/airsim/settings.json as AirSim's settings.json, apply the UE blueprint
#     patches from §2, then start the scene

# (2) hub machine — ONE process serves all N drones
cd RL_scripts/body-axis_v2/scripts
python airsim_datamanager.py          # set uav_number to match your fleet

# (3) trainer machine — pick the group (see §4)
python PPO_SB3_AirSim_CL.py           # curriculum group
# or
python PPO_SB3_AirSim.py              # control group
```

训练会把 TensorBoard 事件写入 `logs/`，把 checkpoint 写入 `models/<timestamp>/`。两者一个
被 git 忽略（checkpoint），一个已提交（logs）。

---

## 4. 论文中的两组 — 一行之差

`PPO_SB3_AirSim.py` 与 `PPO_SB3_AirSim_CL.py` 除两行之外逐字节相同，因此该消融**只**隔离出
渐进式奖励塑形课程：

| | 对照组（`PPO_SB3_AirSim.py`） | 课程组（`PPO_SB3_AirSim_CL.py`） |
|---|---|---|
| `self.training_progress` 初值 | `1.0` | `0.0` |
| `CallbackList` | 不含 `TrainProgresserCal` | **含** `TrainProgresserCal` |

两者都用 `model.learn(total_timesteps=6_000_000)` 训练，调度视野为
`total_timesteps = 1_024_000 × 5 = 5.12 M`，因此课程以及熵/clip 调度在 5.12 M 处到达终值后
保持不变——这就是论文中“课程在 5 M 结束、训练共 6 M”的协议。

> **batch size 说明。** 在 Stable-Baselines3 中 `n_steps=2560` 是*每个环境*的，因此一次更新
> 消费 `n_envs × n_steps = 10 × 2560 = 25 600` 条 transition，而 5.12 M 的调度视野正好是
> 200 个这样的 rollout。

预期结果（来自已提交的日志）：

| 运行 | 最终 `train/success_rate` | 最终 `rollout/ep_rew_mean` |
|---|---:|---:|
| 课程组 | **0.810** | 199.70 |
| 对照组 | 0.764 | — |

---

## 5. 评估

```bash
cd RL_scripts/body-axis_v2/scripts
python PPO_SB3_AirSim_model.py        # loads ../results/SFT_avoid_modelv3.zip
```

它会在 Monte-Carlo 回合数上运行 `evaluate_policy(deterministic=True)`，打印成功率、成功回合
时长的均值/标准差，然后导出一份逐回合的失败报告。在 `make_env()` 中把 `safe_rule=False`
以测量**裸策略**，`safe_rule=True` 则测量**完整系统**。

---

## 6. 复现消融表

```bash
python ablation_experiment.py         # 5 shield configurations × ~300 episodes each
```

结果写入 `scripts/ablation_results/*.json`；已发表的汇总归档于
[`benchmarks/ablation_summary.json`](../benchmarks/ablation_summary.json)。

| 配置 | 回合数 | 成功率 |
|---|---:|---:|
| 纯 DRL | 309 | 90.94 % |
| + 近距斥力 | 307 | 94.46 % |
| + 自适应 VO-τ | 305 | 93.11 % |
| + RRT\*-Safe | 311 | 95.82 % |
| 完整系统 | 525 | 96.95 % |

`ablation_experiment.py` 通过继承 `AirSimDroneEnv` 并重写 `safe_ruler` 来开关安全盾各层；
分布偏移防护与目标邻近闸门在每一组中都保持开启。

---

## 7. 不重新训练也能复现图表

训练曲线已作为抽稀后的 CSV 提交：

```bash
python scripts/export_tb_curves.py RL_scripts/body-axis_v2/logs --out benchmarks/tb_curves
```

每个 CSV 都包含 `step`、`train/success_rate`、`rollout/ep_rew_mean`、
`rollout/ep_len_mean`、`train/approx_kl`、`train/value_loss`、`train/std`、
`train/ent_coef`——足以重绘论文中每一张训练图。原始的 `events.out.tfevents.*` 文件也已
提交，如果你更偏好 TensorBoard：

```bash
tensorboard --logdir RL_scripts/body-axis_v2/logs
```

轨迹图来自 `trajectory_visualizer.py` / `shield_trajectory_visualizer.py`，它们读取
`scripts/figures/` 下已提交的 `.npz` 文件；Monte-Carlo 的逐步轨迹位于
`body-axis_v2_plot/scripts/trajectories_mc.csv`。

---

## 8. 容易踩的坑

- **有效控制频率约为 5.5 Hz，而不是 20 Hz。** `step()` 会睡眠到 0.05 s 的下限，但循环实际
  被 5 Hz 的 LiDAR 旋转与 UDP 往返所限制。一个成功回合约 117–125 步 ≈ 22–24 s 墙钟时间。
- **观测坐标系与动作坐标系刻意不同。** 18×36 的距离图像被旋转到*速度朝向*系，而 Δv 叠加到
  *世界系*速度矢量上。把动作放到机体系会让无人机在探索早期几乎不动，所以这种不对称是刻意
  选择的。
- **`Expect_yaw_Cal()` 是遗留代码。** 实际发送的 yaw 指令是每回合一个随机常数；该函数中的
  速率曲线从未被调用。
- **`EvalCallback` 被构造但从未注册**，因此不会产生 `best_model/`。请改用
  `CheckpointCallback` 的输出或日志。
- **`RecurrentPPO_*` 这些文件名是历史遗留。** 在 `body-axis`、`world-axis-v2` 和
  `airsim-only-with-lidar` 中它们仍带旧名字；所有地方的策略都是前馈 PPO。只有
  `body-axis_v2` 被改名为 `PPO_SB3_AirSim*.py`。
