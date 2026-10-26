# AirSim 基线的运行与评估

[**English**](REPRODUCING.md) | **简体中文**

[← README](../README.zh-CN.md) · [系统架构](ARCHITECTURE.zh-CN.md) · [仿真器](../sim/airsim/ENVIRONMENT.zh-CN.md)

## 可复现范围

当前版本提供源码、配置说明、历史汇总表和最终图表。仅凭当前检出无法重建全部原始训练曲线或逐回合统计。

下列命令是开展新实验的起点，不保证复现历史数值；运行需要外部仿真器／权重及环境相关配置。现有 CI 检查静态仓库属性，不验证训练或飞行行为。

## 1. 运行前提

[environment.yml](../environment.yml)记录 Linux／WSL2 环境：Python 3.11.14、torch 2.6.0+cu124、Stable-Baselines3 2.7.0、sb3-contrib 2.7.1、airsim 1.7.0、numpy 2.0.1 和 Open3D 0.19.0。发布的仿真器包面向 Windows x64／Unreal Engine 4.27。应明确部署拓扑，不假定所有角色位于同一操作系统实例。

    conda env create -f environment.yml
    conda activate airsim-gym-ubuntu2404

[requirements-legacy.txt](../requirements-legacy.txt)是 pip 依赖参考，不能替代仿真器、Python、PyTorch／CUDA 与平台的匹配验证。需在目标环境测试版本兼容性。

## 2. 网络拓扑

    AirSim ← TCP RPC 41451 → 数据中心 ← UDP → 训练环境

| 套接字或字段 | 用途 |
|---|---|
| 数据中心 airsim_ip | 从数据中心可达的 AirSim RPC 地址 |
| 数据中心 trainmanager_ip | 训练器 UDP 状态接收端的目标地址 |
| 环境 datamanager_ip | 发送控制指令的数据中心地址 |
| 环境 trainmanager_ip | 训练器绑定状态接收套接字的本地接口 |
| 数据中心 UDP 5812 | 发送状态消息所绑定的源套接字 |
| 数据中心 UDP 5813 | 接收训练环境的控制指令 |
| 训练器 UDP 2000 + index | 各机体独立的状态接收端，index 从零开始 |

当前入口存在硬编码地址。训练器的两个地址通常不应理解为同一主机：一个是远端目标，一个是本地绑定接口。数据中心的 get_real_ip() 还会自动选择本地接口，应根据实际路由检查其选择。

只有通信双方位于相应的同一网络命名空间时，回环地址才适用。跨 Windows／WSL 或多主机时，使用可达地址并分别验证双向 UDP 和 RPC。路由、防火墙需求随部署变化；TCP 端口代理不能作为通用 UDP 转发方案。

每个 worker 使用的机体名称必须存在于仿真器，并由数据中心提供服务；训练或评估可以只使用其中一个子集。当前完整训练器和数据中心使用十架，默认评估使用一架，easy 数据中心保留历史配置八架。

## 3. 仿真器与传感器

按[仿真器说明](../sim/airsim/ENVIRONMENT.zh-CN.md)取得可运行包，将 [settings.json](../sim/airsim/settings.json)放入仿真器实际使用的 AirSim 配置目录，并运行验证[隔离补丁](../sim/airsim/ISOLATION_PATCH.zh-CN.md)。

当前 settings 定义 UAV01–UAV10，以及 32 线、6 圈／秒、8000 点／秒、80 m 量程、垂直 ±30°、水平 360° 的 LiDAR。策略使用雷达派生几何和任务／运动学状态；methods/Sensors1.json 是历史模板，不是当前传感器规格。

多机隔离属于独立采样协议。关闭机间碰撞与雷达遮挡时，应保留与环境的碰撞响应。

## 4. 训练

完成网络配置并启动仿真器后：

    python RL_scripts/body-axis_v2/scripts/airsim_datamanager.py

再选择一个训练器：

    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py
    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py

前者令课程进度从零开始并注册 TrainProgresserCal，后者固定进度为一。行为差异在奖励调度，而非场景难度。

当前代码为十环境 × 每环境 2560 步，即每轮 25,600 条 transition。课程和熵使用 512 万步调度视野；学习率与 clip 按 PPO learn 进度在约 600 万步预算内调度。

历史 PPO 日志呈现 2048 步报告间隔，与当前十环境计数不吻合。原始运行配置和步数语义尚未完整对齐，不应将当前参数清单表述为已核实的历史精确配置。

历史训练指标 0.810／约 0.764，是各环境最近最多 100 幕滚动成功率的均值，包含训练期探索，不是固定测试集成功率。

新事件文件和 checkpoint 在本地生成，相关路径已被 Git 忽略。

## 5. 评估与消融

从[模型清单](../MODELS.zh-CN.md)取得算法对应的权重。评估入口期望 PPO 文件位于 RL_scripts/body-axis_v2/results/SFT_avoid_modelv3.zip。

    python RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py

运行前检查配置：当前 make_env() 设置 safe_rule=False。进行安全盾实验时需明确启用，并保持策略及场景协议可比较。

已知实现限制：

- 默认单环境评估 525 幕，但成功标志只保留最近 300 幕，最终打印的 overall success rate 实际是滚动统计；成功回合用时另有 300 条窗口。
- 保留的 509/525 属于历史摘要，不能视为已由当前打印路径独立复现。
- ablation_experiment.py 和部分绘图工具仍导入历史名称 RecurrentPPO_SB3_AirSim_model，运行前需适配为现有模块。
- 历史 collision_rate 字段按“100 减成功率”计算，存在超时时也包含超时。
- 成功率、奖励与失败诊断可能使用不同窗口。新实验应建立统一的逐回合记账协议。

这些观察不能据以认定历史汇总错误，而是说明当前代码复现这些结果仍有缺口。

## 6. 图表与本地导出

[benchmarks](../benchmarks/README.zh-CN.md)保留摘要和最终图表。现有最终图片、SVG 和 PDF 均保留；部分绘图工具需要新的本地记录。

scripts/export_tb_curves.py 仅作为本地事件文件导出工具，可处理自行生成的事件文件。导出文件应放在被忽略的本地目录。

## 7. 新实验应记录的内容

记录仿真器／构建版本、策略权重身份、配置、种子、场景采样、决策／控制时序、坐标约定，以及分别统计的回合结果。核查 transition 时序：异步环境可能返回本次动作发送前已采集的观测。

闭环频率应端到端测量。数据中心的 sleep 或 6 Hz LiDAR 参数均不能单独决定策略速率。历史所述约 5.5 Hz 仅是旧运行描述，不是对新部署的验证结果。

当前训练器构造的 EvalCallback 没有加入回调列表，不应默认会产生 best_model 输出，应检查实际 checkpoint 行为。

公开数据和验证要求见[贡献指南](../CONTRIBUTING.zh-CN.md)。
