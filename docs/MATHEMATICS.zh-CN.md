# 数学原理与实现公式

[**English**](MATHEMATICS.md) | **简体中文**

[← 系统架构](ARCHITECTURE.zh-CN.md) · [README](../README.zh-CN.md)

本文展开当前 AirSim 实现中的坐标变换、概率建图、奖励塑形、网络结构及安全盾公式。公式对应交付源码；论文实验与后续扩展的范围见[发表元数据](../paper/README.zh-CN.md)。除明确注明外，位置以 m、速度以 m/s、时间以 s 表示，向量范数为欧氏范数。$\operatorname{clip}(x,l,u)=\min(\max(x,l),u)$，向量裁剪按分量进行。

## 1. 坐标、建图与观测

实现入口：[vehicle_manager.py](../RL_scripts/body-axis_v2/scripts/vehicle_manager.py)、[LidarProcess.py](../methods/AirSim_ReinforceLearning/LidarProcess.py)、[BSHM.py](../methods/AirSim_ReinforceLearning/BSHM.py)。

### 1.1 传感器端点与稀疏体素

传感器点 $\mathbf q_s$ 经 LiDAR 位姿及机体初始位置偏移变为世界 NED 系端点：

$$
\mathbf q_w=R(\mathbf Q)\mathbf q_s+\mathbf t_{\rm lidar}+\mathbf p_{\rm start}.
$$

$R(\mathbf Q)$ 是位姿四元数对应的旋转矩阵；$\mathbf p_{\rm start}$ 将各机体局部位置对齐到公共世界坐标。NED 的正 $z$ 指向下方。建图前端点先按 0.5 m 降采样，因此地图体素更细并不意味着输入具有同等空间分辨率。

块边长 $G=64$，深度参数 $L=8$，体素边长 $h=G/2^L=0.25$。世界点的块键、体素键和体素中心为：

$$
\mathbf b=\lfloor\mathbf q_w/G\rfloor,\qquad
\mathbf k=\lfloor\mathbf q_w/h\rfloor,\qquad
\mathbf c_{\mathbf k}=(\mathbf k+\tfrac12)h.
$$

体素键使用世界坐标，不是块内局部索引。哈希表用于块及体素查询；名为 OctreeMapManager 的兼容接口不表示查询依赖八叉树遍历。

### 1.2 占据概率的 odds 更新

未记录体素的先验为 $P=0.5$。同一次批更新内，落入同一体素的点数为 $n$。将概率换算为 odds，并乘以有界证据因子：

$$
O=\frac{P}{1-P+10^{-8}},\qquad
F_n=
\begin{cases}
\min(1.5^n,5),&\text{占据证据},\\
\max(0.67^n,0.198),&\text{自由证据},
\end{cases}
$$

$$
O'=OF_n,\qquad
P'=\operatorname{clip}\left(\frac{O'}{1+O'},0.001,0.999\right).
$$

例如先验 0.5 收到一个占据点后约为 0.6。多点证据提高占据置信度，因子上限及概率裁剪限制其饱和；这些经验参数不证明同体素点相互独立。实现保存概率字段，而非 log-odds。主感知路径仅写入占据端点，虽有自由更新 API，却没有进行射线自由空间清除。

后台名义每 0.25 s 对 $P>0.5$ 执行：

$$
P\leftarrow\max(0.985P,0.5).
$$

若没有新观测，经过 $k$ 次衰减可写为 $\max(0.985^kP_0,0.5)$，直到记录被删除。衰减不刷新最后观测时间；降至基线且超过 5 s 未观测可删除，超过 8 s 未观测则无条件删除。LRU 默认限制为 2048 块，优先驱逐无占据体素的旧块。这是概率衰减与存储管理，不是显式动态物体跟踪。

### 1.3 航向距离图像

令无人机世界位置为 $\mathbf x$，速度为 $\mathbf v$。查询以 $\mathbf x$ 为中心、各轴延伸 40 m 的包围盒，选择 $P>0.5$ 的体素中心。投影的距离上限为 $R=20$ m。

以下角度以度表示。水平速度航向及其量化为：

$$
\psi=\frac{180}{\pi}\operatorname{atan2}(v_y,v_x),\qquad
\psi_q=10\,\operatorname{round}\left(\frac{\operatorname{wrap}_{[-180,180)}(\psi)}{10}\right).
$$

对 $\mathbf r=\mathbf c_{\mathbf k}-\mathbf x$：

$$
\alpha=\operatorname{wrap}_{[-180,180)}
\left(\frac{180}{\pi}\operatorname{atan2}(r_y,r_x)-\psi_q\right),\qquad
\beta=\frac{180}{\pi}\operatorname{atan2}(r_z,\sqrt{r_x^2+r_y^2}),
$$

$$
j=\operatorname{clip}\left(\left\lfloor\frac{\alpha+180}{10}\right\rfloor,0,35\right),\qquad
i=\operatorname{clip}\left(\left\lfloor\frac{\beta+90}{10}\right\rfloor,0,17\right).
$$

源码使用 digitize 后裁剪，包含上边界的最后一个箱。令 $B_{ij}$ 为落入该箱的点集，则：

$$
T_{ij}=\frac{\min\left(R,\min_{\mathbf r\in B_{ij}}\|\mathbf r\|\right)}{R},
\qquad T_{ij}=1\ \text{当 }B_{ij}=\varnothing.
$$

因此距离图为 $18\times36=648$ 维，近障值较小。空箱与超过上限的距离都表示为 1，不能区分“未知”和“已验证自由”。其朝向是速度航向，不是机体姿态 yaw。

### 1.4 任务向量与动作

目标为 $\mathbf g$，$\mathbf d=\mathbf g-\mathbf x$，$d_g=\|\mathbf d\|$。当 $d_g>0$ 时：

$$
\mathbf s_{\rm task}=
\left[
\operatorname{clip}(\mathbf d/d_g,-1,1),\
\operatorname{clip}(1-d_g/150,-1,1),\
\hat{\mathbf v},\
\|\mathbf v\|/7.5
\right],
$$

其中零速时 $\hat{\mathbf v}=\mathbf0$，否则为速度单位向量。目标方向源码没有零距离保护；速率比也没有再次裁剪。展平距离图与 8 维向量拼接为 656 维输入。

目标与速度向量处于世界系，距离图处于量化航向系。策略输出同样处于世界系的速度增量 $\mathbf a\in[-3.75,3.75]^3$：

$$
\mathbf w=\mathbf v+\mathbf a,\qquad
\mathbf v_{\rm DRL}=
\begin{cases}
\mathbf w,&\|\mathbf w\|\le7.5,\\
7.5\,\mathbf w/\|\mathbf w\|,&\|\mathbf w\|>7.5.
\end{cases}
$$

这是每决策步施加的速度增量。若改变控制频率，不能仍将其解释为同一物理加速度。异步观测与动作的时间对齐见[系统架构](ARCHITECTURE.zh-CN.md)。

## 2. 奖励塑形与渐进课程

来源：[PPO_SB3_AirSim_CL.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py)；无课程对照为 [PPO_SB3_AirSim.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py)。以下为训练函数的实际组合。

### 2.1 课程与总奖励

令 $N$ 为累计环境步数：

$$
p=\min(N/5{,}120{,}000,1),\qquad
c(p)=
\begin{cases}
0,&p\le0.25,\\
(p-0.25)/0.75,&p>0.25.
\end{cases}
$$

对照组固定 $p=1$。课程通过奖励权重与范围改变优化目标，不是逐渐增加障碍物。总奖励为：

$$
r=r_{\rm obs}+c(p)r_{\rm smooth}
 +(1-k_{\rm avoid})r_{\rm fwd}
 +r_{\rm dist}+r_{\rm surv}+r_{\rm terminal},
\qquad r_{\rm surv}=-0.1-0.1p.
$$

### 2.2 方向加权的障碍奖励

告警参数 $D=0.2$ 对应 4 m，危险集合使用两倍告警距离：

$$
M=\{(i,j):T_{ij}\le2D\},\qquad m_t=\min_{i,j}T_{ij}.
$$

奖励代码由索引重建的角度为：

$$
\tilde\alpha_j=(j/35-\tfrac12)2\pi,\qquad
\tilde\beta_i=(i/17-\tfrac12)\pi.
$$

这组端点角不同于投影箱的中心角。权重分区为：

| 方向 | 绝对角度区间 | 权重 |
|---|---|---|
| 水平 | $[0,\pi/6]$, $(\pi/6,\pi/3]$, $(\pi/3,\pi/2]$, $(\pi/2,\pi]$ | 1.75、1.50、1.25、1.00 |
| 仰角 | $[0,\pi/6]$, $(\pi/6,\pi/3]$, $(\pi/3,\pi/2]$ | 1.25、1.00、0.90 |

当 $M$ 非空：

$$
W_{\rm dir}=
\left(\frac1{|M|}\sum_M w_h(\tilde\alpha_j)\right)
\left(\frac1{|M|}\sum_M w_v(\tilde\beta_i)\right),
$$

$$
f(q)=e^{-1}-\exp(1-q/D),\qquad
R_{\rm distribution}=\frac{W_{\rm dir}}{|M|}
 \sum_M f(\operatorname{clip}(T_{ij},0,2D)).
$$

$W_{\rm dir}$ 是两个平均值的乘积，并非逐格联合权重的平均。该结构使前方及较小仰角的近障方向获得较强惩罚。

若存在上一时刻最小距离且 $m_t<2D$，定义 $\Delta m=m_t-\operatorname{clip}(m_{t-1},0,2D)$：

$$
R_\Delta=
\begin{cases}
\max(\Delta m/D,-1),&\Delta m<0,\\
\min(\Delta m/(2D),0.5),&\Delta m\ge0.
\end{cases}
$$

否则 $R_\Delta=0$。接近障碍的惩罚斜率是远离奖励的两倍。完整分段为：

$$
r_{\rm obs}=
\begin{cases}
\operatorname{clip}(R_{\rm distribution}+R_\Delta,-4,0.5),&M\ne\varnothing,\\
1-\exp(-(m_t-2D)/D),&M=\varnothing.
\end{cases}
$$

空集合分支令 $W_{\rm dir}=1$，不经过上述裁剪，故可超过 0.5。距离项与趋势项直接相加，没有 70%／30% 系数。虽然 $f(2D)=0$，分区、集合切换、裁剪和不同分支并不保证全局可微或梯度连续。

### 2.3 前进门控与距离里程碑

$$
k_{\rm avoid}=e^{-3m_t}\operatorname{clip}(e^{W_{\rm dir}-1.5},0,1),
$$

$$
r_{\rm fwd}=\operatorname{clip}
\left((\hat{\mathbf d}\cdot\hat{\mathbf v})\|\mathbf v\|\frac{50}{150},
 -1-p,1+p\right).
$$

前进项衡量沿目标方向的速度；接近障碍时门控降低其贡献。它是奖励塑形意图，不是无碰撞约束。

令 $x_t=d_g/150$，$b$ 为上次获得正距离奖励时记录的里程碑；$b$ 和上一距离初值均为 1：

$$
r_{\rm dist}=
\begin{cases}
+1,&x_t<1\ \land\ x_t<b-0.01,\\
-1,&\text{否则且 }x_t>x_{t-1}\ \land\ k_{\rm avoid}<0.25,\\
0,&\text{其余}.
\end{cases}
$$

仅第一种情况更新 $b\leftarrow x_t$。阈值 0.01 对应 1.5 m；$b$ 不是每一步更新的精确历史最小值。此优先级对应源码中后一个正奖励 if 覆盖前一个负奖励 if。

### 2.4 平滑奖励

实际训练调用 hard 模式，使用最近最多十个速度样本；不足四个时返回 0。名义 $\Delta t=0.1$ s，先取标量速率：

$$
s_\ell=\|\mathbf v_\ell\|,\quad
a_\ell=\operatorname{gradient}(s,\Delta t)_\ell,\quad
j_\ell=\operatorname{gradient}(a,\Delta t)_\ell.
$$

这里的 $a_\ell$ 是速率变化率，不是三维加速度向量。阈值由最后一个样本速率确定：

$$
a_{\max}=4+6s_{\rm last}/10,\qquad j_{\max}=8+4s_{\rm last}/10,
$$

$$
P_a=Q_{75}\big((|a|-a_{\max})_+\big),\qquad
P_j=\max\big((|j|-j_{\max})_+\big).
$$

$Q_{75}$ 为第 75 百分位，$(z)_+=\max(z,0)$。方向项先过滤速率不超过 0.1 的样本，再对剩余相邻单位速度计算：

$$
\omega_\ell=
\frac{\arcsin(\operatorname{clip}
(\|\hat{\mathbf v}_\ell\times\hat{\mathbf v}_{\ell+1}\|,0,1))}{\Delta t},
\qquad P_\omega=\operatorname{mean}(\omega_\ell^2).
$$

有效方向少于两个时 $P_\omega=0$。最终：

$$
r_{\rm smooth}=\operatorname{clip}
\left(e^{-0.1P_a-0.05P_j-0.05P_\omega}-0.5,-0.5,0.5\right).
$$

指数项将超限程度转化为有界奖励，但不构成执行机构的硬约束。$\arcsin(\sin\theta)$ 对大于 90° 的方向差会折叠；过滤后的样本间隔也仍按同一 $\Delta t$ 处理。因此该项是方向变化代理量，而非任意转角下精确的角速度。

### 2.5 终止与调度

终止分支依次判断：$d_g<5$ 时加 50 并成功终止；否则碰撞且回合已开始超过 3 s 时减 80 并终止；否则达到 600 步时减 80 并截断；其余为 0。条件同时发生时按此顺序处理。

课程和熵系数（0.05→0.005）在 512 万步达到终值；学习率（$10^{-4}\to10^{-6}$）及 PPO clip（0.2→0.1）按 learn 的训练进度调度，其预算约 600 万步。这些调度不共享完全相同的时间轴。

## 3. 双分支网络的维度推导

来源：训练脚本中的特征提取器。距离图通过卷积提取局部角度结构，任务向量通过 MLP 保留目标和运动信息，然后融合：

$$
\mathbf z_T=f_{\rm CNN}(T)\in\mathbb R^{256},\quad
\mathbf z_s=f_{\rm MLP}(\mathbf s_{\rm task})\in\mathbb R^{96},\quad
\mathbf z=f_{\rm fuse}([\mathbf z_T;\mathbf z_s])\in\mathbb R^{256}.
$$

| 阶段 | 输出形状或维度 |
|---|---|
| 距离图输入 | $B\times1\times18\times36$ |
| 5×5 卷积，padding 2；BN、ELU；2×2 池化 | $B\times16\times9\times18$ |
| 同类卷积与池化 | $B\times32\times4\times9$ |
| 第三次卷积、BN、ELU，无池化 | $B\times64\times4\times9$ |
| 展平与全连接 | $2304\to1024\to256$，ELU，末端 LayerNorm |
| 任务分支 | $8\to32\to64\to96$，ELU，末端 LayerNorm |
| 拼接与融合 | $352\to512\to512\to256$，ELU，末端 LayerNorm |

$B$ 是批大小。Actor 和 Critic 共享该提取器，随后分别进入隐藏宽度为 $[512,256,256,256,128]$ 的 ELU MLP，再输出策略分布参数和状态价值。没有循环状态；时间上下文主要来自地图维护。

十个环境各采集 2560 步，一轮 rollout 为 $10\times2560=25{,}600$ 条 transition。batch_size=256、n_epochs=4 时，每轮共有 $4\times25{,}600/256=400$ 个小批更新；折扣因子为 0.99，GAE 系数为 0.95。这描述当前配置，不将其视为已核实的历史运行配置。

## 4. 可选混合安全盾

来源：[PPO_SB3_AirSim_model.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py)。以下模块属于仓库扩展；训练奖励不会自动等价于这些速度约束。

### 4.1 航向变换与近距斥力

安全盾使用未量化的水平速度航向 $\psi$：

$$
\mathbf v_h=R_z(-\psi)\mathbf v_w,\qquad
\mathbf v_w=R_z(\psi)\mathbf v_h.
$$

它与距离图中的量化航向 $\psi_q$ 存在差异。安全盾将距离箱中心角 $\alpha_j=-175^\circ+10^\circ j$、$\beta_i=-85^\circ+10^\circ i$ 转为单位方向：

$$
\mathbf u_{ij}=(\cos\beta_i\cos\alpha_j,\cos\beta_i\sin\alpha_j,\sin\beta_i).
$$

对 $T_{ij}<0.25$ 的箱，令 $d_{ij}=20T_{ij}$：

$$
\mathbf f=-\sum_{T_{ij}<0.25}\frac{\mathbf u_{ij}}{\max(d_{ij},0.3)^2},
$$

$$
A=2.5\operatorname{clip}(1-m_t/0.25,0,1)
       \operatorname{clip}(|\{T_{ij}<0.25\}|/30,0.5,2).
$$

若 $\|\mathbf f\|$ 非零，则候选速度加上 $A\mathbf f/\|\mathbf f\|$，随后做速度限幅及反向分量处理。倒数平方提高近障方向权重，密度项提高多个危险方向同时出现时的修正量。

### 4.2 自适应 VO 与 L1 修正

预测视野由原始策略提议速率 $s_{\rm DRL}=\|\mathbf v_{\rm DRL}\|$ 与最近距离比决定：

$$
\tau=\operatorname{clip}\left(
3\operatorname{clip}(s_{\rm DRL}/7.5,0.1,1)
 \operatorname{clip}(m_t/0.15,0.3,1.5),0.5,4\right).
$$

从 6 m 内的方向点进行 1 m 网格聚类，最多选最近 25 个障碍，障碍速度设为零。设障碍相对位置为 $\mathbf r$、相对速度为 $\mathbf w=\mathbf v-\mathbf v_o$，合并半径为 $R_c=0.5+r_o+0.425$。非零相对速度下：

$$
t_{\rm CPA}=\frac{\mathbf r^\top\mathbf w}{\|\mathbf w\|^2},\qquad
d(t)=\|\mathbf r-t\mathbf w\|.
$$

以 $[0,\tau]$ 内最近接近距离筛选冲突，另设重叠和紧急分支。常规分支采用 ORCA 式半修正构造半空间 $\mathbf n_i^\top\mathbf z\ge d_i$；静态障碍不会主动承担另一半避让，不能由该系数推断互惠避碰保证。

令 $\mathbf v_0$ 为斥力处理后的候选速度，LP 求解：

$$
\min_{\mathbf z,\mathbf a,\mathbf b}\sum_{j=1}^{3}(a_j+b_j)
$$

$$
\text{s.t.}\quad
\mathbf z-\mathbf a+\mathbf b=\mathbf v_0,\quad
\mathbf n_i^\top\mathbf z\ge d_i,\quad
-7.5\le z_j\le7.5,\quad \mathbf a,\mathbf b\ge0.
$$

辅助变量分解正负偏差，因此目标等价于最小化 $\|\mathbf z-\mathbf v_0\|_1$。这里是逐轴盒约束，不是带欧氏球约束的二次规划。求解后额外进行欧氏速率限幅；LP 不可用或紧急时使用几何回退。额外限幅不保证仍满足全部原半空间。

### 4.3 RRT* 退路、策略混合与限速

局部 RRT* 后台名义以 2 Hz 规划，视野最高 8 m，使用有限采样预算。仅在斥力或 VO 已介入、修正速度同时低于 $0.3s_{\rm DRL}$ 和 1.5 m/s，且规划结果年龄小于 1.5 s 时融合：

$$
\mathbf v_b=(1-\beta)\mathbf v_{\rm corr}
 +\beta(0.67s_{\rm DRL})\hat{\mathbf d}_{\rm RRT},
\qquad
\beta=\begin{cases}0.6,&\text{找到路径},\\0.4,&\text{否则}.\end{cases}
$$

融合后限上界，并可将非零速度抬至 $\max(0.67s_{\rm DRL},1)$。有限采样及未找到路径时的方向回退不保证全局最优。

若修正方向与原策略方向余弦为负，则：

$$
\mathbf v_s\leftarrow\mathbf v_{\rm DRL}
 +0.6(\mathbf v_s-\mathbf v_{\rm DRL}).
$$

这是插值而非角度投影：一维反例 $v_{\rm DRL}=1,v_s=-1$ 给出 $-0.2$，仍然反向。斥力或 VO 介入后，距目标 $d_g<30$ 时进一步使用：

$$
\eta=\max(0,(d_g-10)/20),\qquad
\mathbf v_s\leftarrow(1-\eta)\mathbf v_{\rm DRL}+\eta\mathbf v_s.
$$

最后只缩短超限向量，速率上限为：

$$
s_{\rm cap}(m_t)=
\begin{cases}
1.5,&m_t<0.1,\\
7.5(0.2+0.8m_t/0.35),&0.1\le m_t<0.35,\\
7.5,&m_t\ge0.35.
\end{cases}
$$

阈值 0.1 和 0.35 分别对应 2 m 与 7 m；在 0.1 处是不连续切换。后续混合和限速没有重新校验所有 VO 约束，因此这些公式描述启发式控制流程，不能合成为端到端安全证明。

