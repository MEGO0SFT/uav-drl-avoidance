# Mathematical Foundations and Implementation

**English** | [简体中文](MATHEMATICS.zh-CN.md)

[← Architecture](ARCHITECTURE.md) · [README](../README.md)

This guide expands the coordinate transforms, probabilistic mapping, reward shaping, network dimensions and shield formulas in the delivered AirSim code. See [publication metadata](../paper/README.md) for the distinction between the paper and repository extensions. Unless stated otherwise, position is in metres, velocity in metres per second, time in seconds, and vector norms are Euclidean. $\operatorname{clip}(x,l,u)=\min(\max(x,l),u)$; vector clipping is componentwise.

## 1. Coordinates, mapping and observations

Implementation: [vehicle_manager.py](../RL_scripts/body-axis_v2/scripts/vehicle_manager.py), [LidarProcess.py](../methods/AirSim_ReinforceLearning/LidarProcess.py) and [BSHM.py](../methods/AirSim_ReinforceLearning/BSHM.py).

### 1.1 Sensor endpoints and sparse voxels

A sensor point $\mathbf q_s$ is transformed to the world NED frame using the LiDAR pose and vehicle start offset:

$$
\mathbf q_w=R(\mathbf Q)\mathbf q_s+\mathbf t_{\rm lidar}+\mathbf p_{\rm start}.
$$

$R(\mathbf Q)$ is the pose quaternion's rotation matrix; $\mathbf p_{\rm start}$ aligns vehicle-local positions with the common world frame. Positive NED $z$ points downwards. Endpoints are downsampled at 0.5 m before mapping, so a finer map voxel does not imply equally fine input resolution.

With block size $G=64$, depth parameter $L=8$ and voxel size $h=G/2^L=0.25$, the block key, voxel key and voxel centre are:

$$
\mathbf b=\lfloor\mathbf q_w/G\rfloor,\qquad
\mathbf k=\lfloor\mathbf q_w/h\rfloor,\qquad
\mathbf c_{\mathbf k}=(\mathbf k+\tfrac12)h.
$$

Voxel keys use world coordinates, not block-local indices. Hash tables provide block and voxel lookup; the compatibility name OctreeMapManager does not mean occupancy queries traverse an octree.

### 1.2 Occupancy updates in odds space

An unrecorded voxel starts with prior $P=0.5$. Let $n$ be the number of points assigned to that voxel within one batch. Convert probability to odds and multiply by a bounded evidence factor:

$$
O=\frac{P}{1-P+10^{-8}},\qquad
F_n=
\begin{cases}
\min(1.5^n,5),&\text{occupied evidence},\\
\max(0.67^n,0.198),&\text{free evidence},
\end{cases}
$$

$$
O'=OF_n,\qquad
P'=\operatorname{clip}\left(\frac{O'}{1+O'},0.001,0.999\right).
$$

For example, one occupied point changes a prior of 0.5 to approximately 0.6. Multiple points strengthen occupancy evidence, while the factor bounds and probability clipping limit saturation. These empirical parameters do not establish independence of points in the same voxel. The implementation stores probabilities, not log-odds. The active perception path only inserts occupied endpoints; the free-update API does not imply ray-based free-space clearing.

Nominally every 0.25 s, probabilities above 0.5 undergo:

$$
P\leftarrow\max(0.985P,0.5).
$$

Without new observations, after $k$ updates this is $\max(0.985^kP_0,0.5)$ until deletion. Decay does not refresh the last observation time. A record that reaches the baseline and has not been observed for over 5 s can be removed; a record older than 8 s is removed regardless of probability. LRU storage defaults to 2048 blocks and preferentially evicts older blocks without occupied voxels. This is probability decay and storage management, not explicit moving-object tracking.

### 1.3 Heading distance image

Let the vehicle world position and velocity be $\mathbf x$ and $\mathbf v$. Query a box extending 40 m along each axis around $\mathbf x$, retaining voxel centres with $P>0.5$. Projection distances are capped at $R=20$ m.

Angles below are in degrees. Horizontal velocity heading and its quantisation are:

$$
\psi=\frac{180}{\pi}\operatorname{atan2}(v_y,v_x),\qquad
\psi_q=10\,\operatorname{round}\left(\frac{\operatorname{wrap}_{[-180,180)}(\psi)}{10}\right).
$$

For $\mathbf r=\mathbf c_{\mathbf k}-\mathbf x$:

$$
\alpha=\operatorname{wrap}_{[-180,180)}
\left(\frac{180}{\pi}\operatorname{atan2}(r_y,r_x)-\psi_q\right),\qquad
\beta=\frac{180}{\pi}\operatorname{atan2}(r_z,\sqrt{r_x^2+r_y^2}),
$$

$$
j=\operatorname{clip}\left(\left\lfloor\frac{\alpha+180}{10}\right\rfloor,0,35\right),\qquad
i=\operatorname{clip}\left(\left\lfloor\frac{\beta+90}{10}\right\rfloor,0,17\right).
$$

The code uses digitize followed by clipping, including the upper boundary in the final bin. Let $B_{ij}$ contain the points assigned to a bin:

$$
T_{ij}=\frac{\min\left(R,\min_{\mathbf r\in B_{ij}}\|\mathbf r\|\right)}{R},
\qquad T_{ij}=1\ \text{when }B_{ij}=\varnothing.
$$

The image has $18\times36=648$ entries; smaller values indicate nearer obstacles. Empty bins and saturated distances both map to 1, so the representation does not distinguish unknown space from verified free space. Its orientation follows velocity heading, not body yaw.

### 1.4 Task vector and action

For target $\mathbf g$, define $\mathbf d=\mathbf g-\mathbf x$ and $d_g=\|\mathbf d\|$. Assuming $d_g>0$:

$$
\mathbf s_{\rm task}=
\left[
\operatorname{clip}(\mathbf d/d_g,-1,1),\
\operatorname{clip}(1-d_g/150,-1,1),\
\hat{\mathbf v},\
\|\mathbf v\|/7.5
\right],
$$

where $\hat{\mathbf v}=\mathbf0$ at zero speed and otherwise is the velocity unit vector. The source does not guard division by zero for the target direction; the speed ratio is not clipped again. Concatenating the flattened distance image and this eight-dimensional vector gives 656 inputs.

Target and velocity vectors use world coordinates, while the distance image uses a quantised heading frame. The policy also outputs a world-frame velocity increment $\mathbf a\in[-3.75,3.75]^3$:

$$
\mathbf w=\mathbf v+\mathbf a,\qquad
\mathbf v_{\rm DRL}=
\begin{cases}
\mathbf w,&\|\mathbf w\|\le7.5,\\
7.5\,\mathbf w/\|\mathbf w\|,&\|\mathbf w\|>7.5.
\end{cases}
$$

This increment is applied per decision step. Changing the control frequency changes its physical interpretation; it is not a frequency-independent acceleration. See [architecture](ARCHITECTURE.md) for asynchronous observation/action alignment.

## 2. Reward shaping and progressive curriculum

Source: [PPO_SB3_AirSim_CL.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_CL.py); the non-curriculum control is [PPO_SB3_AirSim.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim.py). The formulas below follow the actual training functions.

### 2.1 Curriculum and total reward

Let $N$ be the cumulative environment-step count:

$$
p=\min(N/5{,}120{,}000,1),\qquad
c(p)=
\begin{cases}
0,&p\le0.25,\\
(p-0.25)/0.75,&p>0.25.
\end{cases}
$$

The control fixes $p=1$. The curriculum changes reward weights and ranges, rather than increasing obstacle count. Total reward is:

$$
r=r_{\rm obs}+c(p)r_{\rm smooth}
 +(1-k_{\rm avoid})r_{\rm fwd}
 +r_{\rm dist}+r_{\rm surv}+r_{\rm terminal},
\qquad r_{\rm surv}=-0.1-0.1p.
$$

### 2.2 Direction-weighted obstacle reward

The warning parameter $D=0.2$ corresponds to 4 m; the danger set includes distances up to twice that value:

$$
M=\{(i,j):T_{ij}\le2D\},\qquad m_t=\min_{i,j}T_{ij}.
$$

The reward code reconstructs angles from indices as:

$$
\tilde\alpha_j=(j/35-\tfrac12)2\pi,\qquad
\tilde\beta_i=(i/17-\tfrac12)\pi.
$$

These endpoint angles differ from projection-bin centres. Directional weights are:

| Direction | Absolute-angle intervals | Weights |
|---|---|---|
| Horizontal | $[0,\pi/6]$, $(\pi/6,\pi/3]$, $(\pi/3,\pi/2]$, $(\pi/2,\pi]$ | 1.75, 1.50, 1.25, 1.00 |
| Elevation | $[0,\pi/6]$, $(\pi/6,\pi/3]$, $(\pi/3,\pi/2]$ | 1.25, 1.00, 0.90 |

For nonempty $M$:

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

$W_{\rm dir}$ is the product of two averages, not an average of per-cell joint weights. This gives stronger penalties to nearby directions in front of the vehicle and at smaller elevation magnitudes.

If a previous minimum distance exists and $m_t<2D$, let $\Delta m=m_t-\operatorname{clip}(m_{t-1},0,2D)$:

$$
R_\Delta=
\begin{cases}
\max(\Delta m/D,-1),&\Delta m<0,\\
\min(\Delta m/(2D),0.5),&\Delta m\ge0.
\end{cases}
$$

Otherwise $R_\Delta=0$. The penalty slope for approaching an obstacle is twice the reward slope for retreating. The complete expression is:

$$
r_{\rm obs}=
\begin{cases}
\operatorname{clip}(R_{\rm distribution}+R_\Delta,-4,0.5),&M\ne\varnothing,\\
1-\exp(-(m_t-2D)/D),&M=\varnothing.
\end{cases}
$$

The empty-set branch sets $W_{\rm dir}=1$ and is not clipped by the first branch, so it may exceed 0.5. Distribution and trend terms are added directly, without 70%/30% coefficients. Although $f(2D)=0$, partitioning, set changes, clipping and the different branches do not establish global differentiability or gradient continuity.

### 2.3 Forward gating and distance milestones

$$
k_{\rm avoid}=e^{-3m_t}\operatorname{clip}(e^{W_{\rm dir}-1.5},0,1),
$$

$$
r_{\rm fwd}=\operatorname{clip}
\left((\hat{\mathbf d}\cdot\hat{\mathbf v})\|\mathbf v\|\frac{50}{150},
 -1-p,1+p\right).
$$

The forward term measures velocity along the target direction. Its contribution decreases near obstacles through the gate; this is a shaping mechanism, not a collision constraint.

Let $x_t=d_g/150$ and let $b$ be the milestone recorded at the last positive distance reward. Both $b$ and the previous-distance value start at 1:

$$
r_{\rm dist}=
\begin{cases}
+1,&x_t<1\ \land\ x_t<b-0.01,\\
-1,&\text{otherwise if }x_t>x_{t-1}\ \land\ k_{\rm avoid}<0.25,\\
0,&\text{otherwise}.
\end{cases}
$$

Only the first case updates $b\leftarrow x_t$. The 0.01 threshold corresponds to 1.5 m; $b$ is not an exact running minimum updated every step. This priority reflects the later positive-reward if overriding the earlier negative-reward if in the source.

### 2.4 Smoothness reward

Training uses hard mode with at most the latest ten velocity samples; fewer than four samples produce zero reward. With nominal $\Delta t=0.1$ s, first compute scalar speed:

$$
s_\ell=\|\mathbf v_\ell\|,\quad
a_\ell=\operatorname{gradient}(s,\Delta t)_\ell,\quad
j_\ell=\operatorname{gradient}(a,\Delta t)_\ell.
$$

Here $a_\ell$ is the rate of change of speed, not a three-dimensional acceleration vector. Thresholds depend on the last sample's speed:

$$
a_{\max}=4+6s_{\rm last}/10,\qquad j_{\max}=8+4s_{\rm last}/10,
$$

$$
P_a=Q_{75}\big((|a|-a_{\max})_+\big),\qquad
P_j=\max\big((|j|-j_{\max})_+\big).
$$

$Q_{75}$ denotes the 75th percentile and $(z)_+=\max(z,0)$. For the direction term, first discard samples with speed at most 0.1, then use consecutive remaining unit velocities:

$$
\omega_\ell=
\frac{\arcsin(\operatorname{clip}
(\|\hat{\mathbf v}_\ell\times\hat{\mathbf v}_{\ell+1}\|,0,1))}{\Delta t},
\qquad P_\omega=\operatorname{mean}(\omega_\ell^2).
$$

With fewer than two valid directions, $P_\omega=0$. Finally:

$$
r_{\rm smooth}=\operatorname{clip}
\left(e^{-0.1P_a-0.05P_j-0.05P_\omega}-0.5,-0.5,0.5\right).
$$

The exponential terms convert excess motion into a bounded reward, not hard actuator constraints. $\arcsin(\sin\theta)$ folds differences larger than 90°, and filtered samples still use the same nominal interval. This is a direction-change surrogate rather than an exact angular rate for arbitrary turns.

### 2.5 Termination and schedules

Terminal checks run in this order: if $d_g<5$, add 50 and terminate successfully; otherwise, a collision more than 3 s after episode start subtracts 80 and terminates; otherwise, reaching 600 steps subtracts 80 and truncates; all other cases add zero. Simultaneous conditions follow that order.

Curriculum and entropy coefficient (0.05→0.005) reach their final values at 5.12 million steps. Learning rate ($10^{-4}\to10^{-6}$) and PPO clip (0.2→0.1) follow learn progress with a budget of about six million steps. These schedules do not have identical timelines.

## 3. Deriving the two-branch network dimensions

Source: the feature extractor in the training scripts. Convolutions extract local angular structure from the distance image, while an MLP embeds target and motion information before fusion:

$$
\mathbf z_T=f_{\rm CNN}(T)\in\mathbb R^{256},\quad
\mathbf z_s=f_{\rm MLP}(\mathbf s_{\rm task})\in\mathbb R^{96},\quad
\mathbf z=f_{\rm fuse}([\mathbf z_T;\mathbf z_s])\in\mathbb R^{256}.
$$

| Stage | Output shape or dimensions |
|---|---|
| Distance image input | $B\times1\times18\times36$ |
| 5×5 convolution, padding 2; BN, ELU; 2×2 pooling | $B\times16\times9\times18$ |
| Another convolution and pooling stage | $B\times32\times4\times9$ |
| Third convolution, BN and ELU, no pooling | $B\times64\times4\times9$ |
| Flatten and fully connected layers | $2304\to1024\to256$, ELU, final LayerNorm |
| Task branch | $8\to32\to64\to96$, ELU, final LayerNorm |
| Concatenation and fusion | $352\to512\to512\to256$, ELU, final LayerNorm |

$B$ is the batch size. Actor and critic share this extractor, then use separate ELU MLPs with hidden widths $[512,256,256,256,128]$ before their policy-distribution and value outputs. There is no recurrent state; temporal context mainly comes from map maintenance.

Ten environments collecting 2560 steps each produce $10\times2560=25{,}600$ transitions per rollout. With batch_size=256 and n_epochs=4, a rollout entails $4\times25{,}600/256=400$ minibatch updates. The discount is 0.99 and GAE coefficient is 0.95. This describes current configuration, not a verified reconstruction of historical runs.

## 4. Optional hybrid shield

Source: [PPO_SB3_AirSim_model.py](../RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py). These modules are repository extensions; training rewards are not equivalent to the velocity constraints below.

### 4.1 Heading transforms and local repulsion

The shield uses unquantised horizontal velocity heading $\psi$:

$$
\mathbf v_h=R_z(-\psi)\mathbf v_w,\qquad
\mathbf v_w=R_z(\psi)\mathbf v_h.
$$

This differs from the distance image's quantised $\psi_q$. The shield converts bin-centre angles $\alpha_j=-175^\circ+10^\circ j$ and $\beta_i=-85^\circ+10^\circ i$ to unit directions:

$$
\mathbf u_{ij}=(\cos\beta_i\cos\alpha_j,\cos\beta_i\sin\alpha_j,\sin\beta_i).
$$

For bins with $T_{ij}<0.25$, let $d_{ij}=20T_{ij}$:

$$
\mathbf f=-\sum_{T_{ij}<0.25}\frac{\mathbf u_{ij}}{\max(d_{ij},0.3)^2},
$$

$$
A=2.5\operatorname{clip}(1-m_t/0.25,0,1)
       \operatorname{clip}(|\{T_{ij}<0.25\}|/30,0.5,2).
$$

For nonzero $\|\mathbf f\|$, add $A\mathbf f/\|\mathbf f\|$ to the candidate velocity, followed by speed limiting and backward-component handling. Inverse-square weighting emphasises nearer directions; the density factor strengthens the correction when several directions are dangerous.

### 4.2 Adaptive VO and L1 correction

The prediction horizon depends on original policy-proposal speed $s_{\rm DRL}=\|\mathbf v_{\rm DRL}\|$ and minimum distance ratio:

$$
\tau=\operatorname{clip}\left(
3\operatorname{clip}(s_{\rm DRL}/7.5,0.1,1)
 \operatorname{clip}(m_t/0.15,0.3,1.5),0.5,4\right).
$$

Directional points within 6 m are clustered on a 1 m grid; at most the nearest 25 obstacles are retained, with obstacle velocity set to zero. Let relative obstacle position be $\mathbf r$, relative velocity $\mathbf w=\mathbf v-\mathbf v_o$, and combined radius $R_c=0.5+r_o+0.425$. For nonzero relative velocity:

$$
t_{\rm CPA}=\frac{\mathbf r^\top\mathbf w}{\|\mathbf w\|^2},\qquad
d(t)=\|\mathbf r-t\mathbf w\|.
$$

Closest approach within $[0,\tau]$ identifies conflicts, with separate overlap and emergency branches. The ordinary branch uses an ORCA-style half correction to construct half-spaces $\mathbf n_i^\top\mathbf z\ge d_i$. Static obstacles do not actively undertake the other half of avoidance, so the factor does not establish a reciprocal collision-avoidance guarantee.

For the repulsion-adjusted candidate $\mathbf v_0$, the LP solves:

$$
\min_{\mathbf z,\mathbf a,\mathbf b}\sum_{j=1}^{3}(a_j+b_j)
$$

$$
\text{s.t.}\quad
\mathbf z-\mathbf a+\mathbf b=\mathbf v_0,\quad
\mathbf n_i^\top\mathbf z\ge d_i,\quad
-7.5\le z_j\le7.5,\quad \mathbf a,\mathbf b\ge0.
$$

The auxiliary variables split positive and negative deviations, making the objective equivalent to minimising $\|\mathbf z-\mathbf v_0\|_1$. Bounds are componentwise, not a Euclidean-ball constraint in a quadratic program. A Euclidean speed cap is applied after solving; unavailable LP solutions and emergencies use geometric fallback. The extra cap need not preserve every original half-space.

### 4.3 RRT* fallback, policy blending and speed limits

Local RRT* runs in the background at a nominal 2 Hz, with a horizon up to 8 m and a finite sampling budget. Blending occurs only after repulsion or VO intervenes, when corrected speed is below both $0.3s_{\rm DRL}$ and 1.5 m/s, and the plan is less than 1.5 s old:

$$
\mathbf v_b=(1-\beta)\mathbf v_{\rm corr}
 +\beta(0.67s_{\rm DRL})\hat{\mathbf d}_{\rm RRT},
\qquad
\beta=\begin{cases}0.6,&\text{path found},\\0.4,&\text{otherwise}.\end{cases}
$$

The blend is capped above, and a nonzero speed may then be raised to $\max(0.67s_{\rm DRL},1)$. Finite sampling and directional fallback when no path is found do not guarantee global optimality.

If the cosine between corrected and original policy directions is negative:

$$
\mathbf v_s\leftarrow\mathbf v_{\rm DRL}
 +0.6(\mathbf v_s-\mathbf v_{\rm DRL}).
$$

This is interpolation, not angular projection: the one-dimensional example $v_{\rm DRL}=1,v_s=-1$ produces $-0.2$, still reversed. After repulsion or VO intervention, a target distance $d_g<30$ also triggers:

$$
\eta=\max(0,(d_g-10)/20),\qquad
\mathbf v_s\leftarrow(1-\eta)\mathbf v_{\rm DRL}+\eta\mathbf v_s.
$$

The final cap only shortens vectors exceeding the limit:

$$
s_{\rm cap}(m_t)=
\begin{cases}
1.5,&m_t<0.1,\\
7.5(0.2+0.8m_t/0.35),&0.1\le m_t<0.35,\\
7.5,&m_t\ge0.35.
\end{cases}
$$

Thresholds 0.1 and 0.35 correspond to 2 m and 7 m; the switch at 0.1 is discontinuous. Subsequent blending and limiting do not recheck every VO constraint. These formulas describe a heuristic control pipeline and cannot be combined into an end-to-end safety proof.

