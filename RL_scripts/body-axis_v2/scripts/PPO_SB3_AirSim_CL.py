"""
PPO_SB3_AirSim_CL.py — curriculum-learning group of the paper's ablation.

=============================================================================
THIS FILE IS THE CURRICULUM GROUP.  Its twin, ``PPO_SB3_AirSim.py``, is the
NO-CURRICULUM CONTROL GROUP.  The two files are byte-identical except for
exactly two lines, so that the ablation isolates the progressive reward-shaping
curriculum and nothing else:

    ┌─────────────────────────────┬──────────────┬──────────────────────────┐
    │                             │ control      │ curriculum               │
    ├─────────────────────────────┼──────────────┼──────────────────────────┤
    │ self.training_progress init │ 1.0          │ 0.0                      │
    │ CallbackList                │ without      │ with                     │
    │                             │ TrainProgres-│ TrainProgres-            │
    │                             │ serCal       │ serCal                   │
    └─────────────────────────────┴──────────────┴──────────────────────────┘

Both runs use the same 6 M-step budget; ``total_timesteps`` for the curriculum
and entropy schedules is 1 024 000 × 5 = 5.12 M, so both schedules reach their
final value at 5.12 M and are then held constant — this is the "curriculum ends
at 5 M, training totals 6 M" protocol reported in the paper.

The run directories produced by the original experiments are named after the experiment:
``models/2026-03-18_paper-curriculum-6M/`` (this file) and
``models/2026-04-13_paper-control-6M/`` (the control), with matching TensorBoard runs under
``logs/`` and curve CSVs under ``benchmarks/tb_curves/``.
=============================================================================

Filename note: despite the historical ``RecurrentPPO`` naming used elsewhere in
this repository, the policy is a plain feed-forward PPO with the custom
``OctomapCNN`` feature extractor — there is no LSTM/GRU involved.
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
import os, threading, time, pickle, sys
import vehicle_manager as VM
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import torch 
import torch.nn as nn
import pickle
import struct
import socket

MAX_PACKET_SIZE = 62000
global abmatrix
a_b_matrix = np.zeros((7,24,2), dtype=np.float64)
x, y = np.arange(-180,180,15), np.arange(-45, 50, 15)
xx, yy = np.meshgrid(x,y)
a_b_matrix[:,:,0] = yy
a_b_matrix[:,:,1] = xx
abmatrix = a_b_matrix/180*np.pi
    
def v_clip_reward_calc(stat_deque, octomap_stat, Seqspace_directions, dt=0.1, mode='hard'):
    """
    计算无人机飞行平滑性奖励，基于速度向量的历史数据\n
    奖励考虑加速度、jerk（加速度变化率）和方向变化率，鼓励平稳飞行\n
    Args:
        stat_deque: 包含历史状态的双端队列，每个状态包含速度向量
        dt: 状态采样时间间隔，默认为0.05秒
    Returns:
        平滑性奖励，范围[0, 1]，值越大表示飞行越平滑"""
    if len(stat_deque) < 4:  # 至少4个点，计算更可靠
        return 0.0
    if mode == 'hard':
        # 取合适长度的历史数据（5-10点最佳）
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        
        # 提取数据
        velocities = []
        for stat in stats:
            if "velocity" in stat:
                velocities.append(stat["velocity"])
        
        velocities = np.array(velocities)
        
        if len(velocities) < 4:
            return 0.0
        
        # 1. 速度大小序列
        v_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # 2. 计算加速度
        accels = np.gradient(v_magnitudes, dt)
        
        # 3. 计算jerk
        jerks = np.gradient(accels, dt) if len(accels) > 1 else np.zeros_like(accels)
        
        # 4. 方向平滑性（使用角度变化率）
        if len(velocities) >= 3:
            # 计算速度方向单位向量
            v_norms = np.linalg.norm(velocities, axis=1)
            valid_mask = v_norms > 0.1  # 忽略接近零的速度
            
            if np.sum(valid_mask) >= 2:
                # 只使用有效数据
                valid_velocities = velocities[valid_mask]
                valid_v_norms = v_norms[valid_mask]
                
                # 方向单位向量
                directions = valid_velocities / valid_v_norms[:, np.newaxis]
                
                # 计算角速度（方向变化率）
                # 连续方向向量的叉积大小 ≈ sin(θ)
                if len(directions) >= 2:
                    crosses = np.cross(directions[:-1], directions[1:])
                    sin_theta = np.linalg.norm(crosses, axis=1)
                    
                    # 角速度 = Δθ / Δt
                    # 对于小角度，sinθ ≈ θ
                    angular_velocities = np.arcsin(np.clip(sin_theta, 0, 1)) / dt
                    
                    # 角度变化惩罚（均方角速度）
                    angular_penalty = np.mean(angular_velocities**2)
                else:
                    angular_penalty = 0.0
            else:
                angular_penalty = 0.0
        else:
            angular_penalty = 0.0
        
        # 5. 自适应阈值（基于当前速度）
        current_speed = v_magnitudes[-1] if len(v_magnitudes) > 0 else 0.0
        
        # 动态阈值：高速时允许稍大加速度
        max_accel = 4.0 + 6.0 * current_speed / 10.0  # 范围[4.0, 10.0]
        max_jerk = 8.0 + 4.0 * current_speed / 10.0   # 范围[8.0, 12.0]
        
        # 6. 计算惩罚（使用鲁棒的统计量）
        # 加速度惩罚：只关注超过阈值部分的上四分位数
        accel_excess = np.maximum(np.abs(accels) - max_accel, 0)
        accel_penalty = np.percentile(accel_excess, 75) if len(accel_excess) > 0 else 0.0
        
        # jerk惩罚：使用最大值（最坏情况）
        jerk_excess = np.maximum(np.abs(jerks) - max_jerk, 0)
        jerk_penalty = np.max(jerk_excess) if len(jerk_excess) > 0 else 0.0
        
        # 7. 综合奖励
        # 使用指数衰减形式的奖励
        smoothness_score = np.exp(-0.1 * accel_penalty) * np.exp(-0.05 * jerk_penalty) * np.exp(-0.05 * angular_penalty)
        
        # 确保在[-0.5, 0.5]范围
        return float(np.clip(smoothness_score-0.5, -0.5, 0.5))
    elif mode == 'easy':
        # 简单模式， 只考虑两项
        # 取合适长度的历史数据（5-10点最佳）
        # 取octomap_stat衡量速度方向
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]

        velocities = []
        for stat in stats:
            if "velocity" in stat:
                velocities.append(stat["velocity"])

        current_speed = stat_deque[-1]["velocity"]
        
        velocities = np.array(velocities)
        if len(velocities) < 2:
            return 0.0  
        a_xyz = np.gradient(velocities, dt, axis=0)
        # 衡量各个方向加速度变化率的标准差
        a_xyz_std = np.std(a_xyz, axis=0)
        # 计算奖励
        v_norm = np.linalg.norm(current_speed)/10.0
        v_norm_reward = np.exp(-(v_norm-0.2)*(v_norm-0.8)/0.09-1) - 1
        accel_reward = np.clip(0.5 - 0.05 * np.sum(a_xyz_std), -0.5, 0.5)
        return accel_reward + v_norm_reward
    else:
        return 0.0

def Avoid_Reward_Optimized_BodyFrame(octomap_body, q, prev_min_dist=None, D=0.3334):
    """
    基于体坐标系距离矩阵的避障奖励计算函数
    
    Args:
        octomap_body: 体坐标系下的距离矩阵(2D数组，已蕴含角度信息)
        q: 无人机姿态四元数[w, x, y, z]
        prev_min_dist: 上一时刻的最短障碍距离，用于计算变化率
        D: 警戒距离，默认为0.3334
    
    Returns:
        R_trend: 避障趋势奖励 [-e, 0]
        W_dir: 危险度权重 [0.5, 2.0]
    """
    # 直接使用体坐标系距离矩阵（无需旋转）
    octomap_matrix = octomap_body.reshape(18, 36)
    
    # ===== 1. 警戒范围内的障碍物检测 =====
    # 创建警戒区域掩码（2D范围内）
    mask_2D = octomap_matrix <= 2*D
    
    # 如果没有警戒范围内的障碍物，返回最优奖励
    if not np.any(mask_2D):
        min_dist = np.min(octomap_matrix)
        R_safe = (1 - np.exp(-(min_dist - 2*D) / D))
        return R_safe, 1.0
    # ===== 2. 计算危险度权重W_dir（基于矩阵索引推断角度） =====
    # 获取矩阵形状
    H, W = octomap_matrix.shape  # H: 俯仰角维度，W: 方位角维度
    
    # 根据索引计算角度（假设矩阵按均匀角度分布生成）
    # 获取警戒范围内的索引
    row_idx, col_idx = np.where(mask_2D)
    
    # 计算水平角（方位角）和俯仰角（基于索引位置）
    # 假设方位角范围：-180°到180°，俯仰角范围：-90°到90°
    azimuth = (col_idx / (W - 1) - 0.5) * 2 * np.pi  # 归一化到[-π, π]
    elevation = (row_idx / (H - 1) - 0.5) * np.pi  # 归一化到[-π/2, π/2]
    # 水平角权重分区
    azimuth_abs = np.abs(azimuth)
    h_counts = np.zeros(4, dtype=np.int32)
    h_counts[0] = np.sum(azimuth_abs <= np.pi/6)
    h_counts[1] = np.sum((azimuth_abs > np.pi/6) & (azimuth_abs <= np.pi/3))
    h_counts[2] = np.sum((azimuth_abs > np.pi/3) & (azimuth_abs <= np.pi/2))
    h_counts[3] = np.sum(azimuth_abs > np.pi/2)
    
    # 垂直角权重分区
    elevation_abs = np.abs(elevation)
    v_counts = np.zeros(3, dtype=np.int32)
    v_counts[0] = np.sum(elevation_abs <= np.pi/6)
    v_counts[1] = np.sum((elevation_abs > np.pi/6) & (elevation_abs <= np.pi/3))
    v_counts[2] = np.sum(elevation_abs > np.pi/3)
    
    # 计算权重（向量化）
    h_weights = np.array([1.75, 1.5, 1.25, 1.0], dtype=np.float32)
    v_weights = np.array([1.25, 1.0, 0.9], dtype=np.float32)
    
    if np.sum(h_counts) > 0:
        w_h = np.sum(h_counts * h_weights) / np.sum(h_counts)
    else:
        w_h = 1.0
    
    if np.sum(v_counts) > 0:
        w_v = np.sum(v_counts * v_weights) / np.sum(v_counts)
    else:
        w_v = 1.0
    
    W_dir = w_h * w_v
    
    # ===== 3. 计算R_trend的第一部分：距离分布奖励 =====
    # 获取警戒范围内的距离值
    octomap_in_2D = octomap_matrix[mask_2D]
    
    # 避免距离为0或负数（数值稳定性）
    octomap_in_2D_clipped = np.clip(octomap_in_2D, 0, 2*D)
    
    # 计算指数距离的加权平均
    exp_distances = 1/np.e - np.exp(1 - octomap_in_2D_clipped / D)
    avg_exp_distance = np.mean(exp_distances)
    
    # 应用权重
    R_distribution = avg_exp_distance * W_dir
    
    # ===== 4. 计算R_trend的第二部分：距离变化奖励 =====
    # 获取当前最小距离
    current_min_d = np.min(octomap_in_2D)
    
    R_change = 0.0
    if current_min_d < 2*D and prev_min_dist is not None:
        # 计算距离变化
        delta_d = current_min_d - np.clip(prev_min_dist, 0, 2*D)
        
        # 变化率奖励：靠近障碍物惩罚，远离奖励
        # 限制在[-0.5, 0.5]范围内
        if delta_d < 0:
            R_change = max(1 * delta_d / D, -1)
        else:
            R_change = min(0.5 * delta_d / D, 0.5)
    
    # ===== 5. 组合两部分奖励 =====
    # 加权组合：距离分布占70%，距离变化占30%
    R_trend = R_distribution + R_change
    
    # ===== 6. 最终调整 =====
    # 确保奖励在[-e*1.8, 0]范围内，其中e≈2.718
    R_trend = np.clip(R_trend, -4, 0.5)
    
    return float(R_trend), float(W_dir)

        
def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  # for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 获取本机ip地址
def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'

class AirSimDroneEnv(gym.Env):
    def __init__(self, uavname='UAV01', datamanager_ip='', datamanager_port=5812, trainmanager_ip='', start_point=np.array([0,0,-0.5], dtype=np.float64)):
        self.vehicle = VM.VehicleThread(uavname, stat_range=20, start_point=start_point, max_task_steps=600)
        # 多模态观察空间定义
        octomap_shape = (36, 18)
        stat_shape = (8,) # 无人机位置姿态信息等状态空间形状
        self.observation_space = spaces.Box(low=-1, high=1, shape=(36*18+8,), dtype=np.float64) 
        # 动作空间定义
        self.action_space = spaces.Box(low=-3.75, high=3.75, shape=(3,), dtype=np.float64)  # 避障速度方位角+高度角+速度大小
        self.octomap_shape = octomap_shape
        self.stat_shape = stat_shape
        self.observation_deque = deque(maxlen=10)
        self.obstacle_deque = deque(maxlen=10)
        self.last_success_times = deque(maxlen=100) # 记录最近100幕成功次数
        self.target_closer = 1.0
        self.uavname = uavname
        self.max_velocity = 7.5  # 最大速度7.5 m/s
        self.k_avoid = 0.0  # 避障权重系数
        # 解析UAV ID
        self.uav_id = int(uavname[3:])-1  # 假设uavname格式为'UAV01'，提取数字部分
        self.datamanager_ip = datamanager_ip if datamanager_ip else get_real_ip()
        self.datamanager_port = datamanager_port
        self.trainmanager_ip = trainmanager_ip if trainmanager_ip else get_real_ip()
        
        # UDP socket用于发送命令和接收数据
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind((trainmanager_ip, 2000+self.uav_id))
        self.udp_deque = deque(maxlen=25)
        # [curriculum-group variant] starts at 0.0 so the very first reward uses the
        # easy curriculum stage; TrainProgresserCal then overwrites it every step.
        self.training_progress = 0.0
        self.collision_occurred = False
        self.collision_occurred_reset = False
        self.target_reached = False
        self.outtime = False
        self.episode_steps = 0
        self.episode_length = self.vehicle.max_task_steps # 每集最大步数
        self.total_steps = 0
        self.episode_counter = 0
        self.episode_start_time = time.time()
        self.lastupdatetime = time.time()
        self.min_d = None
        self.last_distance = 1
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def set_training_progress(self, progress: float) -> None:
        """
        接收训练进度 (0.0 ~ 1.0)，并据此动态调整环境参数
        此方法会被回调函数自动调用
        """
        self.training_progress = np.clip(progress, 0.0, 1.0)

    def success_rate_cal(self):
        """
        计算最近100幕的成功率
        """
        if len(self.last_success_times) == 0:
            return 0.0
        return sum(self.last_success_times)/len(self.last_success_times)

    def _recv_loop(self):
        """
        UDP接收端，用于接收新的无人机状态
        """
        buffers = {}  # 缓存分片数据：{包序号: {片序号: 数据}}
        while True:
            packet, _ = self.udp_sock.recvfrom(MAX_PACKET_SIZE + 16)  # 预留头部空间
            pkt_id, total_pkts = struct.unpack('!II', packet[:8])
            seq = struct.unpack('!I', packet[8:12])[0]
            chunk = packet[12:]
        
            if pkt_id not in buffers:
                buffers[pkt_id] = {seq: chunk}
            else:
                buffers[pkt_id][seq] = chunk
        
            # 检查是否收齐所有分片
            if len(buffers[pkt_id]) == total_pkts:
                full_data = b''.join([buffers[pkt_id][i] for i in sorted(buffers[pkt_id])])
                new_stat = pickle.loads(full_data)
                del buffers[pkt_id]
                if new_stat['timestamp'] > self.lastupdatetime:
                    self.observation_deque.append(new_stat)
                    self.collision_occurred = new_stat['collision']

    def send_command(self, number, command, msg):
        """发送控制指令"""
        try:
            data = [command, msg]
            data = pickle.dumps(data)
            total_size = len(data)
            num_packets = (total_size+MAX_PACKET_SIZE-1)// MAX_PACKET_SIZE
            header = struct.pack('!II', number, num_packets)  # 包序号和总片数
            for seq in range(num_packets):
                start = seq * MAX_PACKET_SIZE
                chunk = data[start:start + MAX_PACKET_SIZE]
                self.udp_sock.sendto(header + struct.pack('!I', seq)+chunk, (self.datamanager_ip, self.datamanager_port+1))
                time.sleep(0.01)  # 发送间隔
        except Exception as e:
            print(f"发送指令失败: {e}")

    def reset(self, seed=None):
        """重置环境状态"""
        self.outtime = False
        self.min_d = None
        self.k_avoid = 0.0
        self.target_closer = 1.0
        self.collision_occurred = False
        self.collision_occurred_reset = False
        self.target_reached = False
        self.episode_steps = 0
        self.episode_counter += 1
        self.episode_start_time = time.time()
        self.obstacle_deque.clear()
        # 发送reset指令
        try:
            self.send_command(self.episode_counter%25+int(self.uavname[3:])*1000, "reset", [self.uavname])
        except Exception as e:
            print(f"发送reset命令失败: {e}")
        time.sleep(0.2)
        self.observation_deque.clear()
        # 等待数据中心返回观测
        self.vehicle.start_time = time.time()
        self._reset_settings()
        self.vehicle.environment_reset(modesetting='easy')
        obs = self._get_obs()
        self.last_distance = 1
        info = {
                'timeout': self.outtime,
                'collision_occurred': self.collision_occurred,
                'target_reached': self.target_reached,
            }
        return obs, info
    
    def _reset_settings(self):
        """每次重置后的第一帧数据，作为环境设置状态"""
        while len(self.observation_deque) == 0:
            time.sleep(0.01)
        init_obs = self.observation_deque.popleft()
        position, velocity, orientation, eular_angle, lidar_data, collision = init_obs['position'], init_obs['velocity'], init_obs['orientation'], init_obs['eular_angle'], init_obs['Lidar'], init_obs['collision']
        self.vehicle.stat_deque.append({
            "position": position,
            "orientation": orientation,
            "eular_angles": eular_angle,
            "velocity": velocity
        })

    def _get_obs(self):
        while len(self.observation_deque) == 0 or not self.vehicle.running:
            time.sleep(0.01)
        else:
            try:
                obs = self.observation_deque.popleft()
                self.lastupdatetime = obs['timestamp']
                position, velocity, orientation, eular_angle, lidar_data, collision = obs['position'], obs['velocity'], obs['orientation'], obs['eular_angle'], obs['Lidar'], obs['collision']
                stat, octomap = self.vehicle._Lidar_RLstat_update(position, orientation, eular_angle, velocity, lidar_data)
                return np.concatenate([octomap.reshape(-1), stat])
            except Exception as e:
                print(f'_get_obs error: {e}')
                octomap = np.ones(self.octomap_shape).flatten()
                stat = np.zeros(self.stat_shape)
                return np.concatenate([octomap, stat]).astype(np.float64)

    def _execute_action(self, action_original):
        # 解析动作
        action = action_original.copy()
        body_x, speed= self.vehicle.body_x, self.vehicle.stat_deque[-1]["velocity"]
        if self.uavname == 'UAV01' and self.total_steps %5==0:
            print(f'{self.uavname}动作输出: {action}')
        avoid_speed = np.zeros(3, dtype=np.float64)
        avoid_speed[0:3] = action[0:3]
        action_global =avoid_speed
        yaw = self.vehicle.expect_yaw*180
        target_speed = action_global + speed
        # 限幅与急停处理
        if np.linalg.norm(target_speed) >self.max_velocity:
            target_speed = target_speed / np.linalg.norm(target_speed) * self.max_velocity
        ctrl_speed = target_speed 
        # 发送速度命令
        try:
            msg = [self.uavname, np.array([ctrl_speed[0], ctrl_speed[1], ctrl_speed[2], yaw], dtype=np.float64)]
            self.send_command(self.total_steps%5+int(self.uavname[3:])*10, "action", msg)
        except Exception as e:
            print(f"发送动作命令失败: {e}")

    def _compute_reward(self, octomap_stat, stat_vector):
        """计算奖励函数"""
        reward, sparse_reward, dense_reward= 0.0, 0, 0
        current_time = time.time()
        # 能源消耗反馈
        # 生存奖励，防止无人机一直徘徊不前进
        survive_reward = -0.1 -0.1*self.training_progress
        # 避障奖励(dense reward)
        # 避障奖励取决于octomap状态空间，根据距离log值的平均进行计算
        # 障碍物奖励
        r_obstacle, W_dir = Avoid_Reward_Optimized_BodyFrame(octomap_stat, self.vehicle.stat_deque[-1]["orientation"], prev_min_dist=self.min_d, D=0.2) 
        self.min_d = np.min(octomap_stat) 
        self.k_avoid = np.exp(-self.min_d*3)*np.clip(np.exp(W_dir - 1.5), 0, 1)  # 避障权重系数
        if self.uavname == 'UAV01' and self.total_steps %5==0:
            print(f'{self.uavname}避障奖励: r_obstacle:{r_obstacle:.3f}, W_dir:{W_dir:.3f}, k_avoid:{self.k_avoid:.3f}')
        # 衡量速度突变情况(仅在vehicle.stat_deque存在历史速度信息时计算)
        if self.training_progress > 0.25:
            v_clip_reward = v_clip_reward_calc(self.vehicle.stat_deque,octomap_stat,self.vehicle.Seqspace_directions, 0.1, 'hard')
        else:
            v_clip_reward = 0.0
        # 计算速度是否朝向目标点方向
        target_dir, v_dir, v_norm = stat_vector[0:3], stat_vector[4:7], stat_vector[7]
        delta_l = np.dot(target_dir, v_dir)*v_norm*self.max_velocity  # 目标方向速度分量
        forward_reward = np.clip(delta_l/self.vehicle.max_distance*50, -1-self.training_progress, 1+self.training_progress)  # 目标距离减少奖励
        distance_reward = 0.0
        # 目标点接近奖励
        normalize_distance = self.vehicle.current_target_d_norm/self.vehicle.max_distance
        if (normalize_distance > self.last_distance) and (self.k_avoid < 0.25):
            distance_reward -= 1.0
        if normalize_distance < 1 and normalize_distance < self.target_closer-0.01:
            # distance_reward = (0.2-stat_vector[11])/0.2*2.0  # 接近目标点额外奖励
            distance_reward = 1.0  # 接近目标点额外奖励
            self.target_closer = normalize_distance
        self.last_distance = normalize_distance
        # 航向角奖励
        # expect_yaw, yaw= stat_vector[12], self.vehicle.stat_deque[-1]["eular_angles"][2]/np.pi
        # yaw_diff = (yaw-expect_yaw+1.0) % 2.0 -1.0
        # yaw_reward = 0.5-abs(yaw_diff)
        # 稠密奖励部分结束
        dense_reward = 1.0*r_obstacle + (self.training_progress-0.25)/0.75*v_clip_reward + (1.0-self.k_avoid)*forward_reward + 1.0*distance_reward + 1.0*survive_reward #+ 1.0*yaw_reward
        if self.uavname == 'UAV01' and self.total_steps %5==0:
            print(f'{self.uavname}稠密奖励: r_obstacle:{r_obstacle:.3f}, v_clip:{v_clip_reward:.3f}, forward:{forward_reward:.3f}, distance:{distance_reward:.3f}, survive:{survive_reward:.3f}, dense_reward:{dense_reward:.3f}')
            print(f'{self.uavname}训练进度: {self.training_progress:.3f}')
        reward += dense_reward
        # 目标点到达奖励
        d_norm = self.vehicle.current_target_d_norm
        if self.uavname == 'UAV01' and self.total_steps %5==0:
            print(f'{self.uavname}当前目标点距离: {d_norm:.3f} m')
        if d_norm < 5:
            print(f'{self.uavname}到达目标点!')
            sparse_reward += 50
            self.target_reached = True
            reward += sparse_reward
            return reward
        # 碰撞惩罚,触发碰撞后，会开启新一幕
        if self.collision_occurred and current_time- self.episode_start_time > 3.0:
            reward -= 80
            self.collision_occurred_reset = True
            return reward
        # 超时惩罚
        if self.episode_steps >= self.episode_length:
            reward -= 80
            self.outtime = True
            return reward
        return reward

    def _check_done(self):
        truncated, terminated = False, False
            
        # 到达目标结束
        if self.target_reached:
            terminated = True
            self.last_success_times.append(1)
            return terminated, truncated
        
        if self.outtime:
            truncated = True
            self.last_success_times.append(0)
            return terminated, truncated
        
        # 碰撞结束
        if self.collision_occurred_reset:
            terminated = True
            self.last_success_times.append(0)
            return terminated, truncated
        return terminated, truncated


    def step(self, action):
        self.episode_steps += 1
        self.total_steps += 1
        step_start_time = time.time()
        # 等待观测
        obs = self._get_obs()
        stat_vector = obs[-8:]
        octomap_stat = obs[:36*18]
        # 这里可根据obs/stat内容判断collision等
        # TODO: collision等标志位可由数据中心返回
        reward = self._compute_reward(octomap_stat, stat_vector)
        # 执行动作,其中action表示的是目标坐标系下的速度
        self._execute_action(action)
        terminated, truncated = self._check_done()
        info = {
            'timeout': self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached': self.target_reached,
        }
        time.sleep(max(0, 0.05 - (time.time() - step_start_time)))  # 确保每步至少0.05秒
        return obs, reward, terminated, truncated, info

class OctomapCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # LSTM网络处理序列数据
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 分层一：速度方向上细化高度-方位角距离矩阵
        self.octomap_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # (B,16,18,36)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  # (B,16,9,18)
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # (B,32,9,18)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  # (B,32,4,9)
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # (B,64,4,9)
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Flatten(),
            
            nn.Linear(64 * 4*9, 1024),
            nn.ELU(),

            nn.Linear(1024, 256),
            nn.ELU(),
            nn.LayerNorm(256),
        )
        
        # 分层二：无人机状态信息与任务信息序列
        self.stat_fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),

            nn.Linear(32, 64),
            nn.ELU(),

            nn.Linear(64, 96),
            nn.ELU(),
            nn.LayerNorm(96),
        )

        # 融合后全连接层加宽，激活用GELU
        self.combined_fc = nn.Sequential(
            nn.Linear(256+96, 512),
            nn.ELU(),

            nn.Linear(512, 512),
            nn.ELU(),

            nn.Linear(512, features_dim),
            nn.ELU(),
            nn.LayerNorm(features_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        obs_t = x 
            
        # 2. 拆分 OctoMap 和状态信息
        octomap_t = obs_t[:, :18*36].view(-1, 1, 18, 36)  
        stat_t = obs_t[:, -8:]                          
            
        # 3. 分别提取特征
        octomap_features_t = self.octomap_cnn(octomap_t)  
        stat_features_t = self.stat_fc(stat_t)            
            
        # 4. 融合特征
        combined_features_t = torch.cat((octomap_features_t, stat_features_t), dim=1) 
            
        # 5. 映射到最终的特征维度
        final_feature_t = self.combined_fc(combined_features_t) 
        return final_feature_t

def linear_schedule(initial_value: float, final_value: float=0):
    """
    线性衰减学习率调度器
    Args:
        initial_value: 初始学习率
        final_value: 最终学习率
    Returns:
        调度函数
    """
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

class LinearEntropyScheduler(BaseCallback):
    """
    线性调整熵系数的回调函数
    """
    def __init__(self, start_ent_coef, end_ent_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 计算训练进度比例
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        # 线性计算当前的ent_coef
        current_ent_coef = self.start_ent_coef - (self.start_ent_coef - self.end_ent_coef) * progress
        
        # 更新模型优化器中的熵系数
        self.model.ent_coef = current_ent_coef
        # 记录当前值以便监控
        self.logger.record("train/ent_coef", current_ent_coef)
        return True
    
class SuccessRateCallback(BaseCallback):
    """
    计算成功率的回调函数
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 调用每个环境的成功率计算函数，并收集结果
        rates = self.training_env.env_method('success_rate_cal')
        
        # 计算并记录成功率
        if len(rates) > 0:
            success_rate = np.mean(rates)
            self.logger.record("train/success_rate", float(success_rate))
        
        return True
    
class TrainProgresserCal(BaseCallback):
    """
    计算总体训练进度回调函数
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 计算训练进度比例
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        # 将训练进度反馈到各个训练环境中
        self.training_env.env_method('set_training_progress', progress)
        return True

def make_env(uavname):
    def _init():
        env = AirSimDroneEnv(uavname=uavname, datamanager_ip='172.21.236.121', trainmanager_ip='172.21.236.121', start_point=np.array([0,0,-0.5], dtype=np.float64))
        env = Monitor(env)  # 添加监控
        return env
    return _init

if __name__ == "__main__":
    print("正在初始化环境...")
    # 2. 定义策略参数，指定你的特征提取器
    policy_kwargs = dict(
        features_extractor_class=OctomapCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[512, 256, 256,256, 128], vf=[512,256,256,256,128]),
        activation_fn=nn.ELU
    )
    env = SubprocVecEnv([make_env(f"UAV{i+1:02d}") for i in range(10)])

    print("环境初始化完成。")

    # 设置保存模型的回调函数
    # 切换工作区域到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("当前工作目录:", os.getcwd())
    # 保存至/models/time目录下
    Train_Date = time.strftime("%Y%m%d-%H%M%S")
    total_timesteps = 1024000*5
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="../models/"+Train_Date, name_prefix=Train_Date+"model_", save_replay_buffer=True, verbose=1)
    entropy_scheduler = LinearEntropyScheduler(start_ent_coef=0.05,end_ent_coef=0.005,total_timesteps=total_timesteps)
    TrainProgresserCal_callback = TrainProgresserCal(total_timesteps=total_timesteps)
    SuccessRateCallback_callback = SuccessRateCallback()
    # [curriculum-group variant] TrainProgresserCal is registered here; this single line
    # (plus training_progress starting at 0.0) is the ONLY difference from
    # PPO_SB3_AirSim.py, the no-curriculum control group.
    callbacks = CallbackList([checkpoint_callback, entropy_scheduler, TrainProgresserCal_callback, SuccessRateCallback_callback])
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join("../models/"+Train_Date, "best_model"),
        log_path=os.path.join("../models/"+Train_Date, "eval_logs"),
        eval_freq=10_000,                
        deterministic=True,             # 评估时使用确定性策略（关闭探索）
        n_eval_episodes=2,              # 每次评估运行 2 个 episode
        verbose=1,
    )
    
    model = PPO(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        max_grad_norm=np.e,
        tensorboard_log="../logs/",
        n_steps=2560,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(initial_value=0.2,final_value=0.1),
        learning_rate=linear_schedule(initial_value=1e-4,final_value=1e-6),
        ent_coef=0.03,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # 1. 查看当前 log_std（通常是一个可学习参数）
    print("当前 log_std:", model.policy.log_std)

    # 2. 强制设置 std = 0.75 → log_std = ln(0.75)
    target_std = 1
    new_log_std = np.log(target_std)

    # 3. 直接赋值（注意：需转为 torch.Tensor 并匹配设备）
    model.policy.log_std.data = torch.tensor([new_log_std] * 3, 
                                                device=model.device, 
                                                dtype=torch.float32)
    # from mycallbacks import ParallelTrainingStatsCallback
    # parallel_stats_callback = ParallelTrainingStatsCallback(window_size=500, log_interval=100, verbose=1, tensorboard_log=model.tensorboard_log)
    # callback_list = [checkpoint_callback, eval_callback]
    # callbacks = CallbackList(callback_list)
    
    # # 添加调试信息
    # print("回调函数列表:", [type(cb).__name__ for cb in callback_list])
    # for cb in callback_list:
    #     print(f"回调函数 {type(cb).__name__} 初始化设置: verbose={cb.verbose}")
    
    # # 加载预训练模型
    # model_path = "../results/testmodel1.zip"
    # if os.path.exists(model_path):
    #     print(f"加载模型: {model_path}")
    #     model = PPO.load(model_path, env=env)
        
    #     model.n_steps = 2560
    #     model.batch_size = 256
    #     model.n_epochs = 4
    #     model.gamma = 0.975
    #     model.gae_lambda = 0.95
    #     model.ent_coef = 0.01
    #     model.max_grad_norm = np.e
    #     model.learning_rate = linear_schedule(initial_value=1e-4, final_value=1e-6)  # 使用较小的初始学习率
    #     model.clip_range = linear_schedule(initial_value=0.2, final_value=0.1)  # 保留原始模型的clip_range函数
    #     model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     model.tensorboard_log = "../logs/"

    #     # 开始学习
    #     model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)
    #     model.save(os.path.join("../models/"+Train_Date, Train_Date+"_fine_tuned_model"))

    # model = PPO.load(...)
    # model.learn(...)
    
    
    # 开始学习
    model.learn(total_timesteps=6000000, callback=callbacks, log_interval=1)
    model.save(os.path.join("../models/"+Train_Date, Train_Date+"_final_model"))

