"""
SAC (Soft Actor-Critic) 训练代码
基于 PPO 版本改写，用于与 PPO/DDPG 进行对比实验
主要差异：
  - 算法换为 SAC（off-policy, 最大熵强化学习）
  - 使用 ReplayBuffer，支持异步更新
  - ent_coef 可设为 'auto' 让 SAC 自动调整，也可手动固定
  - SubprocVecEnv 仍可用于并行数据采集，但 SAC 本身只需单环境
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList, EvalCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
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
a_b_matrix = np.zeros((7, 24, 2), dtype=np.float64)
x, y = np.arange(-180, 180, 15), np.arange(-45, 50, 15)
xx, yy = np.meshgrid(x, y)
a_b_matrix[:, :, 0] = yy
a_b_matrix[:, :, 1] = xx
abmatrix = a_b_matrix / 180 * np.pi


# ───────────────────────── 奖励辅助函数（与PPO版本完全一致） ─────────────────────────

def v_clip_reward_calc(stat_deque, octomap_stat, Seqspace_directions, dt=0.1, mode='hard'):
    if len(stat_deque) < 4:
        return 0.0
    if mode == 'hard':
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        velocities = []
        for stat in stats:
            if "velocity" in stat:
                velocities.append(stat["velocity"])
        velocities = np.array(velocities)
        if len(velocities) < 4:
            return 0.0
        v_magnitudes = np.linalg.norm(velocities, axis=1)
        accels = np.gradient(v_magnitudes, dt)
        jerks = np.gradient(accels, dt) if len(accels) > 1 else np.zeros_like(accels)
        if len(velocities) >= 3:
            v_norms = np.linalg.norm(velocities, axis=1)
            valid_mask = v_norms > 0.1
            if np.sum(valid_mask) >= 2:
                valid_velocities = velocities[valid_mask]
                valid_v_norms = v_norms[valid_mask]
                directions = valid_velocities / valid_v_norms[:, np.newaxis]
                if len(directions) >= 2:
                    crosses = np.cross(directions[:-1], directions[1:])
                    sin_theta = np.linalg.norm(crosses, axis=1)
                    angular_velocities = np.arcsin(np.clip(sin_theta, 0, 1)) / dt
                    angular_penalty = np.mean(angular_velocities ** 2)
                else:
                    angular_penalty = 0.0
            else:
                angular_penalty = 0.0
        else:
            angular_penalty = 0.0
        current_speed = v_magnitudes[-1] if len(v_magnitudes) > 0 else 0.0
        max_accel = 4.0 + 6.0 * current_speed / 10.0
        max_jerk = 8.0 + 4.0 * current_speed / 10.0
        accel_excess = np.maximum(np.abs(accels) - max_accel, 0)
        accel_penalty = np.percentile(accel_excess, 75) if len(accel_excess) > 0 else 0.0
        jerk_excess = np.maximum(np.abs(jerks) - max_jerk, 0)
        jerk_penalty = np.max(jerk_excess) if len(jerk_excess) > 0 else 0.0
        smoothness_score = (np.exp(-0.1 * accel_penalty)
                            * np.exp(-0.05 * jerk_penalty)
                            * np.exp(-0.05 * angular_penalty))
        return float(np.clip(smoothness_score - 0.5, -0.5, 0.5))
    elif mode == 'easy':
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
        a_xyz_std = np.std(a_xyz, axis=0)
        v_norm = np.linalg.norm(current_speed) / 10.0
        v_norm_reward = np.exp(-(v_norm - 0.2) * (v_norm - 0.8) / 0.09 - 1) - 1
        accel_reward = np.clip(0.5 - 0.05 * np.sum(a_xyz_std), -0.5, 0.5)
        return accel_reward + v_norm_reward
    else:
        return 0.0


def Avoid_Reward_Optimized_BodyFrame(octomap_body, q, prev_min_dist=None, D=0.3334):
    octomap_matrix = octomap_body.reshape(18, 36)
    mask_2D = octomap_matrix <= 2 * D
    if not np.any(mask_2D):
        min_dist = np.min(octomap_matrix)
        R_safe = (1 - np.exp(-(min_dist - 2 * D) / D))
        return R_safe, 1.0
    H, W = octomap_matrix.shape
    row_idx, col_idx = np.where(mask_2D)
    azimuth = (col_idx / (W - 1) - 0.5) * 2 * np.pi
    elevation = (row_idx / (H - 1) - 0.5) * np.pi
    azimuth_abs = np.abs(azimuth)
    h_counts = np.zeros(4, dtype=np.int32)
    h_counts[0] = np.sum(azimuth_abs <= np.pi / 6)
    h_counts[1] = np.sum((azimuth_abs > np.pi / 6) & (azimuth_abs <= np.pi / 3))
    h_counts[2] = np.sum((azimuth_abs > np.pi / 3) & (azimuth_abs <= np.pi / 2))
    h_counts[3] = np.sum(azimuth_abs > np.pi / 2)
    elevation_abs = np.abs(elevation)
    v_counts = np.zeros(3, dtype=np.int32)
    v_counts[0] = np.sum(elevation_abs <= np.pi / 6)
    v_counts[1] = np.sum((elevation_abs > np.pi / 6) & (elevation_abs <= np.pi / 3))
    v_counts[2] = np.sum(elevation_abs > np.pi / 3)
    h_weights = np.array([1.75, 1.5, 1.25, 1.0], dtype=np.float32)
    v_weights = np.array([1.25, 1.0, 0.9], dtype=np.float32)
    w_h = np.sum(h_counts * h_weights) / np.sum(h_counts) if np.sum(h_counts) > 0 else 1.0
    w_v = np.sum(v_counts * v_weights) / np.sum(v_counts) if np.sum(v_counts) > 0 else 1.0
    W_dir = w_h * w_v
    octomap_in_2D = octomap_matrix[mask_2D]
    octomap_in_2D_clipped = np.clip(octomap_in_2D, 0, 2 * D)
    exp_distances = 1 / np.e - np.exp(1 - octomap_in_2D_clipped / D)
    avg_exp_distance = np.mean(exp_distances)
    R_distribution = avg_exp_distance * W_dir
    current_min_d = np.min(octomap_in_2D)
    R_change = 0.0
    if current_min_d < 2 * D and prev_min_dist is not None:
        delta_d = current_min_d - np.clip(prev_min_dist, 0, 2 * D)
        if delta_d < 0:
            R_change = max(1 * delta_d / D, -1)
        else:
            R_change = min(0.5 * delta_d / D, 0.5)
    R_trend = R_distribution + R_change
    R_trend = np.clip(R_trend, -4, 0.5)
    return float(R_trend), float(W_dir)


def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'


# ───────────────────────── 环境（与PPO版本完全一致） ─────────────────────────

class AirSimDroneEnv(gym.Env):
    def __init__(self, uavname='UAV01', datamanager_ip='', datamanager_port=5812,
                 trainmanager_ip='', start_point=np.array([0, 0, -0.5], dtype=np.float64)):
        self.vehicle = VM.VehicleThread(uavname, stat_range=20,
                                        start_point=start_point, max_task_steps=600)
        octomap_shape = (36, 18)
        stat_shape = (8,)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(36 * 18 + 8,), dtype=np.float64)
        self.action_space = spaces.Box(low=-3.75, high=3.75, shape=(3,), dtype=np.float64)
        self.octomap_shape = octomap_shape
        self.stat_shape = stat_shape
        self.observation_deque = deque(maxlen=10)
        self.obstacle_deque = deque(maxlen=10)
        self.last_success_times = deque(maxlen=100)
        self.target_closer = 1.0
        self.uavname = uavname
        self.max_velocity = 7.5
        self.k_avoid = 0.0
        self.uav_id = int(uavname[3:]) - 1
        self.datamanager_ip = datamanager_ip if datamanager_ip else get_real_ip()
        self.datamanager_port = datamanager_port
        self.trainmanager_ip = trainmanager_ip if trainmanager_ip else get_real_ip()
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind((trainmanager_ip, 2000 + self.uav_id))
        self.udp_deque = deque(maxlen=25)
        self.training_progress = 1.0
        self.collision_occurred = False
        self.collision_occurred_reset = False
        self.target_reached = False
        self.outtime = False
        self.episode_steps = 0
        self.episode_length = self.vehicle.max_task_steps
        self.total_steps = 0
        self.episode_counter = 0
        self.episode_start_time = time.time()
        self.lastupdatetime = time.time()
        self.min_d = None
        self.last_distance = 1
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = np.clip(progress, 0.0, 1.0)

    def success_rate_cal(self):
        if len(self.last_success_times) == 0:
            return 0.0
        return sum(self.last_success_times) / len(self.last_success_times)

    def _recv_loop(self):
        buffers = {}
        while True:
            packet, _ = self.udp_sock.recvfrom(MAX_PACKET_SIZE + 16)
            pkt_id, total_pkts = struct.unpack('!II', packet[:8])
            seq = struct.unpack('!I', packet[8:12])[0]
            chunk = packet[12:]
            if pkt_id not in buffers:
                buffers[pkt_id] = {seq: chunk}
            else:
                buffers[pkt_id][seq] = chunk
            if len(buffers[pkt_id]) == total_pkts:
                full_data = b''.join([buffers[pkt_id][i] for i in sorted(buffers[pkt_id])])
                new_stat = pickle.loads(full_data)
                del buffers[pkt_id]
                if new_stat['timestamp'] > self.lastupdatetime:
                    self.observation_deque.append(new_stat)
                    self.collision_occurred = new_stat['collision']

    def send_command(self, number, command, msg):
        try:
            data = pickle.dumps([command, msg])
            total_size = len(data)
            num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            header = struct.pack('!II', number, num_packets)
            for seq in range(num_packets):
                start = seq * MAX_PACKET_SIZE
                chunk = data[start:start + MAX_PACKET_SIZE]
                self.udp_sock.sendto(
                    header + struct.pack('!I', seq) + chunk,
                    (self.datamanager_ip, self.datamanager_port + 1)
                )
                time.sleep(0.01)
        except Exception as e:
            print(f"发送指令失败: {e}")

    def reset(self, seed=None):
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
        try:
            self.send_command(
                self.episode_counter % 25 + int(self.uavname[3:]) * 1000,
                "reset", [self.uavname]
            )
        except Exception as e:
            print(f"发送reset命令失败: {e}")
        time.sleep(0.2)
        self.observation_deque.clear()
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
        while len(self.observation_deque) == 0:
            time.sleep(0.01)
        init_obs = self.observation_deque.popleft()
        position = init_obs['position']
        velocity = init_obs['velocity']
        orientation = init_obs['orientation']
        eular_angle = init_obs['eular_angle']
        self.vehicle.stat_deque.append({
            "position": position,
            "orientation": orientation,
            "eular_angles": eular_angle,
            "velocity": velocity
        })

    def _get_obs(self):
        while len(self.observation_deque) == 0 or not self.vehicle.running:
            time.sleep(0.01)
        try:
            obs = self.observation_deque.popleft()
            self.lastupdatetime = obs['timestamp']
            position, velocity, orientation, eular_angle, lidar_data, collision = (
                obs['position'], obs['velocity'], obs['orientation'],
                obs['eular_angle'], obs['Lidar'], obs['collision']
            )
            stat, octomap = self.vehicle._Lidar_RLstat_update(
                position, orientation, eular_angle, velocity, lidar_data
            )
            return np.concatenate([octomap.reshape(-1), stat])
        except Exception as e:
            print(f'_get_obs error: {e}')
            octomap = np.ones(self.octomap_shape).flatten()
            stat = np.zeros(self.stat_shape)
            return np.concatenate([octomap, stat]).astype(np.float64)

    def _execute_action(self, action_original):
        action = action_original.copy()
        speed = self.vehicle.stat_deque[-1]["velocity"]
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}动作输出: {action}')
        avoid_speed = np.zeros(3, dtype=np.float64)
        avoid_speed[0:3] = action[0:3]
        yaw = self.vehicle.expect_yaw * 180
        target_speed = avoid_speed + speed
        if np.linalg.norm(target_speed) > self.max_velocity:
            target_speed = target_speed / np.linalg.norm(target_speed) * self.max_velocity
        ctrl_speed = target_speed
        try:
            msg = [self.uavname,
                   np.array([ctrl_speed[0], ctrl_speed[1], ctrl_speed[2], yaw], dtype=np.float64)]
            self.send_command(self.total_steps % 5 + int(self.uavname[3:]) * 10, "action", msg)
        except Exception as e:
            print(f"发送动作命令失败: {e}")

    def _compute_reward(self, octomap_stat, stat_vector):
        reward, sparse_reward, dense_reward = 0.0, 0, 0
        current_time = time.time()
        survive_reward = -0.1 - 0.1 * self.training_progress
        r_obstacle, W_dir = Avoid_Reward_Optimized_BodyFrame(
            octomap_stat, self.vehicle.stat_deque[-1]["orientation"],
            prev_min_dist=self.min_d, D=0.2
        )
        self.min_d = np.min(octomap_stat)
        self.k_avoid = np.exp(-self.min_d * 3) * np.clip(np.exp(W_dir - 1.5), 0, 1)
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}避障奖励: r_obstacle:{r_obstacle:.3f}, '
                  f'W_dir:{W_dir:.3f}, k_avoid:{self.k_avoid:.3f}')
        if self.training_progress > 0.25:
            v_clip_reward = v_clip_reward_calc(
                self.vehicle.stat_deque, octomap_stat,
                self.vehicle.Seqspace_directions, 0.1, 'hard'
            )
        else:
            v_clip_reward = 0.0
        target_dir, v_dir, v_norm = stat_vector[0:3], stat_vector[4:7], stat_vector[7]
        delta_l = np.dot(target_dir, v_dir) * v_norm * self.max_velocity
        forward_reward = np.clip(
            delta_l / self.vehicle.max_distance * 50,
            -1 - self.training_progress, 1 + self.training_progress
        )
        distance_reward = 0.0
        normalize_distance = self.vehicle.current_target_d_norm / self.vehicle.max_distance
        if (normalize_distance > self.last_distance) and (self.k_avoid < 0.25):
            distance_reward -= 1.0
        if normalize_distance < 1 and normalize_distance < self.target_closer - 0.01:
            distance_reward = 1.0
            self.target_closer = normalize_distance
        self.last_distance = normalize_distance
        dense_reward = (1.0 * r_obstacle
                        + (self.training_progress - 0.25) / 0.75 * v_clip_reward
                        + (1.0 - self.k_avoid) * forward_reward
                        + 1.0 * distance_reward
                        + 1.0 * survive_reward)
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}稠密奖励: r_obstacle:{r_obstacle:.3f}, '
                  f'v_clip:{v_clip_reward:.3f}, forward:{forward_reward:.3f}, '
                  f'distance:{distance_reward:.3f}, survive:{survive_reward:.3f}, '
                  f'dense_reward:{dense_reward:.3f}')
            print(f'{self.uavname}训练进度: {self.training_progress:.3f}')
        reward += dense_reward
        d_norm = self.vehicle.current_target_d_norm
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}当前目标点距离: {d_norm:.3f} m')
        if d_norm < 5:
            print(f'{self.uavname}到达目标点!')
            sparse_reward += 50
            self.target_reached = True
            reward += sparse_reward
            return reward
        if self.collision_occurred and current_time - self.episode_start_time > 3.0:
            reward -= 80
            self.collision_occurred_reset = True
            return reward
        if self.episode_steps >= self.episode_length:
            reward -= 80
            self.outtime = True
            return reward
        return reward

    def _check_done(self):
        truncated, terminated = False, False
        if self.target_reached:
            terminated = True
            self.last_success_times.append(1)
            return terminated, truncated
        if self.outtime:
            truncated = True
            self.last_success_times.append(0)
            return terminated, truncated
        if self.collision_occurred_reset:
            terminated = True
            self.last_success_times.append(0)
            return terminated, truncated
        return terminated, truncated

    def step(self, action):
        self.episode_steps += 1
        self.total_steps += 1
        step_start_time = time.time()
        obs = self._get_obs()
        stat_vector = obs[-8:]
        octomap_stat = obs[:36 * 18]
        reward = self._compute_reward(octomap_stat, stat_vector)
        self._execute_action(action)
        terminated, truncated = self._check_done()
        info = {
            'timeout': self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached': self.target_reached,
        }
        time.sleep(max(0, 0.05 - (time.time() - step_start_time)))
        return obs, reward, terminated, truncated, info


# ───────────────────────── 特征提取网络（与PPO版本完全一致） ─────────────────────────

class OctomapCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.octomap_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.LayerNorm(256),
        )
        self.stat_fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 96),
            nn.ELU(),
            nn.LayerNorm(96),
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 96, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, features_dim),
            nn.ELU(),
            nn.LayerNorm(features_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
        octomap_t = x[:, :18 * 36].view(-1, 1, 18, 36)
        stat_t = x[:, -8:]
        octomap_features_t = self.octomap_cnn(octomap_t)
        stat_features_t = self.stat_fc(stat_t)
        combined = torch.cat((octomap_features_t, stat_features_t), dim=1)
        return self.combined_fc(combined)


# ───────────────────────── 调度器与回调（与PPO版本一致，去掉PPO专属的熵调度） ─────────────────────────

def linear_schedule(initial_value: float, final_value: float = 0):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        rates = self.training_env.env_method('success_rate_cal')
        if len(rates) > 0:
            self.logger.record("train/success_rate", float(np.mean(rates)))
        return True


class TrainProgresserCal(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.training_env.env_method('set_training_progress', progress)
        return True


# ───────────────────────── SAC 专属：自动熵系数监控回调 ─────────────────────────

class SACEntropyMonitorCallback(BaseCallback):
    """
    监控 SAC 自动熵系数 (ent_coef) 的变化，写入 TensorBoard
    SAC 的 ent_coef 在 'auto' 模式下会自动优化，无需手动调度
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # SAC 自动熵系数存储在 model.ent_coef_tensor
        if hasattr(self.model, 'ent_coef_tensor'):
            ent_coef_val = float(self.model.ent_coef_tensor.item())
            self.logger.record("train/ent_coef", ent_coef_val)
        return True


# ───────────────────────── 环境工厂 ─────────────────────────

def make_env(uavname):
    def _init():
        env = AirSimDroneEnv(
            uavname=uavname,
            datamanager_ip='172.21.236.121',
            trainmanager_ip='172.21.236.121',
            start_point=np.array([0, 0, -0.5], dtype=np.float64)
        )
        env = Monitor(env)
        return env
    return _init


# ───────────────────────── 主训练入口 ─────────────────────────

if __name__ == "__main__":
    print("正在初始化 SAC 环境...")

    # SAC 的 policy_kwargs：actor/critic 网络结构
    # SAC 使用 'MlpPolicy'，net_arch 直接指定 actor 和 critic 各层宽度
    policy_kwargs = dict(
        features_extractor_class=OctomapCNN,
        features_extractor_kwargs=dict(features_dim=256),
        # SAC net_arch 格式：列表表示共享层 + 各自头部
        # 这里与 PPO 保持相近的参数量以保证对比公平
        net_arch=dict(pi=[512, 256, 256, 256, 128], qf=[512, 256, 256, 256, 128]),
        activation_fn=nn.ELU,
    )

    # SAC 为 off-policy 算法，单环境即可；
    # 但使用多环境可加速数据采集（SB3 SAC 支持 SubprocVecEnv）
    # 对比实验中使用与 PPO 相同的 10 并行环境
    env = SubprocVecEnv([make_env(f"UAV{i + 1:02d}") for i in range(10)])
    print("环境初始化完成。")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("当前工作目录:", os.getcwd())

    Train_Date = time.strftime("%Y%m%d-%H%M%S") + "_SAC"
    total_timesteps = 1024000 * 6

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="../models/" + Train_Date,
        name_prefix=Train_Date + "model_",
        save_replay_buffer=True,   # SAC 有 ReplayBuffer，务必保存
        verbose=1
    )
    TrainProgresserCal_callback = TrainProgresserCal(total_timesteps=1024000*5)
    SuccessRateCallback_callback = SuccessRateCallback()
    SACEntropyMonitor_callback = SACEntropyMonitorCallback()

    callbacks = CallbackList([
        checkpoint_callback,
        TrainProgresserCal_callback,
        SuccessRateCallback_callback,
        SACEntropyMonitor_callback,
    ])

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join("../models/" + Train_Date, "best_model"),
        log_path=os.path.join("../models/" + Train_Date, "eval_logs"),
        eval_freq=10_000,
        deterministic=True,
        n_eval_episodes=2,
        verbose=1,
    )

    # ── SAC 超参数说明 ──────────────────────────────────────────────────────────
    # learning_rate        : 与 PPO 一致，线性衰减 1e-4 → 1e-6
    # buffer_size          : 经验回放池大小，off-policy 核心参数
    # learning_starts      : 开始更新前先用随机策略填充 buffer 的步数
    # batch_size           : 每次从 buffer 采样的批大小
    # tau                  : 目标网络软更新系数（Polyak 系数）
    # gamma                : 折扣因子，与 PPO 保持一致
    # train_freq           : 每采集多少步更新一次
    # gradient_steps       : 每次更新执行多少梯度步（-1 = 与 train_freq 相同）
    # ent_coef='auto'      : SAC 自动调整熵系数，无需手动调度
    # target_entropy='auto': 目标熵 = -dim(action_space)
    # ────────────────────────────────────────────────────────────────────────────
    model = SAC(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="../logs/",
        learning_rate=linear_schedule(initial_value=1e-4, final_value=1e-6),
        buffer_size=500_000,          # 经验回放池（对比实验可酌情调整）
        learning_starts=10_000,       # 预热步数
        batch_size=256,
        tau=0.005,                    # 目标网络软更新系数
        gamma=0.99,                   # 与 PPO 保持一致
        train_freq=1,                 # 每步更新一次（off-policy 常用设置）
        gradient_steps=1,
        ent_coef='auto',              # SAC 自动熵系数调整
        target_entropy='auto',
        use_sde=False,                # 不使用 gSDE（与 PPO 保持一致）
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print("SAC 模型初始化完成，开始训练...")
    print(f"总训练步数: {total_timesteps}")
    print(f"模型保存路径: ../models/{Train_Date}")

    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)
    model.save(os.path.join("../models/" + Train_Date, Train_Date + "_final_model"))
    print("SAC 训练完成，模型已保存。")