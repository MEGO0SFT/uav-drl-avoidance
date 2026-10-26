"""
Monte-Carlo Evaluation Script for DRL Drone Policy
===================================================
- Runs 100 episodes with randomised start/goal points.
- Start range   (NED): x∈[0,10],   y∈[-20,20],  z∈[-40,-20]
- Goal  range   (NED): x=130,      y∈[-20,20],  z∈[-40,-10]
- Resamples while start-goal distance is outside [110, 150] m.
- Records per-step trajectories (global frame, UE4 convention Z-up).
- Writes all episodes (success + failure) to a single CSV with a
  `success` column.

This file removes the safe_ruler stack (RRT*, VO shield, repulsion,
speed modulation). Only the raw DRL policy output is executed.
"""

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os, threading, time, pickle, sys, csv
import vehicle_manager as VM
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import torch
import struct
import socket

MAX_PACKET_SIZE = 62000

# ─────────────────────────────────────────────────────────────────────────────
#  Monte-Carlo sampling configuration
# ─────────────────────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 100

# All ranges are in AirSim NED frame (Z-down).
START_X_RANGE = (0.0,    10.0)
START_Y_RANGE = (-20.0,  20.0)
START_Z_RANGE = (-40.0, -20.0)

GOAL_X_FIXED  = 130.0
GOAL_Y_RANGE  = (-20.0,  20.0)
GOAL_Z_RANGE  = (-40.0, -10.0)

DIST_MIN = 110.0   # lower bound: avoid trivial episodes
DIST_MAX = 150.0   # upper bound: as stated in the task spec

TRAJECTORY_CSV_PATH = "trajectories_mc.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  Reward helpers (kept — used for logging only, not for safe ruler)
# ─────────────────────────────────────────────────────────────────────────────

def v_clip_reward_calc(stat_deque, octomap_stat, Seqspace_directions, dt=0.1, mode='hard'):
    if len(stat_deque) < 4:
        return 0.0
    if mode == 'hard':
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        velocities = [s["velocity"] for s in stats if "velocity" in s]
        velocities = np.array(velocities)
        if len(velocities) < 4:
            return 0.0
        v_magnitudes = np.linalg.norm(velocities, axis=1)
        accels = np.gradient(v_magnitudes, dt)
        jerks  = np.gradient(accels, dt) if len(accels) > 1 else np.zeros_like(accels)
        if len(velocities) >= 3:
            v_norms    = np.linalg.norm(velocities, axis=1)
            valid_mask = v_norms > 0.1
            if np.sum(valid_mask) >= 2:
                valid_velocities = velocities[valid_mask]
                valid_v_norms    = v_norms[valid_mask]
                directions       = valid_velocities / valid_v_norms[:, np.newaxis]
                if len(directions) >= 2:
                    crosses            = np.cross(directions[:-1], directions[1:])
                    sin_theta          = np.linalg.norm(crosses, axis=1)
                    angular_velocities = np.arcsin(np.clip(sin_theta, 0, 1)) / dt
                    angular_penalty    = np.mean(angular_velocities ** 2)
                else:
                    angular_penalty = 0.0
            else:
                angular_penalty = 0.0
        else:
            angular_penalty = 0.0
        current_speed = v_magnitudes[-1] if len(v_magnitudes) > 0 else 0.0
        max_accel = 4.0 + 6.0 * current_speed / 10.0
        max_jerk  = 8.0 + 4.0 * current_speed / 10.0
        accel_excess  = np.maximum(np.abs(accels) - max_accel, 0)
        accel_penalty = np.percentile(accel_excess, 75) if len(accel_excess) > 0 else 0.0
        jerk_excess  = np.maximum(np.abs(jerks) - max_jerk, 0)
        jerk_penalty = np.max(jerk_excess) if len(jerk_excess) > 0 else 0.0
        smoothness_score = (np.exp(-0.1 * accel_penalty) *
                            np.exp(-0.05 * jerk_penalty) *
                            np.exp(-0.05 * angular_penalty))
        return float(np.clip(smoothness_score - 0.5, -0.5, 0.5))
    elif mode == 'easy':
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        velocities = [s["velocity"] for s in stats if "velocity" in s]
        current_speed = stat_deque[-1]["velocity"]
        velocities    = np.array(velocities)
        if len(velocities) < 2:
            return 0.0
        a_xyz     = np.gradient(velocities, dt, axis=0)
        a_xyz_std = np.std(a_xyz, axis=0)
        v_norm        = np.linalg.norm(current_speed) / 10.0
        v_norm_reward = np.exp(-(v_norm - 0.2) * (v_norm - 0.8) / 0.09 - 1) - 1
        accel_reward  = np.clip(0.5 - 0.05 * np.sum(a_xyz_std), -0.5, 0.5)
        return accel_reward + v_norm_reward
    else:
        return 0.0


def Avoid_Reward_Optimized_BodyFrame(octomap_body, q, prev_min_dist=None, D=0.3334):
    octomap_matrix = octomap_body.reshape(18, 36)
    mask_2D = octomap_matrix <= 2 * D
    if not np.any(mask_2D):
        min_dist = np.min(octomap_matrix)
        R_safe   = (1 - np.exp(-(min_dist - 2 * D) / D))
        return R_safe, 1.0, 0.0
    H, W = octomap_matrix.shape
    row_idx, col_idx = np.where(mask_2D)
    azimuth   = (col_idx / (W - 1) - 0.5) * 2 * np.pi
    elevation = (row_idx / (H - 1) - 0.5) * np.pi
    azimuth_abs = np.abs(azimuth)
    h_counts    = np.zeros(4, dtype=np.int32)
    h_counts[0] = np.sum(azimuth_abs <= np.pi / 6)
    h_counts[1] = np.sum((azimuth_abs > np.pi / 6)  & (azimuth_abs <= np.pi / 3))
    h_counts[2] = np.sum((azimuth_abs > np.pi / 3)  & (azimuth_abs <= np.pi / 2))
    h_counts[3] = np.sum(azimuth_abs > np.pi / 2)
    elevation_abs = np.abs(elevation)
    v_counts      = np.zeros(3, dtype=np.int32)
    v_counts[0]   = np.sum(elevation_abs <= np.pi / 6)
    v_counts[1]   = np.sum((elevation_abs > np.pi / 6) & (elevation_abs <= np.pi / 3))
    v_counts[2]   = np.sum(elevation_abs > np.pi / 3)
    h_weights = np.array([1.75, 1.5, 1.25, 1.0], dtype=np.float32)
    v_weights = np.array([1.25, 1.0, 0.9],        dtype=np.float32)
    w_h = np.sum(h_counts * h_weights) / np.sum(h_counts) if np.sum(h_counts) > 0 else 1.0
    w_v = np.sum(v_counts * v_weights) / np.sum(v_counts) if np.sum(v_counts) > 0 else 1.0
    W_dir = w_h * w_v
    octomap_in_2D         = octomap_matrix[mask_2D]
    octomap_in_2D_clipped = np.clip(octomap_in_2D, 0, 2 * D)
    exp_distances         = 1 / np.e - np.exp(1 - octomap_in_2D_clipped / D)
    avg_exp_distance      = np.mean(exp_distances)
    R_distribution        = avg_exp_distance * W_dir
    current_min_d = np.min(octomap_in_2D)
    R_change, delta_d = 0.0, 0.0
    if current_min_d < 2 * D and prev_min_dist is not None:
        delta_d = current_min_d - np.clip(prev_min_dist, 0, 2 * D)
        if delta_d < 0:
            R_change = max(1 * delta_d / D, -1)
        else:
            R_change = min(0.5 * delta_d / D, 0.5)
    R_trend = R_distribution + R_change
    R_trend = np.clip(R_trend, -4, 0.5)
    return float(R_trend), float(W_dir), float(delta_d)


def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'


# ─────────────────────────────────────────────────────────────────────────────
#  Monte-Carlo start/goal sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_start_goal(rng=None, max_tries=1000):
    """
    Sample a (start, goal) pair in AirSim NED frame with the constraint
    DIST_MIN <= ||goal - start|| <= DIST_MAX.

    Returns
    -------
    start_ned, goal_ned : (3,) np.ndarray (float64)
    """
    if rng is None:
        rng = np.random
    for _ in range(max_tries):
        sx = rng.uniform(*START_X_RANGE)
        sy = rng.uniform(*START_Y_RANGE)
        sz = rng.uniform(*START_Z_RANGE)
        gx = GOAL_X_FIXED
        gy = rng.uniform(*GOAL_Y_RANGE)
        gz = rng.uniform(*GOAL_Z_RANGE)
        start = np.array([sx, sy, sz], dtype=np.float64)
        goal  = np.array([gx, gy, gz], dtype=np.float64)
        if DIST_MIN <= np.linalg.norm(goal - start) <= DIST_MAX:
            return start, goal
    raise RuntimeError("Failed to sample a valid start/goal pair within "
                       f"{max_tries} attempts.")


def ned_to_ue4(p_ned):
    """Convert a position from AirSim NED (Z-down) to UE4-style right-handed
    Z-up convention used for plotting. X and Y are preserved; Z flips sign."""
    p = np.asarray(p_ned, dtype=np.float64)
    return np.array([p[0], p[1], -p[2]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  Environment (safe_ruler removed)
# ─────────────────────────────────────────────────────────────────────────────

class AirSimDroneEnv(gym.Env):
    def __init__(self, uavname='UAV01', datamanager_ip='', datamanager_port=5812,
                 trainmanager_ip='', start_point=np.array([0, 0, -0.5], dtype=np.float64)):
        self.vehicle = VM.VehicleThread(uavname, stat_range=20, start_point=start_point,
                                        max_task_steps=600)
        octomap_shape = (36, 18)
        stat_shape    = (8,)

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(36 * 18 + 8,), dtype=np.float64)
        self.action_space = spaces.Box(low=-3.75, high=3.75,
                                       shape=(3,), dtype=np.float64)
        self.octomap_shape   = octomap_shape
        self.stat_shape      = stat_shape
        self.observation_deque         = deque(maxlen=10)
        self.obstacle_deque            = deque(maxlen=10)
        self.last_success_times        = deque(maxlen=300)
        self.last_success_episode_time = deque(maxlen=300)
        self.target_closer    = 1.0
        self.uavname          = uavname
        self.max_velocity     = 7.5
        self.k_avoid          = 0.0
        self.uav_id           = int(uavname[3:]) - 1
        self.datamanager_ip   = datamanager_ip  if datamanager_ip  else get_real_ip()
        self.datamanager_port = datamanager_port
        self.trainmanager_ip  = trainmanager_ip if trainmanager_ip else get_real_ip()

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind((trainmanager_ip, 2000 + self.uav_id))
        self.udp_deque          = deque(maxlen=25)
        self.training_progress  = 1.0
        self.pause_enable       = True
        self.collision_occurred = False
        self.collision_occurred_reset = False
        self.target_reached     = False
        self.outtime            = False
        self.episode_steps      = 0
        self.episode_length     = self.vehicle.max_task_steps
        self.total_steps        = 0
        self.episode_counter    = 0
        self.episode_start_time = time.time()
        self.pause_time         = self.episode_start_time
        self.lastupdatetime     = time.time()
        self.min_d              = None
        self.delta_d            = 0.0
        self.last_distance      = 1

        # ─── Monte-Carlo trajectory recording state ────────────────────────
        # Each element of self.trajectories is a dict:
        #   {"episode_id": int, "success": 0/1, "start_ned": (3,), "goal_ned": (3,),
        #    "points_ned": list of (t, x, y, z)}
        self.trajectories            = []
        self._current_traj_points    = []   # list[(t_rel, x, y, z)] in NED
        self._current_start_ned      = None
        self._current_goal_ned       = None
        self._episode_index          = -1    # zero-based

        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = progress

    # =========================================================================
    #  Monte-Carlo target override
    # =========================================================================

    def _apply_mc_start_goal(self, start_ned, goal_ned):
        """
        Override the vehicle's target (and effective starting offset) so that
        the current episode uses the specified (start, goal) in NED frame.

        The vehicle's internal `pos` offset is kept at the value supplied to
        __init__; the AirSim side is expected to respawn the UAV at
        start_ned (handled by the datamanager when the 'reset' command is
        received — this has been configured upstream).
        """
        self.vehicle.target       = goal_ned.astype(np.float64)
        self.vehicle.max_distance = 150.0
        self.vehicle.running      = True

        # Store for CSV dumping
        self._current_start_ned = start_ned.astype(np.float64)
        self._current_goal_ned  = goal_ned.astype(np.float64)

        print(f"[MC] episode={self._episode_index}  "
              f"start={np.round(start_ned,2)}  goal={np.round(goal_ned,2)}  "
              f"dist={np.linalg.norm(goal_ned-start_ned):.2f} m")

    # =========================================================================
    #  Statistics helpers
    # =========================================================================

    def success_rate_cal(self):
        if len(self.last_success_times) == 0:
            return 0.0
        return sum(self.last_success_times), len(self.last_success_times)

    def success_episode_time_hook(self):
        if len(self.last_success_episode_time) == 0:
            return 0.0
        return self.last_success_episode_time, len(self.last_success_episode_time)

    def get_trajectories(self):
        """Return the list of recorded trajectories (called via env_method)."""
        return self.trajectories

    # =========================================================================
    #  UDP receive thread
    # =========================================================================

    def _recv_loop(self):
        buffers = {}
        while True:
            packet, _ = self.udp_sock.recvfrom(MAX_PACKET_SIZE + 16)
            pkt_id, total_pkts = struct.unpack('!II', packet[:8])
            seq   = struct.unpack('!I', packet[8:12])[0]
            chunk = packet[12:]
            if pkt_id not in buffers:
                buffers[pkt_id] = {seq: chunk}
            else:
                buffers[pkt_id][seq] = chunk
            if len(buffers[pkt_id]) == total_pkts:
                full_data = b''.join([buffers[pkt_id][i] for i in sorted(buffers[pkt_id])])
                new_stat  = pickle.loads(full_data)
                del buffers[pkt_id]
                if new_stat['timestamp'] > self.lastupdatetime:
                    self.observation_deque.append(new_stat)
                    self.collision_occurred = new_stat['collision']

    def send_command(self, number, command, msg):
        try:
            data        = pickle.dumps([command, msg])
            total_size  = len(data)
            num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            header      = struct.pack('!II', number, num_packets)
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

    # =========================================================================
    #  Gym interface
    # =========================================================================

    def reset(self, seed=None):
        # Finalise the previous trajectory before starting a new one
        if self._current_start_ned is not None and len(self._current_traj_points) > 0:
            self.trajectories.append({
                "episode_id": self._episode_index,
                "success":    int(self.target_reached),
                "start_ned":  self._current_start_ned.copy(),
                "goal_ned":   self._current_goal_ned.copy(),
                "points_ned": list(self._current_traj_points),
            })
        self._current_traj_points = []

        self._episode_index += 1

        # ─── Monte-Carlo sample start & goal in NED ──────────────────────
        start_ned, goal_ned = sample_start_goal()

        self.outtime               = False
        self.min_d                 = None
        self.k_avoid               = 0.0
        self.pause_enable          = True
        self.target_closer         = 1.0
        self.collision_occurred    = False
        self.delta_d               = 0.0
        self.collision_occurred_reset = False
        self.target_reached        = False
        self.episode_steps         = 0
        self.episode_counter      += 1
        self.episode_start_time    = time.time()
        self.pause_time            = self.episode_start_time
        self.obstacle_deque.clear()

        try:
            # The upstream datamanager is expected to interpret msg[1] as the
            # NED respawn position when present. If your datamanager expects
            # a different packet shape, adjust the payload below.
            self.send_command(
                self.episode_counter % 25 + int(self.uavname[3:]) * 1000,
                "reset",
                [self.uavname, start_ned.tolist()]
            )
        except Exception as e:
            print(f"发送reset命令失败: {e}")
        time.sleep(0.2)
        self.observation_deque.clear()
        self.vehicle.start_time = time.time()
        self._reset_settings()

        # Override Target_Setting with our MC goal
        self.vehicle.environment_reset(modesetting='easy')
        self._apply_mc_start_goal(start_ned, goal_ned)

        obs = self._get_obs()
        self.last_distance = 1
        info = {
            'timeout':            self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached':     self.target_reached,
            'episode_id':         self._episode_index,
            'start_ned':          start_ned.tolist(),
            'goal_ned':           goal_ned.tolist(),
        }
        return obs, info

    def _reset_settings(self):
        while len(self.observation_deque) == 0:
            time.sleep(0.01)
        init_obs = self.observation_deque.popleft()
        position, velocity, orientation, eular_angle, lidar_data, collision = (
            init_obs['position'], init_obs['velocity'], init_obs['orientation'],
            init_obs['eular_angle'], init_obs['Lidar'], init_obs['collision']
        )
        self.vehicle.stat_deque.append({
            "position":     position,
            "orientation":  orientation,
            "eular_angles": eular_angle,
            "velocity":     velocity,
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

            # ─── Trajectory logging — global NED position ────────────────
            # position is the vehicle frame offset from self.vehicle.pos, so
            # the absolute NED position is position + self.vehicle.pos.
            global_pos_ned = position + self.vehicle.pos
            t_rel = time.time() - self.episode_start_time
            self._current_traj_points.append((
                float(t_rel),
                float(global_pos_ned[0]),
                float(global_pos_ned[1]),
                float(global_pos_ned[2]),
            ))

            return np.concatenate([octomap.reshape(-1), stat])
        except Exception as e:
            print(f'_get_obs error: {e}')
            octomap = np.ones(self.octomap_shape).flatten()
            stat    = np.zeros(self.stat_shape)
            return np.concatenate([octomap, stat]).astype(np.float64)

    def _execute_action(self, action_original, octomap_stat):
        """Execute raw DRL action — no safety shield. Matches the user's
        original design exactly: the velocity stored in stat_deque is NOT
        overwritten with the commanded velocity (it stays as the last
        sampled physical velocity from AirSim, used as the base for the
        next action's delta-v and for smoothness rewards)."""
        action = action_original.copy()
        body_x = self.vehicle.body_x
        speed  = self.vehicle.stat_deque[-1]["velocity"]

        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}动作输出: {action}')

        avoid_speed      = np.zeros(3, dtype=np.float64)
        avoid_speed[0:3] = action[0:3]
        action_global    = avoid_speed
        yaw              = self.vehicle.expect_yaw * 180

        target_speed = action_global + speed
        if np.linalg.norm(target_speed) > self.max_velocity:
            target_speed = target_speed / np.linalg.norm(target_speed) * self.max_velocity
        ctrl_speed = target_speed

        try:
            msg = [self.uavname,
                   np.array([ctrl_speed[0], ctrl_speed[1], ctrl_speed[2], yaw],
                            dtype=np.float64)]
            self.send_command(
                self.total_steps % 5 + int(self.uavname[3:]) * 10, "action", msg
            )
        except Exception as e:
            print(f"发送动作命令失败: {e}")

    def _compute_reward(self, octomap_stat, stat_vector):
        reward, sparse_reward = 0.0, 0
        current_time   = time.time()
        survive_reward = -0.1 - 0.1 * self.training_progress

        r_obstacle, W_dir, delta_d = Avoid_Reward_Optimized_BodyFrame(
            octomap_stat,
            self.vehicle.stat_deque[-1]["orientation"],
            prev_min_dist=self.min_d,
            D=0.2
        )
        self.min_d   = np.min(octomap_stat)
        self.delta_d = delta_d
        self.k_avoid = (np.exp(-self.min_d * 3) *
                        np.clip(np.exp(W_dir - 1.5), 0, 1))

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
        delta_l        = np.dot(target_dir, v_dir) * v_norm * self.max_velocity
        forward_reward = np.clip(
            delta_l / self.vehicle.max_distance * 50,
            -1 - self.training_progress, 1 + self.training_progress
        )

        distance_reward    = 0.0
        normalize_distance = (self.vehicle.current_target_d_norm /
                              self.vehicle.max_distance)
        if (normalize_distance > self.last_distance) and (self.k_avoid < 0.25):
            distance_reward -= 1.0
        if normalize_distance < 1 and normalize_distance < self.target_closer - 0.01:
            distance_reward    = 1.0
            self.target_closer = normalize_distance
        self.last_distance = normalize_distance

        dense_reward = (
            1.0 * r_obstacle +
            (self.training_progress - 0.25) / 0.75 * v_clip_reward +
            (1.0 - self.k_avoid) * forward_reward +
            1.0 * distance_reward +
            1.0 * survive_reward
        )
        reward += dense_reward

        d_norm = self.vehicle.current_target_d_norm
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}当前目标点距离: {d_norm:.3f} m')

        if d_norm < 5:
            print(f'{self.uavname}到达目标点!')
            sparse_reward      += 50
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
            self.last_success_episode_time.append(time.time() - self.episode_start_time)
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
        self.total_steps   += 1
        time_step_start     = time.time()

        obs          = self._get_obs()
        stat_vector  = obs[-8:]
        octomap_stat = obs[:36 * 18]

        self._execute_action(action, octomap_stat)
        reward = self._compute_reward(octomap_stat, stat_vector)

        terminated, truncated = self._check_done()
        info = {
            'timeout':            self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached':     self.target_reached,
        }

        # ─── Finalise trajectory when episode ends ───────────────────────
        if (terminated or truncated) and self._current_start_ned is not None:
            self.trajectories.append({
                "episode_id": self._episode_index,
                "success":    int(self.target_reached),
                "start_ned":  self._current_start_ned.copy(),
                "goal_ned":   self._current_goal_ned.copy(),
                "points_ned": list(self._current_traj_points),
            })
            # Prevent double-append on next reset
            self._current_start_ned   = None
            self._current_goal_ned    = None
            self._current_traj_points = []

        time_step_end = time.time()
        time.sleep(max(0.0, 0.05 - (time_step_end - time_step_start)))
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
#  CSV writer — converts NED → UE4 Z-up frame for plotting
# ─────────────────────────────────────────────────────────────────────────────

def dump_trajectories_to_csv(trajectories, csv_path):
    """
    Write ALL episodes (success + failure) into a long-format CSV.

    Columns
    -------
    episode_id, step, t_rel, success,
    x, y, z,                         # UE4 frame (Z-up), global (origin = PlayerStart)
    start_x, start_y, start_z,       # episode start, UE4 frame
    goal_x,  goal_y,  goal_z         # episode goal,  UE4 frame
    """
    n_total   = len(trajectories)
    n_success = sum(1 for t in trajectories if t["success"] == 1)
    print(f"\n[CSV] Writing {n_total} episodes "
          f"({n_success} success, {n_total - n_success} failure) to {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "step", "t_rel", "success",
            "x", "y", "z",
            "start_x", "start_y", "start_z",
            "goal_x",  "goal_y",  "goal_z",
        ])

        for traj in trajectories:
            start_ue4 = ned_to_ue4(traj["start_ned"])
            goal_ue4  = ned_to_ue4(traj["goal_ned"])

            for step_idx, (t_rel, x, y, z) in enumerate(traj["points_ned"]):
                pt_ue4 = ned_to_ue4(np.array([x, y, z]))
                writer.writerow([
                    traj["episode_id"],
                    step_idx,
                    f"{t_rel:.4f}",
                    traj["success"],
                    f"{pt_ue4[0]:.4f}", f"{pt_ue4[1]:.4f}", f"{pt_ue4[2]:.4f}",
                    f"{start_ue4[0]:.4f}", f"{start_ue4[1]:.4f}", f"{start_ue4[2]:.4f}",
                    f"{goal_ue4[0]:.4f}",  f"{goal_ue4[1]:.4f}",  f"{goal_ue4[2]:.4f}",
                ])

    print(f"[CSV] Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def make_env(uavname):
    def _init():
        env = AirSimDroneEnv(
            uavname=uavname,
            datamanager_ip='172.21.236.121',
            trainmanager_ip='172.21.236.121',
            start_point=np.array([0, 0, -0.5], dtype=np.float64),
        )
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    print("正在初始化环境...")
    env = SubprocVecEnv([make_env(f"UAV{i+1:02d}") for i in range(1)])
    print("环境初始化完成。")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("当前工作目录:", os.getcwd())

    model_path = "../results/SFT_avoid_modelv3.zip"
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = PPO.load(model_path, env=env)
        target_std  = 0.25
        new_log_std = np.log(target_std)
        model.policy.log_std.data = torch.tensor(
            [new_log_std] * 3,
            device=model.device,
            dtype=torch.float32
        )
    else:
        print(f"模型文件不存在: {model_path}")
        sys.exit(1)

    from stable_baselines3.common.evaluation import evaluate_policy
    print(f"开始蒙特卡洛评估 ({N_EVAL_EPISODES} episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # ─── Aggregate statistics ────────────────────────────────────────────
    success_msg       = env.env_method('success_rate_cal')
    success_time_used = env.env_method('success_episode_time_hook')

    success_times, total_times = 0.0, 0
    total_success_time         = np.array([])
    for i in range(len(success_msg)):
        success_times     += success_msg[i][0]
        total_times       += success_msg[i][1]
        total_success_time = np.concatenate((total_success_time,
                                              success_time_used[i][0]))

    if total_times > 0:
        print(f"Overall Success Rate: {success_times}/{total_times} "
              f"= {success_times / total_times:.2%}")
    if len(total_success_time) > 0:
        print(f"Average Time for Successful Episodes: "
              f"{np.average(total_success_time):.4f} seconds")
        print(f"Std Time for Successful Episodes: "
              f"{np.std(total_success_time):.4f} seconds")

    # ─── Save trajectories ───────────────────────────────────────────────
    all_trajectories = []
    for traj_list in env.env_method('get_trajectories'):
        all_trajectories.extend(traj_list)

    dump_trajectories_to_csv(all_trajectories, TRAJECTORY_CSV_PATH)