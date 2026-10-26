"""
shield_eval_and_plot.py
=======================
评估 + 安全盾激活轨迹 + BSHM快照 四阶段可视化。

★ 标记处需要根据你的项目修改。
"""

import os, sys, time
import numpy as np
import torch
from collections import Counter, defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from stable_baselines3 import PPO

# ★ 改为你的环境文件名
from RecurrentPPO_SB3_AirSim_model import (
    AirSimDroneEnv,
    _proximity_repulsion, _extract_obstacles_from_octomap,
    _vo_safety_shield, _compute_adaptive_tau, _apply_speed_modulation,
    _rotate_global_to_body, _rotate_body_to_global,
    _REPULSION_THRESHOLD, _REPULSION_GAIN,
    _VO_ACTIVATION_D, _VO_ROBOT_RADIUS, _VO_SAFE_MARGIN,
    _ALPHA_MAX, _SPEED_FLOOR, _PLAN_AGE_MAX,
)
from shield_trajectory_visualizer import (
    ShieldTrajectoryRecorder, get_dominant_layer, LAYER_NAMES
)

# ═══════════════════════════════════════════
# ★ 配置区
# ═══════════════════════════════════════════

MODEL_PATH      = "../results/SFT_avoid_modelv3.zip"
DATAMANAGER_IP  = "172.21.236.121"
TRAINMANAGER_IP = "172.21.236.121"
START_POINT     = np.array([0, 0, -0.5], dtype=np.float64)
UAV_NAME        = "UAV01"
SAVE_DIR        = "./figures"
N_EPISODES      = 1


# ═══════════════════════════════════════════
# 带安全盾激活记录的环境
# ═══════════════════════════════════════════

class ShieldInstrumentedEnv(AirSimDroneEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_shield_state = set()

    def safe_ruler(self, min_d, delta_d, target_speed_global,
                   octomap_body=None, enable=False):
        self.last_shield_state = set()

        if not enable:
            v_norm = np.linalg.norm(target_speed_global)
            if v_norm > self.max_velocity:
                return target_speed_global * (self.max_velocity / v_norm)
            return target_speed_global

        if min_d is None:
            min_d = np.inf
        drl_speed = np.linalg.norm(target_speed_global)
        if drl_speed < 1e-9:
            return target_speed_global

        stat_range = self.vehicle.stat_range if hasattr(self.vehicle, 'stat_range') else 20.0
        tau_adaptive = _compute_adaptive_tau(
            current_speed=drl_speed, min_d_ratio=min_d,
            max_velocity=self.max_velocity, stat_range=stat_range)

        yaw_rad = self._current_yaw_rad
        drl_vel_body = _rotate_global_to_body(target_speed_global, yaw_rad)

        # Layer 0: 近距斥力场
        repulsion_applied = False
        if octomap_body is not None:
            drl_vel_body, repulsion_applied = _proximity_repulsion(
                drl_vel_body, octomap_body, stat_range=stat_range,
                danger_threshold=_REPULSION_THRESHOLD,
                force_gain=_REPULSION_GAIN, max_velocity=self.max_velocity)
            if repulsion_applied:
                self.last_shield_state.add('repulsion')

        # Layer 1: VO
        obstacle_list = []
        if octomap_body is not None:
            obstacle_list = _extract_obstacles_from_octomap(
                octomap_body, stat_range=stat_range,
                activation_dist=_VO_ACTIVATION_D)

        vo_safe_body, was_corrected, n_violations = _vo_safety_shield(
            robot_vel_body=drl_vel_body, obstacle_list=obstacle_list,
            max_velocity=self.max_velocity, robot_r=_VO_ROBOT_RADIUS,
            safe_margin=_VO_SAFE_MARGIN, tau=tau_adaptive)

        if was_corrected:
            self._vo_correction_count += 1
            self.last_shield_state.add('vo')
            result_body = vo_safe_body
        else:
            result_body = drl_vel_body

        # Layer 2: RRT*
        if was_corrected or repulsion_applied:
            result_speed = np.linalg.norm(result_body)
            if result_speed < drl_speed * 0.3 and result_speed < 1.5:
                with self._rrt_lock:
                    rrt_dir_global = self._rrt_result.get("velocity_dir", None)
                    path_found = self._rrt_result.get("path_found", False)
                    plan_age = time.time() - self._rrt_result.get("timestamp", 0.0)
                if rrt_dir_global is not None and plan_age < _PLAN_AGE_MAX:
                    self.last_shield_state.add('rrt')
                    rrt_dir_body = _rotate_global_to_body(rrt_dir_global, yaw_rad)
                    rrt_speed = drl_speed * _SPEED_FLOOR
                    blend_w = 0.6 if path_found else 0.4
                    blended = ((1.0 - blend_w) * result_body +
                               blend_w * rrt_dir_body * rrt_speed)
                    bl_norm = np.linalg.norm(blended)
                    if bl_norm > self.max_velocity:
                        blended = blended * (self.max_velocity / bl_norm)
                    elif bl_norm > 1e-9:
                        min_speed = max(drl_speed * _SPEED_FLOOR, 1.0)
                        if bl_norm < min_speed:
                            blended = blended / bl_norm * min_speed
                    result_body = blended

        safe_vel_global = _rotate_body_to_global(result_body, yaw_rad)

        # 分布偏移防护
        if was_corrected or repulsion_applied:
            drl_dir = target_speed_global / drl_speed
            safe_dir = safe_vel_global / (np.linalg.norm(safe_vel_global) + 1e-9)
            cos_sim = float(np.clip(np.dot(drl_dir, safe_dir), -1.0, 1.0))
            if cos_sim < 0.0:
                correction = safe_vel_global - target_speed_global
                safe_vel_global = target_speed_global + _ALPHA_MAX * correction
                v_norm = np.linalg.norm(safe_vel_global)
                if v_norm > self.max_velocity:
                    safe_vel_global = safe_vel_global * (self.max_velocity / v_norm)

        # 目标接近门控
        d_to_target = self.vehicle.current_target_d_norm
        if d_to_target < 30.0 and (was_corrected or repulsion_applied):
            proximity_scale = max(0.0, (d_to_target - 10.0) / 20.0)
            safe_vel_global = ((1.0 - proximity_scale) * target_speed_global +
                               proximity_scale * safe_vel_global)

        # Layer 3: 速度调制
        speed_before = np.linalg.norm(safe_vel_global)
        safe_vel_global = _apply_speed_modulation(
            safe_vel_global, min_d, self.max_velocity)
        if abs(np.linalg.norm(safe_vel_global) - speed_before) > 0.01:
            self.last_shield_state.add('speed_mod')

        v_norm = np.linalg.norm(safe_vel_global)
        if v_norm > self.max_velocity:
            safe_vel_global = safe_vel_global * (self.max_velocity / v_norm)

        return safe_vel_global


# ═══════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("初始化环境...")
    env = ShieldInstrumentedEnv(
        uavname=UAV_NAME,
        datamanager_ip=DATAMANAGER_IP,
        trainmanager_ip=TRAINMANAGER_IP,
        start_point=START_POINT,
        safe_rule=True,
    )

    print(f"加载模型: {MODEL_PATH}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(MODEL_PATH, env=env, device=device)

    recorder = ShieldTrajectoryRecorder(
        env=env, stat_range=20.0, prob_threshold=0.5,
        snapshot_stages=[0.25, 0.50, 0.75, 1.0])

    for ep in range(N_EPISODES):
        print(f"\n{'='*50}\nEpisode {ep+1}/{N_EPISODES}\n{'='*50}")

        obs, info = env.reset()
        recorder.reset()

        done = False
        step = 0
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.record(env.last_shield_state)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if step % 50 == 0:
                pos = recorder.positions[-1]
                active = env.last_shield_state or {'drl_only'}
                n_snaps = len(recorder._snapshot_taken)
                print(f"  Step {step}/{env.episode_length} | "
                      f"Reward={total_reward:.1f} | "
                      f"快照={n_snaps}/4 | "
                      f"激活={active} | "
                      f"位置=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

        recorder.force_final_snapshot()

        result = ("到达目标" if info.get('target_reached') else
                  "碰撞" if info.get('collision_occurred') else
                  "超时" if info.get('timeout') else "未知")
        print(f"\n结果: {result} | 步数: {step} | 奖励: {total_reward:.2f}")
        print(f"轨迹点: {len(recorder.positions)} | 快照: {len(recorder.snapshots)}/4")

        # 快照完整性
        for s in [0.25, 0.50, 0.75, 1.0]:
            if s in recorder.snapshots:
                snap = recorder.snapshots[s]
                print(f"  {s:.0%}: 轨迹={snap['traj_idx']}, "
                      f"BSHM={len(snap['coords'])}")
            else:
                print(f"  {s:.0%}: ❌ 缺失")

        # 全局激活统计
        all_doms = [get_dominant_layer(s) for s in recorder.shield_states]
        print("\n各层主导步数:")
        for layer, count in Counter(all_doms).most_common():
            print(f"  {LAYER_NAMES.get(layer, layer)}: "
                  f"{count}步 ({count/step*100:.1f}%)")

        # 绘图
        print("\n生成轨迹图...")
        recorder.plot(save_dir=SAVE_DIR,
                      filename=f'shield_trajectory_ep{ep+1}')
        recorder.save(os.path.join(SAVE_DIR,
                                    f'shield_trajectory_data_ep{ep+1}.npz'))

    print(f"\n完成！输出目录: {SAVE_DIR}/")


if __name__ == "__main__":
    main()