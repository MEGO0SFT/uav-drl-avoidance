"""
eval_and_plot.py
================
独立评估脚本：加载模型 → 跑一个 episode → 录制轨迹 + BSHM → 生成四阶段图。

★ 标记处需要根据你的项目修改。
"""

import os, sys, time
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from stable_baselines3 import PPO
from RecurrentPPO_SB3_AirSim_model import AirSimDroneEnv                    # ★ 改为你的环境文件名
from trajectory_visualizer import TrajectoryRecorder

# ═══════════════════════════════════════════════════════════════
# ★ 配置区
# ═══════════════════════════════════════════════════════════════

MODEL_PATH      = "../results/SFT_avoid_modelv3.zip"  # ★ 模型路径
DATAMANAGER_IP  = "172.21.236.121"                     # ★
TRAINMANAGER_IP = "172.21.236.121"                     # ★
START_POINT     = np.array([0, 0, -0.5], dtype=np.float64)
UAV_NAME        = "UAV01"
SAFE_RULE       = False        # ★ 第三章纯DRL=False，第四章=True
SAVE_DIR        = "./figures"
N_EPISODES      = 1            # 录制几个 episode


def main():
    # ── 1. 环境 ──
    print("初始化环境...")
    env = AirSimDroneEnv(
        uavname=UAV_NAME,
        datamanager_ip=DATAMANAGER_IP,
        trainmanager_ip=TRAINMANAGER_IP,
        start_point=START_POINT,
        safe_rule=SAFE_RULE,
    )

    # ── 2. 模型 ──
    print(f"加载模型: {MODEL_PATH}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(MODEL_PATH, env=env, device=device)

    # ── 3. 记录器 ──
    recorder = TrajectoryRecorder(
        env=env,
        stat_range=20.0,
        prob_threshold=0.5,
        snapshot_stages=[0.25, 0.50, 0.75, 1.0],
    )

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

            recorder.record()
            total_reward += reward
            step += 1
            done = terminated or truncated

            if step % 50 == 0:
                prog = step / env.episode_length
                pos = recorder.positions[-1]
                n_snaps = len(recorder._snapshot_taken)
                print(f"  Step {step}/{env.episode_length} ({prog:.0%}) | "
                      f"Reward={total_reward:.1f} | "
                      f"已拍快照={n_snaps}/4 | "
                      f"位置=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

        # 补拍所有遗漏的阶段
        recorder.force_final_snapshot()

        # 结果
        result = ("到达目标" if info.get('target_reached') else
                  "碰撞" if info.get('collision_occurred') else
                  "超时" if info.get('timeout') else "未知")
        print(f"\n结果: {result} | 步数: {step} | 奖励: {total_reward:.2f}")
        print(f"轨迹点: {len(recorder.positions)} | "
              f"快照数: {len(recorder.snapshots)}/4")

        # 检查快照完整性
        for s in [0.25, 0.50, 0.75, 1.0]:
            if s in recorder.snapshots:
                snap = recorder.snapshots[s]
                print(f"  {s:.0%}: 轨迹截断={snap['traj_idx']}, "
                      f"BSHM体素={len(snap['coords'])}")
            else:
                print(f"  {s:.0%}: ❌ 缺失")

        # ── 绘图 ──
        print("\n生成轨迹图...")
        recorder.plot(save_dir=SAVE_DIR,
                      filename=f'trajectory_stages_ep{ep+1}')

        # ── 保存数据（离线复用）──
        recorder.save(os.path.join(SAVE_DIR,
                                    f'trajectory_data_ep{ep+1}.npz'))

    print(f"\n完成！输出目录: {SAVE_DIR}/")


if __name__ == "__main__":
    main()