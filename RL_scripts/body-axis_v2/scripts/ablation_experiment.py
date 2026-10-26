"""
ablation_experiment.py
======================
消融实验：五组安全盾配置 × 300次评估，4架无人机并行。
每组完成后保存结果到本地 JSON，全部完成后绘制学术级柱状图。

五组配置：
  0. 纯DRL（无安全盾）
  1. DRL + 近距斥力场
  2. DRL + 近距斥力场 + 自适应VO
  3. DRL + 近距斥力场 + 自适应VO + RRT*融合
  4. DRL + 近距斥力场 + 自适应VO + RRT*融合 + 速度调制（完整系统）

注意：分布偏移防护始终开启（作为基础安全机制）。

用法：
    python ablation_experiment.py

★ 标记处需要根据你的项目修改。
"""

import os, sys, time, json
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# ★ 改为你的环境文件名
from RecurrentPPO_SB3_AirSim_model import (
    AirSimDroneEnv,
    _REPULSION_THRESHOLD, _REPULSION_GAIN,
    _VO_ROBOT_RADIUS, _VO_SAFE_MARGIN,
    _SPEED_MOD_START, _SPEED_MOD_FLOOR, _EMERGENCY_SPEED, _EMERGENCY_DIST,
)

# ═══════════════════════════════════════════════════════════════
# ★ 配置区
# ═══════════════════════════════════════════════════════════════

MODEL_PATH      = "../results/SFT_avoid_modelv3.zip"   # ★ 模型路径
DATAMANAGER_IP  = "172.21.236.121"                      # ★
TRAINMANAGER_IP = "172.21.236.121"                      # ★
START_POINT     = np.array([0, 0, -0.5], dtype=np.float64)
N_ENVS          = 4                                     # 4架无人机并行
N_EVAL_EPISODES = 300                                   # 每组评估次数
SAVE_DIR        = "./ablation_results"                  # 结果保存目录
FIGURE_DIR      = "./figures"                           # 图表保存目录


# ═══════════════════════════════════════════════════════════════
# 安全盾配置：通过 monkey-patch 控制各层开关
# ═══════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    {
        "name": "纯DRL",
        "tag": "no_shield",
        "enable_repulsion": False,
        "enable_vo": False,
        "enable_rrt": False,
        "enable_speed_mod": False,
    },
    {
        "name": "DRL+斥力场",
        "tag": "repulsion_only",
        "enable_repulsion": True,
        "enable_vo": False,
        "enable_rrt": False,
        "enable_speed_mod": False,
    },
    {
        "name": "DRL+斥力场\n+自适应VO",
        "tag": "repulsion_vo",
        "enable_repulsion": True,
        "enable_vo": True,
        "enable_rrt": False,
        "enable_speed_mod": False,
    },
    {
        "name": "DRL+斥力场\n+自适应VO\n+RRT*融合",
        "tag": "repulsion_vo_rrt",
        "enable_repulsion": True,
        "enable_vo": True,
        "enable_rrt": True,
        "enable_speed_mod": False,
    },
    {
        "name": "完整系统",
        "tag": "full_system",
        "enable_repulsion": True,
        "enable_vo": True,
        "enable_rrt": True,
        "enable_speed_mod": True,
    },
]


# ═══════════════════════════════════════════════════════════════
# 带可控安全盾的环境包装器
# ═══════════════════════════════════════════════════════════════

class AblationEnv(AirSimDroneEnv):
    """
    继承原环境，通过标志位控制安全盾各层的开关。
    分布偏移防护始终开启。
    """

    def __init__(self, *args, **kwargs):
        # 先从 kwargs 中提取消融配置，避免传给父类
        self._ablation_cfg = kwargs.pop('ablation_cfg', None)
        super().__init__(*args, **kwargs)

    def safe_ruler(self, min_d, delta_d, target_speed_global,
                   octomap_body=None, enable=False):
        """重写安全盾，按消融配置逐层启用/禁用。"""
        if not enable or self._ablation_cfg is None:
            v_norm = np.linalg.norm(target_speed_global)
            if v_norm > self.max_velocity:
                return target_speed_global * (self.max_velocity / v_norm)
            return target_speed_global

        cfg = self._ablation_cfg
        if min_d is None:
            min_d = np.inf

        drl_speed = np.linalg.norm(target_speed_global)
        if drl_speed < 1e-9:
            return target_speed_global

        # ── 导入需要的函数 ──
        from RecurrentPPO_SB3_AirSim_model import (
            _compute_adaptive_tau, _rotate_global_to_body,
            _rotate_body_to_global, _proximity_repulsion,
            _extract_obstacles_from_octomap, _vo_safety_shield,
            _apply_speed_modulation, _VO_ACTIVATION_D,
            _VO_ROBOT_RADIUS, _VO_SAFE_MARGIN, _ALPHA_MAX,
            _SPEED_FLOOR, _PLAN_AGE_MAX,
        )

        stat_range = self.vehicle.stat_range if hasattr(self.vehicle, 'stat_range') else 20.0

        # Track A: 自适应 τ（VO 启用时才计算）
        if cfg["enable_vo"]:
            tau_adaptive = _compute_adaptive_tau(
                current_speed=drl_speed, min_d_ratio=min_d,
                max_velocity=self.max_velocity, stat_range=stat_range
            )
        else:
            tau_adaptive = 3.0

        yaw_rad = self._current_yaw_rad
        drl_vel_body = _rotate_global_to_body(target_speed_global, yaw_rad)

        # ── Layer 0: 近距斥力场 ──
        repulsion_applied = False
        if cfg["enable_repulsion"] and octomap_body is not None:
            drl_vel_body, repulsion_applied = _proximity_repulsion(
                drl_vel_body, octomap_body,
                stat_range=stat_range,
                danger_threshold=_REPULSION_THRESHOLD,
                force_gain=_REPULSION_GAIN,
                max_velocity=self.max_velocity,
            )

        # ── Layer 1: VO 安全盾 ──
        was_corrected = False
        n_violations = 0
        if cfg["enable_vo"] and octomap_body is not None:
            obstacle_list = _extract_obstacles_from_octomap(
                octomap_body, stat_range=stat_range,
                activation_dist=_VO_ACTIVATION_D
            )
            vo_safe_body, was_corrected, n_violations = _vo_safety_shield(
                robot_vel_body=drl_vel_body,
                obstacle_list=obstacle_list,
                max_velocity=self.max_velocity,
                robot_r=_VO_ROBOT_RADIUS,
                safe_margin=_VO_SAFE_MARGIN,
                tau=tau_adaptive,
            )
            if was_corrected:
                self._vo_correction_count += 1
                result_body = vo_safe_body
            else:
                result_body = drl_vel_body
        else:
            result_body = drl_vel_body

        # ── Layer 2: RRT* 融合 ──
        if cfg["enable_rrt"] and (was_corrected or repulsion_applied):
            result_speed = np.linalg.norm(result_body)
            if result_speed < drl_speed * 0.3 and result_speed < 1.5:
                with self._rrt_lock:
                    rrt_dir_global = self._rrt_result.get("velocity_dir", None)
                    path_found = self._rrt_result.get("path_found", False)
                    plan_age = time.time() - self._rrt_result.get("timestamp", 0.0)

                if rrt_dir_global is not None and plan_age < _PLAN_AGE_MAX:
                    rrt_dir_body = _rotate_global_to_body(rrt_dir_global, yaw_rad)
                    rrt_speed = drl_speed * _SPEED_FLOOR
                    blend_w = 0.6 if path_found else 0.4
                    blended = (1.0 - blend_w) * result_body + blend_w * rrt_dir_body * rrt_speed
                    bl_norm = np.linalg.norm(blended)
                    if bl_norm > self.max_velocity:
                        blended = blended * (self.max_velocity / bl_norm)
                    elif bl_norm > 1e-9:
                        min_speed = max(drl_speed * _SPEED_FLOOR, 1.0)
                        if bl_norm < min_speed:
                            blended = blended / bl_norm * min_speed
                    result_body = blended

        # ── 变换回全局坐标系 ──
        safe_vel_global = _rotate_body_to_global(result_body, yaw_rad)

        # ── 分布偏移防护（始终开启）──
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

        # ── 目标接近门控（始终开启）──
        d_to_target = self.vehicle.current_target_d_norm
        if d_to_target < 30.0 and (was_corrected or repulsion_applied):
            proximity_scale = max(0.0, (d_to_target - 10.0) / 20.0)
            safe_vel_global = ((1.0 - proximity_scale) * target_speed_global +
                               proximity_scale * safe_vel_global)

        # ── Layer 3: 速度调制 ──
        if cfg["enable_speed_mod"]:
            safe_vel_global = _apply_speed_modulation(
                safe_vel_global, min_d, self.max_velocity
            )

        # 速度上限
        v_norm = np.linalg.norm(safe_vel_global)
        if v_norm > self.max_velocity:
            safe_vel_global = safe_vel_global * (self.max_velocity / v_norm)

        return safe_vel_global


# ═══════════════════════════════════════════════════════════════
# 环境工厂
# ═══════════════════════════════════════════════════════════════

def make_ablation_env(uavname, ablation_cfg):
    def _init():
        env = AblationEnv(
            uavname=uavname,
            datamanager_ip=DATAMANAGER_IP,
            trainmanager_ip=TRAINMANAGER_IP,
            start_point=START_POINT,
            safe_rule=True,
            ablation_cfg=ablation_cfg,
        )
        env = Monitor(env)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════
# 单组评估
# ═══════════════════════════════════════════════════════════════

def run_single_ablation(model_path, config, n_episodes, n_envs):
    """运行单组消融实验，返回统计结果字典。"""
    print(f"\n{'='*60}")
    print(f"消融实验: {config['name'].replace(chr(10), ' ')}")
    print(f"配置: 斥力={config['enable_repulsion']}, "
          f"VO={config['enable_vo']}, "
          f"RRT*={config['enable_rrt']}, "
          f"速度调制={config['enable_speed_mod']}")
    print(f"评估次数: {n_episodes}, 并行环境: {n_envs}")
    print(f"{'='*60}")

    # 创建环境
    env = SubprocVecEnv([
        make_ablation_env(f"UAV{i+1:02d}", config)
        for i in range(n_envs)
    ])

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(model_path, env=env, device=device)

    # 评估
    from stable_baselines3.common.evaluation import evaluate_policy
    t0 = time.time()
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        render=False,
    )
    eval_time = time.time() - t0

    # 收集指标
    success_data = env.env_method('success_rate_cal')
    time_data = env.env_method('success_episode_time_hook')

    total_success = 0
    total_episodes = 0
    all_times = []

    for i in range(len(success_data)):
        if isinstance(success_data[i], tuple):
            total_success += success_data[i][0]
            total_episodes += success_data[i][1]
        else:
            total_episodes += 1

    for i in range(len(time_data)):
        if isinstance(time_data[i], tuple) and len(time_data[i]) > 0:
            times_list = time_data[i][0]
            if hasattr(times_list, '__iter__'):
                all_times.extend(list(times_list))

    success_rate = total_success / max(total_episodes, 1) * 100
    collision_rate = 100 - success_rate
    avg_time = float(np.mean(all_times)) if all_times else 0.0
    std_time = float(np.std(all_times)) if all_times else 0.0

    # VO 介入次数（从各环境收集）
    # 注意：SubprocVecEnv 不能直接访问属性，需要通过 env_method
    # 这里用 info 中的 vo_corrections 来近似

    result = {
        "config_name": config["name"].replace("\n", " "),
        "config_tag": config["tag"],
        "n_episodes": total_episodes,
        "success_count": int(total_success),
        "success_rate": round(success_rate, 2),
        "collision_rate": round(collision_rate, 2),
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "avg_time": round(avg_time, 2),
        "std_time": round(std_time, 2),
        "eval_duration_s": round(eval_time, 1),
    }

    print(f"\n结果: 成功率={success_rate:.1f}% "
          f"({int(total_success)}/{total_episodes}), "
          f"平均耗时={avg_time:.2f}±{std_time:.2f}s, "
          f"奖励={mean_reward:.2f}±{std_reward:.2f}")

    env.close()
    return result


# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════

def plot_ablation_results(results, save_dir):
    """根据消融实验结果绘制学术级柱状图。"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    try:
        import matplotlib.font_manager as fm
        fm._load_fontmanager(try_read_cache=False)
    except:
        pass

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                            'WenQuanYi Zen Hei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.dpi': 200,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.12,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    names = [r["config_name"] for r in results]
    success = [r["success_rate"] for r in results]
    collision = [r["collision_rate"] for r in results]
    avg_time = [r["avg_time"] for r in results]
    mean_reward = [r["mean_reward"] for r in results]

    x = np.arange(len(names))
    bar_w = 0.52

    # 渐进蓝色配色
    colors = ['#D5D8DC', '#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72']
    edge_colors = ['#AAB7B8', '#85C1E9', '#3498DB', '#2471A3', '#154360']

    # 碰撞率递减红色
    colors_col = ['#E74C3C', '#F1948A', '#F5B7B1', '#FADBD8', '#F9EBEA']
    edge_col = ['#922B21', '#CB4335', '#E74C3C', '#F1948A', '#F5CBA7']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.subplots_adjust(wspace=0.32)

    def style_ax(ax, ylabel, ylim=None):
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, linespacing=1.3)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--', zorder=0)
        if ylim:
            ax.set_ylim(ylim)

    def add_labels(ax, bars, fmt='{:.1f}', offset=0, fontsize=9):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold', color='#333')

    # ── (a) 避障成功率 ──
    ax = axes[0]
    bars = ax.bar(x, success, bar_w, color=colors, edgecolor=edge_colors,
                  linewidth=0.8, zorder=3)
    style_ax(ax, '避障成功率 (%)', ylim=[0, max(success) * 1.18])
    add_labels(ax, bars, fmt='{:.1f}%', offset=0.8)
    ax.set_title('(a) 避障成功率', fontsize=12, fontweight='bold', pad=12)

    # ── (b) 碰撞率 ──
    ax = axes[1]
    bars = ax.bar(x, collision, bar_w, color=colors_col, edgecolor=edge_col,
                  linewidth=0.8, zorder=3)
    style_ax(ax, '碰撞率 (%)', ylim=[0, max(collision) * 1.25 + 3])
    add_labels(ax, bars, fmt='{:.1f}%', offset=0.5)
    ax.set_title('(b) 碰撞率', fontsize=12, fontweight='bold', pad=12)

    # ── (c) 平均到达时间 ──
    ax = axes[2]
    # 纯DRL可能成功率极低，时间不具参考性
    display_time = []
    for i, t in enumerate(avg_time):
        if success[i] < 5:  # 成功率 < 5% 时时间无意义
            display_time.append(0)
        else:
            display_time.append(t)

    colors_time = ['#D5D8DC', '#82E0AA', '#58D68D', '#2ECC71', '#1E8449']
    edge_time = ['#AAB7B8', '#52BE80', '#27AE60', '#229954', '#196F3D']
    bars = ax.bar(x, display_time, bar_w, color=colors_time,
                  edgecolor=edge_time, linewidth=0.8, zorder=3)
    style_ax(ax, '平均到达时间 (s)',
             ylim=[0, max(display_time) * 1.25 + 2] if max(display_time) > 0 else [0, 50])
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if display_time[i] == 0 and success[i] < 5:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.5, 'N/A',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#999', style='italic')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}s', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#333')
    ax.set_title('(c) 成功任务平均耗时', fontsize=12, fontweight='bold', pad=12)

    # ── 保存 ──
    os.makedirs(save_dir, exist_ok=True)
    for fmt in ['png', 'pdf', 'svg']:
        path = os.path.join(save_dir, f'ablation_results.{fmt}')
        fig.savefig(path, format=fmt)
    plt.close()
    print(f"\n[绘图] 消融实验图表已保存至 {save_dir}/ablation_results.*")


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    all_results = []

    for i, config in enumerate(ABLATION_CONFIGS):
        print(f"\n\n{'#'*60}")
        print(f"# 消融实验 [{i+1}/{len(ABLATION_CONFIGS)}]: "
              f"{config['name'].replace(chr(10), ' ')}")
        print(f"{'#'*60}")

        result = run_single_ablation(
            model_path=MODEL_PATH,
            config=config,
            n_episodes=N_EVAL_EPISODES,
            n_envs=N_ENVS,
        )
        all_results.append(result)

        # 每组完成后立即保存（防止中途崩溃丢失数据）
        result_path = os.path.join(SAVE_DIR, f"ablation_{config['tag']}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[保存] 结果已保存至 {result_path}")

        # 汇总文件也实时更新
        summary_path = os.path.join(SAVE_DIR, "ablation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── 最终汇总 ──
    print(f"\n\n{'='*60}")
    print("消融实验全部完成！汇总结果：")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['config_name']:30s} | 成功率: {r['success_rate']:5.1f}% | "
              f"碰撞率: {r['collision_rate']:5.1f}% | "
              f"耗时: {r['avg_time']:.1f}±{r['std_time']:.1f}s")

    # ── 绘图 ──
    plot_ablation_results(all_results, FIGURE_DIR)

    print(f"\n全部完成！")
    print(f"  数据目录: {SAVE_DIR}/")
    print(f"  图表目录: {FIGURE_DIR}/")


# ═══════════════════════════════════════════════════════════════
# 离线绘图（从已保存的 JSON 数据重新绘图）
# ═══════════════════════════════════════════════════════════════

def plot_from_saved(summary_json_path, save_dir="./figures"):
    """
    从已保存的 ablation_summary.json 重新绘图。
    用法: python -c "from ablation_experiment import plot_from_saved; plot_from_saved('./ablation_results/ablation_summary.json')"
    """
    with open(summary_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    plot_ablation_results(results, save_dir)


if __name__ == "__main__":
    main()