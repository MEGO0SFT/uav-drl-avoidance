"""
shield_trajectory_visualizer.py
===============================
记录每步安全盾激活状态 + BSHM快照，生成四阶段（25%/50%/75%/100%）
混合决策策略轨迹段示意图。

每段轨迹按主导安全盾层着色，叠加BSHM体素，仅显示已执行轨迹。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from typing import Optional, List
from collections import defaultdict


def _setup_matplotlib():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                            'Microsoft YaHei', 'Arial', 'DejaVu Sans',
                            'Noto Sans CJK SC'],
        'axes.unicode_minus': False,
        'figure.dpi': 200,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# ═══════════════════════════════════════════
# 安全盾层定义与配色
# ═══════════════════════════════════════════

LAYER_NAMES = {
    'drl_only':     '纯DRL决策',
    'repulsion':    '近距斥力场',
    'vo':           '自适应VO',
    'rrt':          'RRT*融合',
    'speed_mod':    '速度调制',
}

LAYER_STYLE = {
    'drl_only':  {'color': '#2980B9', 'priority': 0, 'lw': 2.5},
    'repulsion': {'color': '#E67E22', 'priority': 1, 'lw': 3.0},
    'vo':        {'color': '#E74C3C', 'priority': 3, 'lw': 3.0},
    'rrt':       {'color': '#8E44AD', 'priority': 4, 'lw': 3.5},
    'speed_mod': {'color': '#27AE60', 'priority': 2, 'lw': 2.8},
}


def get_dominant_layer(active_layers: set) -> str:
    if not active_layers:
        return 'drl_only'
    return max(active_layers, key=lambda k: LAYER_STYLE[k]['priority'])


# ═══════════════════════════════════════════
# 记录器
# ═══════════════════════════════════════════

class ShieldTrajectoryRecorder:
    """
    记录每步位置、安全盾激活状态，并在阶段进度时拍摄BSHM快照。
    """

    def __init__(self, env, stat_range=20.0, prob_threshold=0.5,
                 snapshot_stages=None):
        self.env = env
        self.vehicle = env.vehicle
        self.stat_range = stat_range
        self.prob_threshold = prob_threshold
        self.snapshot_stages = snapshot_stages or [0.25, 0.50, 0.75, 1.0]

        self.positions: List[np.ndarray] = []
        self.shield_states: List[set] = []
        self.start_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.total_distance_to_target: float = 1.0

        # {stage: {'coords','probs','traj_idx','shield_states_snapshot'}}
        self.snapshots: dict = {}
        self._snapshot_taken: set = set()

    def reset(self):
        self.positions = []
        self.shield_states = []
        self.snapshots = {}
        self._snapshot_taken = set()

        self.start_pos = self.vehicle.pos.copy()
        if len(self.vehicle.stat_deque) > 0:
            self.start_pos = (self.vehicle.pos +
                              self.vehicle.stat_deque[-1]['position']).copy()
        self.target_pos = self.vehicle.target.copy()
        self.total_distance_to_target = max(
            np.linalg.norm(self.target_pos - self.start_pos), 1.0)

    def record(self, active_layers: set):
        if len(self.vehicle.stat_deque) > 0:
            pos = (self.vehicle.pos +
                   self.vehicle.stat_deque[-1]['position']).copy()
        else:
            pos = self.vehicle.pos.copy()
        self.positions.append(pos)
        self.shield_states.append(active_layers.copy() if active_layers else set())

        # 计算飞行进度
        flown = (np.sum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1))
                 if len(self.positions) > 1 else 0.0)
        progress = min(flown / self.total_distance_to_target, 1.5)
        dist_remaining = np.linalg.norm(self.target_pos - pos)
        approach_progress = 1.0 - dist_remaining / self.total_distance_to_target
        effective_progress =  0.7 * approach_progress + 0.3 * min(progress, 1.0)

        for stage in self.snapshot_stages:
            if stage not in self._snapshot_taken and effective_progress >= stage:
                self._take_snapshot(stage)

    def force_final_snapshot(self):
        for stage in sorted(self.snapshot_stages):
            if stage not in self._snapshot_taken:
                self._take_snapshot(stage)

    def _take_snapshot(self, stage: float):
        self._snapshot_taken.add(stage)
        traj_idx = len(self.positions)

        coords = np.zeros((0, 3))
        probs = np.array([])

        try:
            octo = self.vehicle.OctoMaps
            g_min = octo.global_min_bound
            g_max = octo.global_max_bound

            if g_min is not None and g_max is not None:
                margin = np.array([5.0, 5.0, 5.0])
                _, c, p = octo.get_region_probabilities(
                    g_min - margin, g_max + margin, return_coords=True)
            else:
                traj_arr = np.array(self.positions)
                bbox_min = traj_arr.min(axis=0) - self.stat_range * 2
                bbox_max = traj_arr.max(axis=0) + self.stat_range * 2
                _, c, p = octo.get_region_probabilities(
                    bbox_min, bbox_max, return_coords=True)

            if len(c) > 0:
                mask = p >= self.prob_threshold
                coords = c[mask].copy()
                probs = p[mask].copy()
        except Exception as e:
            print(f"[ShieldRecorder] BSHM快照失败 (stage={stage:.0%}): {e}")

        self.snapshots[stage] = {
            'coords': coords,
            'probs': probs,
            'traj_idx': traj_idx,
        }
        print(f"[ShieldRecorder] 阶段 {stage:.0%}: "
              f"轨迹点={traj_idx}, BSHM体素={len(coords)}")

    # ── 持久化 ──

    def save(self, filepath='shield_trajectory_data.npz'):
        layer_keys = ['repulsion', 'vo', 'rrt', 'speed_mod']
        n = len(self.shield_states)
        matrix = np.zeros((n, len(layer_keys)), dtype=np.int8)
        for i, layers in enumerate(self.shield_states):
            for j, key in enumerate(layer_keys):
                if key in layers:
                    matrix[i, j] = 1

        save_dict = {
            'positions': np.array(self.positions),
            'shield_matrix': matrix,
            'layer_keys': np.array(layer_keys),
            'start_pos': self.start_pos,
            'target_pos': self.target_pos,
            'stages': np.array(sorted(self.snapshots.keys())),
        }
        for stage, snap in self.snapshots.items():
            tag = f'{stage:.2f}'
            save_dict[f'coords_{tag}'] = snap['coords']
            save_dict[f'probs_{tag}'] = snap['probs']
            save_dict[f'traj_idx_{tag}'] = np.array([snap['traj_idx']])

        np.savez_compressed(filepath, **save_dict)
        print(f"[ShieldRecorder] 数据已保存: {filepath}")

    @staticmethod
    def load(filepath):
        d = np.load(filepath, allow_pickle=True)
        layer_keys = list(d['layer_keys'])
        matrix = d['shield_matrix']
        shield_states = []
        for i in range(len(matrix)):
            layers = set()
            for j, key in enumerate(layer_keys):
                if matrix[i, j]:
                    layers.add(key)
            shield_states.append(layers)

        snapshots = {}
        for stage in d['stages']:
            tag = f'{stage:.2f}'
            snapshots[float(stage)] = {
                'coords': d[f'coords_{tag}'],
                'probs': d[f'probs_{tag}'],
                'traj_idx': int(d[f'traj_idx_{tag}'][0]),
            }

        return {
            'positions': d['positions'],
            'shield_states': shield_states,
            'start_pos': d['start_pos'],
            'target_pos': d['target_pos'],
            'snapshots': snapshots,
        }

    # ── 绘图 ──

    def plot(self, save_dir='.', filename='shield_trajectory_stages'):
        data = {
            'positions': np.array(self.positions),
            'shield_states': self.shield_states,
            'start_pos': self.start_pos,
            'target_pos': self.target_pos,
            'snapshots': self.snapshots,
        }
        self._do_plot(data, save_dir, filename)

    @staticmethod
    def load_and_plot(filepath, save_dir='.', filename='shield_trajectory_stages'):
        data = ShieldTrajectoryRecorder.load(filepath)
        ShieldTrajectoryRecorder._do_plot(data, save_dir, filename)

    @staticmethod
    def _do_plot(data, save_dir, filename):
        _setup_matplotlib()

        positions = data['positions']
        shield_states = data['shield_states']
        start_pos = data['start_pos']
        target_pos = data['target_pos']
        snapshots = data['snapshots']

        if len(positions) < 2 or len(snapshots) == 0:
            print("[ShieldRecorder] 数据不足，跳过绘图。")
            return

        sorted_stages = sorted(snapshots.keys())
        stage_labels = {
            0.25: '任务进度 25%', 0.50: '任务进度 50%',
            0.75: '任务进度 75%', 1.00: '到达目标点',
        }

        n = len(sorted_stages)
        ncols = 2
        nrows = (n + 1) // 2
        fig = plt.figure(figsize=(16, 7 * nrows))

        for idx, stage in enumerate(sorted_stages):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            snap = snapshots[stage]
            traj_idx = snap['traj_idx']
            traj_seg = positions[:traj_idx]
            states_seg = shield_states[:traj_idx]
            coords = snap['coords'].copy()
            probs = snap['probs'].copy()

            if len(traj_seg) == 0:
                traj_seg = positions[:1]
                states_seg = shield_states[:1]

            uav_pos = traj_seg[-1]

            # ── 坐标范围（基于已飞轨迹，XY等比例尺）──
            traj_z_min = traj_seg[:, 2].min()
            traj_z_max = traj_seg[:, 2].max()
            traj_z_pad = max(8.0, (traj_z_max - traj_z_min) * 0.5)
            z_lim = [traj_z_min - traj_z_pad, traj_z_max + traj_z_pad]

            stage_pts_xy = [traj_seg[:, :2], start_pos[:2][np.newaxis],
                            target_pos[:2][np.newaxis]]
            stage_xy = np.vstack(stage_pts_xy)
            local_pad = max(12.0, 0.12 * np.ptp(stage_xy[:, 0]))
            x_lim = [stage_xy[:, 0].min() - local_pad, stage_xy[:, 0].max() + local_pad]
            y_lim = [stage_xy[:, 1].min() - local_pad, stage_xy[:, 1].max() + local_pad]
            x_lim[1] = max(x_lim[1], target_pos[0] + 10)
            y_lim[0] = min(y_lim[0], target_pos[1] - 10)
            y_lim[1] = max(y_lim[1], target_pos[1] + 10)

            x_span = x_lim[1] - x_lim[0]
            y_span = y_lim[1] - y_lim[0]
            max_span = max(x_span, y_span)
            x_center = (x_lim[0] + x_lim[1]) / 2
            y_center = (y_lim[0] + y_lim[1]) / 2
            x_lim = [x_center - max_span / 2, x_center + max_span / 2]
            y_lim = [y_center - max_span / 2, y_center + max_span / 2]

            # ── BSHM 体素（裁剪 + 两级分层）──
            if len(coords) > 0:
                clip = ((coords[:, 2] >= z_lim[0]) & (coords[:, 2] <= z_lim[1]) &
                        (coords[:, 0] >= x_lim[0]) & (coords[:, 0] <= x_lim[1]) &
                        (coords[:, 1] >= y_lim[0]) & (coords[:, 1] <= y_lim[1]))
                coords = coords[clip]
                probs = probs[clip]

            if len(coords) > 0:
                max_disp = 3000
                if len(coords) > max_disp:
                    high_idx = np.where(probs >= 0.75)[0]
                    other_idx = np.where(probs < 0.75)[0]
                    n_high = min(len(high_idx), max_disp // 2)
                    n_other = min(len(other_idx), max_disp - n_high)
                    parts = []
                    if n_high > 0:
                        parts.append(np.random.choice(high_idx, n_high, replace=False))
                    if n_other > 0:
                        parts.append(np.random.choice(other_idx, n_other, replace=False))
                    sel = np.concatenate(parts).astype(int)
                    coords = coords[sel]
                    probs = probs[sel]

                mid_mask = (probs >= 0.5) & (probs < 0.75)
                high_mask = probs >= 0.75

                if np.any(mid_mask):
                    ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                               coords[mid_mask, 2],
                               c='#E67E22', s=5, marker='o', alpha=0.35,
                               zorder=2, edgecolors='none', depthshade=False)
                if np.any(high_mask):
                    ax.scatter(coords[high_mask, 0], coords[high_mask, 1],
                               coords[high_mask, 2],
                               c='#C0392B', s=16, marker='o', alpha=0.85,
                               zorder=2, edgecolors='#8B0000',
                               linewidths=0.2, depthshade=False)

            # ── 按安全盾激活状态分段绘制轨迹 ──
            if len(states_seg) > 0:
                segments = []
                current_dom = get_dominant_layer(states_seg[0])
                seg_start = 0
                for i in range(1, len(states_seg)):
                    dom = get_dominant_layer(states_seg[i])
                    if dom != current_dom:
                        segments.append((seg_start, i, current_dom))
                        seg_start = i
                        current_dom = dom
                segments.append((seg_start, len(states_seg), current_dom))

                # 白色底线
                ax.plot(traj_seg[:, 0], traj_seg[:, 1], traj_seg[:, 2],
                        color='white', linewidth=4.5, alpha=0.35, zorder=6)

                drawn_labels = set()
                for s_idx, e_idx, dominant in segments:
                    s = max(s_idx - 1, 0)
                    e = min(e_idx + 1, len(traj_seg))
                    seg_pts = traj_seg[s:e]

                    style = LAYER_STYLE[dominant]
                    label = LAYER_NAMES[dominant] if dominant not in drawn_labels else None
                    drawn_labels.add(dominant)

                    ax.plot(seg_pts[:, 0], seg_pts[:, 1], seg_pts[:, 2],
                            color=style['color'], linewidth=style['lw'],
                            alpha=0.9, zorder=8, label=label)

            # ── 标记点 ──
            ax.scatter(*uav_pos, color='#2980B9', s=120, marker='D',
                       edgecolors='#1A5276', linewidths=1.5,
                       zorder=12, depthshade=False)
            ax.scatter(*start_pos, color='#27AE60', s=100, marker='o',
                       edgecolors='#1B7A3D', linewidths=1.2,
                       zorder=12, depthshade=False)
            ax.scatter(*target_pos, color='#8E44AD', s=130, marker='*',
                       edgecolors='#6C3483', linewidths=1.0,
                       zorder=12, depthshade=False)

            # ── 感知范围正方形 ──
            sr = 20.0
            ux, uy, uz = uav_pos
            sq_x = [ux-sr, ux+sr, ux+sr, ux-sr, ux-sr]
            sq_y = [uy-sr, uy-sr, uy+sr, uy+sr, uy-sr]
            sq_z = [uz] * 5
            ax.plot(sq_x, sq_y, sq_z,
                    color='#1A5276', linewidth=1.8, linestyle='--',
                    alpha=0.75, zorder=6)

            # ── 坐标轴 ──
            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)

            ax.plot(traj_seg[:, 0], traj_seg[:, 1],
                    np.full(len(traj_seg), z_lim[0]),
                    color='#999', linewidth=0.5, alpha=0.12)

            ax.set_xlabel('X - 北 (m)', fontsize=12, labelpad=6)
            ax.set_ylabel('Y - 东 (m)', fontsize=12, labelpad=6)
            ax.set_zlabel('Z - 地 (m)', fontsize=12, labelpad=6)
            ax.tick_params(axis='both', labelsize=7)
            ax.invert_zaxis()

            view_configs = {
                0.25: (20, -70), 0.50: (28, -55),
                0.75: (32, -42), 1.00: (38, -35),
            }
            elev, azim = view_configs.get(stage, (28, -52))
            ax.view_init(elev=elev, azim=azim)

            ax.text2D(1.00, 1.00, 'NED坐标系\nZ↓地面 -Z↑天空',
                      transform=ax.transAxes, fontsize=12, color='#888',
                      ha='right', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='#CCC', alpha=0.7))

            # ── 标题 ──
            label = stage_labels.get(stage, f'进度 {stage:.0%}')
            ax.set_title(label, fontsize=13, fontweight='bold', pad=8,
                         color='#2C3E50')

            # ── 信息标注 ──
            n_voxels = len(snap['coords'])
            flight_d = (np.sum(np.linalg.norm(np.diff(traj_seg, axis=0), axis=1))
                        if len(traj_seg) > 1 else 0)

            # 本阶段各层激活统计
            layer_counts = defaultdict(int)
            for st in states_seg:
                if not st:
                    layer_counts['drl_only'] += 1
                else:
                    dom = get_dominant_layer(st)
                    layer_counts[dom] += 1

            stat_lines = [f'BSHM体素: {n_voxels} | 飞行: {flight_d:.0f}m']
            for lname in ['drl_only', 'repulsion', 'speed_mod', 'vo', 'rrt']:
                if layer_counts[lname] > 0:
                    pct = layer_counts[lname] / max(len(states_seg), 1) * 100
                    stat_lines.append(f'{LAYER_NAMES[lname]}: {pct:.0f}%')

            ax.text2D(0.02, 0.98, '\n'.join(stat_lines),
                      transform=ax.transAxes, fontsize=12, color='#333',
                      va='top', ha='left',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#AAA', alpha=0.9))

        # ── 全局图例 ──
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=LAYER_STYLE['drl_only']['color'],
                   lw=2.5, label='纯DRL决策'),
            Line2D([0], [0], color=LAYER_STYLE['repulsion']['color'],
                   lw=3, label='近距斥力场'),
            Line2D([0], [0], color=LAYER_STYLE['speed_mod']['color'],
                   lw=2.8, label='速度调制'),
            Line2D([0], [0], color=LAYER_STYLE['vo']['color'],
                   lw=3, label='自适应VO'),
            Line2D([0], [0], color=LAYER_STYLE['rrt']['color'],
                   lw=3.5, label='RRT*融合'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#2980B9',
                   ms=8, markeredgecolor='#1A5276', label='无人机当前位置'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
                   ms=8, markeredgecolor='#1B7A3D', label='起飞点'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#8E44AD',
                   ms=10, markeredgecolor='#6C3483', label='目标点'),
            Line2D([0], [0], color='#1A5276', lw=1.8, ls='--', label='感知范围'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=12, frameon=True, edgecolor='#CCC',
                   framealpha=0.95, borderpad=0.6, columnspacing=1.2,
                   bbox_to_anchor=(0.5, -0.01))
        fig.subplots_adjust(hspace=0.22, wspace=0.15, bottom=0.08)

        os.makedirs(save_dir, exist_ok=True)
        for fmt in ['svg']:
            fig.savefig(os.path.join(save_dir, f'{filename}.{fmt}'), format=fmt)
        plt.close()
        print(f"[ShieldRecorder] 图表已保存: {save_dir}/{filename}.svg")