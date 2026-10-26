"""
trajectory_visualizer.py
========================
轨迹与 BSHM 数据记录器 + 四阶段 3D 可视化绘图。

关键设计：
- 快照时机基于「已飞行距离占起点到目标直线距离的比例」，
  而非 step/max_step，确保即使提前终止也能拍到所有中间阶段。
- force_final_snapshot() 会补拍所有遗漏的阶段。
- 每次快照都调用 BSHM 的 get_region_probabilities() 获取完整体素。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from typing import Optional, List


def _setup_matplotlib():
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
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })


class TrajectoryRecorder:
    """
    轨迹与 BSHM 数据记录器。

    Parameters
    ----------
    env : AirSimDroneEnv
        单个环境实例（非 VecEnv）。
    stat_range : float
        BSHM 查询半径，与 VehicleThread.stat_range 一致。
    prob_threshold : float
        体素占据概率阈值，>= 此值绘制为障碍物。
    snapshot_stages : list of float
        在飞行进度达到这些比例时保存 BSHM 快照。
    """

    def __init__(self, env, stat_range=20.0, prob_threshold=0.5,
                 snapshot_stages=None):
        self.env = env
        self.vehicle = env.vehicle
        self.stat_range = stat_range
        self.prob_threshold = prob_threshold
        self.snapshot_stages = snapshot_stages or [0.25, 0.50, 0.75, 1.0]

        self.positions: List[np.ndarray] = []
        self.start_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.total_distance_to_target: float = 1.0

        # {stage_float: {'coords': ndarray, 'probs': ndarray, 'traj_idx': int}}
        self.snapshots: dict = {}
        self._snapshot_taken: set = set()

    # ──────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────

    def reset(self):
        """env.reset() 之后调用。"""
        self.positions = []
        self.snapshots = {}
        self._snapshot_taken = set()

        self.start_pos = self.vehicle.pos.copy()
        if len(self.vehicle.stat_deque) > 0:
            self.start_pos = (self.vehicle.pos +
                              self.vehicle.stat_deque[-1]['position']).copy()
        self.target_pos = self.vehicle.target.copy()
        self.total_distance_to_target = max(
            np.linalg.norm(self.target_pos - self.start_pos), 1.0
        )

    def record(self):
        """env.step() 之后调用。"""
        # 读取全局位置
        if len(self.vehicle.stat_deque) > 0:
            pos = (self.vehicle.pos +
                   self.vehicle.stat_deque[-1]['position']).copy()
        else:
            pos = self.vehicle.pos.copy()
        self.positions.append(pos)

        # 计算飞行进度：已飞行距离 / 起点到目标直线距离
        flown = np.sum(np.linalg.norm(
            np.diff(self.positions, axis=0), axis=1
        )) if len(self.positions) > 1 else 0.0
        progress = min(flown / self.total_distance_to_target, 1.5)

        # 也计算距目标的接近比例作为辅助判断
        dist_remaining = np.linalg.norm(self.target_pos - pos)
        approach_progress = 1.0 - dist_remaining / self.total_distance_to_target

        effective_progress = max(progress, approach_progress)

        # 检查所有未拍摄的阶段
        for stage in self.snapshot_stages:
            if stage not in self._snapshot_taken and effective_progress >= stage:
                self._take_snapshot(stage)

    def force_final_snapshot(self):
        """
        episode 结束时调用。补拍所有遗漏的阶段快照。
        """
        for stage in sorted(self.snapshot_stages):
            if stage not in self._snapshot_taken:
                self._take_snapshot(stage)

    # ──────────────────────────────────────────────────────────────
    # BSHM 快照
    # ──────────────────────────────────────────────────────────────

    def _take_snapshot(self, stage: float):
        """拍摄当前时刻的 BSHM 体素快照。"""
        self._snapshot_taken.add(stage)
        traj_idx = len(self.positions)  # 轨迹截断到这里

        coords = np.zeros((0, 3))
        probs = np.array([])

        try:
            octo = self.vehicle.OctoMaps

            # 优先用全局边界
            g_min = octo.global_min_bound
            g_max = octo.global_max_bound

            if g_min is not None and g_max is not None:
                margin = np.array([5.0, 5.0, 5.0])
                _, c, p = octo.get_region_probabilities(
                    g_min - margin, g_max + margin, return_coords=True
                )
            else:
                # 回退：轨迹包围盒
                traj_arr = np.array(self.positions)
                bbox_min = traj_arr.min(axis=0) - self.stat_range * 2
                bbox_max = traj_arr.max(axis=0) + self.stat_range * 2
                _, c, p = octo.get_region_probabilities(
                    bbox_min, bbox_max, return_coords=True
                )

            if len(c) > 0:
                mask = p >= self.prob_threshold
                coords = c[mask].copy()
                probs = p[mask].copy()

        except Exception as e:
            print(f"[Recorder] BSHM快照失败 (stage={stage:.0%}): {e}")

        self.snapshots[stage] = {
            'coords': coords,
            'probs': probs,
            'traj_idx': traj_idx,
        }
        print(f"[Recorder] 阶段 {stage:.0%} 快照完成: "
              f"轨迹点={traj_idx}, BSHM障碍体素={len(coords)}")

    # ──────────────────────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────────────────────

    def save(self, filepath='trajectory_data.npz'):
        save_dict = {
            'positions': np.array(self.positions),
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
        print(f"[Recorder] 数据已保存: {filepath}")

    @staticmethod
    def load(filepath):
        d = np.load(filepath, allow_pickle=True)
        result = {
            'positions': d['positions'],
            'start_pos': d['start_pos'],
            'target_pos': d['target_pos'],
            'snapshots': {},
        }
        for stage in d['stages']:
            tag = f'{stage:.2f}'
            result['snapshots'][float(stage)] = {
                'coords': d[f'coords_{tag}'],
                'probs': d[f'probs_{tag}'],
                'traj_idx': int(d[f'traj_idx_{tag}'][0]),
            }
        return result

    # ──────────────────────────────────────────────────────────────
    # 绘图
    # ──────────────────────────────────────────────────────────────

    def plot(self, save_dir='.', filename='trajectory_stages'):
        data = {
            'positions': np.array(self.positions),
            'start_pos': self.start_pos,
            'target_pos': self.target_pos,
            'snapshots': self.snapshots,
        }
        self._do_plot(data, save_dir, filename)

    @staticmethod
    def load_and_plot(filepath, save_dir='.', filename='trajectory_stages'):
        data = TrajectoryRecorder.load(filepath)
        TrajectoryRecorder._do_plot(data, save_dir, filename)

    @staticmethod
    def _do_plot(data, save_dir, filename):
        _setup_matplotlib()

        positions = data['positions']
        start_pos = data['start_pos']
        target_pos = data['target_pos']
        snapshots = data['snapshots']

        if len(positions) == 0 or len(snapshots) == 0:
            print("[Recorder] 无数据，跳过绘图。")
            return

        sorted_stages = sorted(snapshots.keys())

        stage_labels = {
            0.25: '任务进度 25%', 0.50: '任务进度 50%',
            0.75: '任务进度 75%', 1.00: '到达目标点',
        }

        # 计算全局坐标范围（用于最终阶段的全景视图）
        all_pts = [positions, start_pos[np.newaxis], target_pos[np.newaxis]]
        for snap in snapshots.values():
            if len(snap['coords']) > 0:
                all_pts.append(snap['coords'])
        all_arr = np.vstack(all_pts)
        global_pad = 10.0
        x_lim_global = [all_arr[:, 0].min() - global_pad, all_arr[:, 0].max() + global_pad]
        y_lim_global = [all_arr[:, 1].min() - global_pad, all_arr[:, 1].max() + global_pad]
        z_lim_global = [all_arr[:, 2].min() - global_pad, all_arr[:, 2].max() + global_pad]

        n = len(sorted_stages)
        ncols = 2
        nrows = (n + 1) // 2
        fig = plt.figure(figsize=(16, 7 * nrows))

        for idx, stage in enumerate(sorted_stages):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            snap = snapshots[stage]
            traj_idx = snap['traj_idx']
            traj_seg = positions[:traj_idx]
            coords = snap['coords']
            probs = snap['probs']

            if len(traj_seg) == 0:
                traj_seg = positions[:1]

            uav_pos = traj_seg[-1]

            # ── 坐标范围先算（后续体素裁剪和地面投影都依赖它）──
            # Z轴紧贴飞行高度走廊
            traj_z_min = traj_seg[:, 2].min()
            traj_z_max = traj_seg[:, 2].max()
            traj_z_pad = max(8.0, (traj_z_max - traj_z_min) * 0.5)
            z_lim = [traj_z_min - traj_z_pad, traj_z_max + traj_z_pad]

            # XY范围：先算自然边界，再统一为正方形（等比例尺）
            stage_pts_xy = [traj_seg[:, :2], start_pos[:2][np.newaxis],
                            target_pos[:2][np.newaxis]]
            stage_xy = np.vstack(stage_pts_xy)
            local_pad = max(12.0, 0.12 * np.ptp(stage_xy[:, 0]))
            x_lim = [stage_xy[:, 0].min() - local_pad, stage_xy[:, 0].max() + local_pad]
            y_lim = [stage_xy[:, 1].min() - local_pad, stage_xy[:, 1].max() + local_pad]

            # 确保目标点可见
            x_lim[1] = max(x_lim[1], target_pos[0] + 10)
            y_lim[0] = min(y_lim[0], target_pos[1] - 10)
            y_lim[1] = max(y_lim[1], target_pos[1] + 10)

            # ★ XY等比例尺：取XY跨度的最大值，双轴统一
            x_span = x_lim[1] - x_lim[0]
            y_span = y_lim[1] - y_lim[0]
            max_span = max(x_span, y_span)
            x_center = (x_lim[0] + x_lim[1]) / 2
            y_center = (y_lim[0] + y_lim[1]) / 2
            x_lim = [x_center - max_span / 2, x_center + max_span / 2]
            y_lim = [y_center - max_span / 2, y_center + max_span / 2]

            # ── BSHM 体素（裁剪到飞行走廊）──
            if len(coords) > 0:
                z_clip_mask = ((coords[:, 2] >= z_lim[0]) &
                               (coords[:, 2] <= z_lim[1]))
                coords = coords[z_clip_mask]
                probs = probs[z_clip_mask]

            if len(coords) > 0:
                xy_clip = ((coords[:, 0] >= x_lim[0]) & (coords[:, 0] <= x_lim[1]) &
                           (coords[:, 1] >= y_lim[0]) & (coords[:, 1] <= y_lim[1]))
                coords = coords[xy_clip]
                probs = probs[xy_clip]

            if len(coords) > 0:
                # 降采样：优先保留高概率体素
                max_disp = 3000
                if len(coords) > max_disp:
                    high_idx = np.where(probs >= 0.75)[0]
                    other_idx = np.where(probs < 0.75)[0]
                    n_high = min(len(high_idx), max_disp // 2)
                    n_other = min(len(other_idx), max_disp - n_high)
                    sel_parts = []
                    if n_high > 0:
                        sel_parts.append(np.random.choice(high_idx, n_high, replace=False))
                    if n_other > 0:
                        sel_parts.append(np.random.choice(other_idx, n_other, replace=False))
                    sel = np.concatenate(sel_parts).astype(int)
                    coords = coords[sel]
                    probs = probs[sel]

                # ★ 两级分层：中概率(0.5-0.75) / 高概率(>=0.75)
                mid_mask  = (probs >= 0.5) & (probs < 0.75)
                high_mask = probs >= 0.75

                # 中概率：橙色中等点
                if np.any(mid_mask):
                    ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                               coords[mid_mask, 2],
                               c='#E67E22', s=5, marker='o', alpha=0.35,
                               zorder=2, edgecolors='none', depthshade=False)

                # 高概率：深红大点（主体障碍物）
                if np.any(high_mask):
                    ax.scatter(coords[high_mask, 0], coords[high_mask, 1],
                               coords[high_mask, 2],
                               c='#C0392B', s=16, marker='o', alpha=0.85,
                               zorder=2, edgecolors='#8B0000',
                               linewidths=0.2, depthshade=False)

            # ── 轨迹（加粗、提高zorder确保在体素上方）──
            ax.plot(traj_seg[:, 0], traj_seg[:, 1], traj_seg[:, 2],
                    color='#1A5276', linewidth=2.8, zorder=8, alpha=1.0)
            # 轨迹高亮描边（白色底线增强可见度）
            ax.plot(traj_seg[:, 0], traj_seg[:, 1], traj_seg[:, 2],
                    color='white', linewidth=4.0, zorder=7, alpha=0.5)

            # 剩余路径虚线
            if traj_idx < len(positions):
                rem = positions[traj_idx:]
                ax.plot(rem[:, 0], rem[:, 1], rem[:, 2],
                        color='#566573', linewidth=1.8, linestyle='--',
                        alpha=0.65, zorder=6)

            # ── 标记（加大尺寸）──
            ax.scatter(*uav_pos, color='#2980B9', s=120, marker='D',
                       edgecolors='#1A5276', linewidths=1.5,
                       zorder=12, depthshade=False)
            ax.scatter(*start_pos, color='#27AE60', s=100, marker='o',
                       edgecolors='#1B7A3D', linewidths=1.2,
                       zorder=12, depthshade=False)
            ax.scatter(*target_pos, color='#8E44AD', s=130, marker='*',
                       edgecolors='#6C3483', linewidths=1.0,
                       zorder=12, depthshade=False)

            # ── 感知范围（正方形）──
            sr = 20.0
            ux, uy, uz = uav_pos[0], uav_pos[1], uav_pos[2]
            sq_x = [ux-sr, ux+sr, ux+sr, ux-sr, ux-sr]
            sq_y = [uy-sr, uy-sr, uy+sr, uy+sr, uy-sr]
            sq_z = [uz] * 5
            ax.plot(sq_x, sq_y, sq_z,
                    color='#1A5276', linewidth=1.8, linestyle='--',
                    alpha=0.75, zorder=6)

            # ── 应用坐标范围 ──
            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)

            # 地面投影
            ax.plot(traj_seg[:, 0], traj_seg[:, 1],
                    np.full(len(traj_seg), z_lim[0]),
                    color='#2980B9', linewidth=0.6, alpha=0.15)

            ax.set_xlabel('X - 北 (m)', fontsize=9, labelpad=6)
            ax.set_ylabel('Y - 东 (m)', fontsize=9, labelpad=6)
            ax.set_zlabel('Z - 地 (m)', fontsize=9, labelpad=6)
            ax.tick_params(axis='both', labelsize=7)
            ax.invert_zaxis()

            # ── 视角：四个阶段明显不同 ──
            view_configs = {
                0.25: (20, -70),   # 低仰角、侧前方 → 聚焦起飞段
                0.50: (28, -55),   # 中仰角、偏正面 → 展示中段
                0.75: (32, -42),   # 较高仰角 → 俯瞰大部分路径
                1.00: (38, -35),   # 最高仰角、正面 → 全景鸟瞰
            }
            elev, azim = view_configs.get(stage, (28, -52))
            ax.view_init(elev=elev, azim=azim)

            # NED 坐标系说明
            ax.text2D(0.98, 0.02, 'NED坐标系\nZ↓地面  -Z↑天空',
                      transform=ax.transAxes, fontsize=7, color='#888',
                      ha='right', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='#CCC', alpha=0.7))

            # 标题
            label = stage_labels.get(stage, f'进度 {stage:.0%}')
            ax.set_title(label, fontsize=13, fontweight='bold', pad=8,
                         color='#2C3E50')

            # 信息标注
            n_voxels = len(snap['coords'])
            flight_d = (np.sum(np.linalg.norm(np.diff(traj_seg, axis=0), axis=1))
                        if len(traj_seg) > 1 else 0)
            ax.text2D(0.02, 0.95, f'BSHM障碍体素: {n_voxels}',
                      transform=ax.transAxes, fontsize=9, color='#E74C3C',
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#E74C3C', alpha=0.85))
            ax.text2D(0.02, 0.88, f'飞行距离: {flight_d:.0f} m',
                      transform=ax.transAxes, fontsize=8, color='#555',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#CCC', alpha=0.85))

        # ── 全局图例 ──
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Line2D([0], [0], color='#2980B9', lw=2, label='已飞行轨迹'),
            Line2D([0], [0], color='#7F8C8D', lw=1.6, ls='--', label='剩余路径'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#2980B9',
                   ms=8, markeredgecolor='#1A5276', label='无人机当前位置'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
                   ms=8, markeredgecolor='#1B7A3D', label='起飞点'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#8E44AD',
                   ms=10, markeredgecolor='#6C3483', label='目标点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
                   ms=8, markeredgecolor='#8B0000', label='高概率体素 (≥0.75)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22',
                   ms=6, markeredgecolor='none', label='中概率体素 (0.5-0.75)'),
            Line2D([0], [0], color='#1A5276', lw=1.8, ls='--', label='感知范围'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   fontsize=9, frameon=True, edgecolor='#CCC',
                   framealpha=0.95, borderpad=0.6, columnspacing=1.2,
                   bbox_to_anchor=(0.5, -0.01))
        fig.subplots_adjust(hspace=0.22, wspace=0.15, bottom=0.08)

        os.makedirs(save_dir, exist_ok=True)
        for fmt in ['png', 'pdf', 'svg']:
            fig.savefig(os.path.join(save_dir, f'{filename}.{fmt}'), format=fmt)
        plt.close()
        print(f"[Recorder] 图表已保存: {save_dir}/{filename}.*")