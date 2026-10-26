"""
消融实验结果绘图 — 分组柱状图（优化版）
- 组内柱紧密贴合无间隙
- 学术论文配色：低饱和度冷暖对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                        'Microsoft YaHei', 'Arial', 'DejaVu Sans',
                        'Noto Sans CJK SC'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# ═══════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════

data = [
    {"tag": "A", "success": 90.94, "collision": 9.06, "n": 309,
     "avg_time": 22.31, "std_time": 4.26,
     "mean_reward": 245.80, "std_reward": 78.36},
    {"tag": "B", "success": 94.46, "collision": 5.54, "n": 307,
     "avg_time": 24.38, "std_time": 4.62,
     "mean_reward": 261.44, "std_reward": 81.76},
    {"tag": "C", "success": 93.11, "collision": 6.89, "n": 305,
     "avg_time": 24.36, "std_time": 4.47,
     "mean_reward": 258.29, "std_reward": 85.50},
    {"tag": "D", "success": 95.82, "collision": 4.18, "n": 311,
     "avg_time": 24.65, "std_time": 5.27,
     "mean_reward": 263.84, "std_reward": 71.64},
    {"tag": "E", "success": 96.95, "collision": 3.05, "n": 525,
     "avg_time": 24.70, "std_time": 5.28,
     "mean_reward": 288.96, "std_reward": 68.27},
]

tags = [d["tag"] for d in data]
success = np.array([d["success"] for d in data])
collision = np.array([d["collision"] for d in data])
avg_time = np.array([d["avg_time"] for d in data])
std_time = np.array([d["std_time"] for d in data])
mean_reward = np.array([d["mean_reward"] for d in data])
std_reward = np.array([d["std_reward"] for d in data])

err_reward = std_reward / 3
err_time = std_time / 3

n_groups = len(tags)

# ── 布局参数：组内零间隙 ──
bar_w = 0.24                  # 单柱宽度
group_w = bar_w * 3           # 三柱总宽 = 0.72
x = np.arange(n_groups)       # 组中心

# 三柱偏移：紧贴排列
off1 = -bar_w                 # 左柱中心
off2 = 0                      # 中柱中心
off3 = bar_w                  # 右柱中心

# ── 学术配色（低饱和、高辨识）──
# 成功率/碰撞率：暖红 + 冷蓝
c_success   = '#E8927C'       # 柔和赤陶色
c_success_e = '#C96B55'
c_fail      = '#7AADE2'       # 柔和钢蓝
c_fail_e    = '#5B8FBF'

# 奖励：沉稳青绿
c_reward    = '#7BC8A4'
c_reward_e  = '#4FA882'

# 耗时：温暖琥珀
c_time      = '#E8C170'
c_time_e    = '#C9A04E'

# 归一化参数
reward_max = 350.0
time_max   = 35.0
reward_pct = mean_reward / reward_max * 100
err_reward_pct = err_reward / reward_max * 100
time_pct = avg_time / time_max * 100
err_time_pct = err_time / time_max * 100

config_labels = ('A: 纯DRL（基线）    B: DRL + 近距斥力场    '
                 'C: B + 自适应VO    D: C + RRT*融合    '
                 'E: 完整系统（D + 安全防护策略）')

save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)

# ═══════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

# ── 柱1：成功率 + 碰撞率堆叠 ──
pos1 = x + off1
bars_s = ax.bar(pos1, success, bar_w,
                color=c_success, edgecolor=c_success_e,
                linewidth=0.5, zorder=3, label='成功率')
bars_c = ax.bar(pos1, collision, bar_w, bottom=success,
                color=c_fail, edgecolor=c_fail_e,
                linewidth=0.5, alpha=0.75, zorder=3, label='碰撞率')

for i in range(n_groups):
    ax.text(pos1[i], success[i] / 2,
            f'{success[i]:.1f}%', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#333')
    y_mid = success[i] + collision[i] / 2
    ax.text(pos1[i], y_mid,
            f'{collision[i]:.1f}%', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#222')

# ── 柱2：平均奖励 ──
pos2 = x + off2
bars_r = ax.bar(pos2, reward_pct, bar_w,
                color=c_reward, edgecolor=c_reward_e,
                linewidth=0.5, zorder=3, label='平均累积奖励',
                yerr=err_reward_pct, capsize=3,
                error_kw={'lw': 0.8, 'color': '#555', 'capthick': 0.8})

for i in range(n_groups):
    h = reward_pct[i]
    ax.text(pos2[i], h + err_reward_pct[i] + 1.0,
            f'{mean_reward[i]:.0f}±{std_reward[i]:.0f}',
            ha='center', va='bottom',
            fontsize=6.5, fontweight='bold', color='#444')

# ── 柱3：平均耗时 ──
pos3 = x + off3
bars_t = ax.bar(pos3, time_pct, bar_w,
                color=c_time, edgecolor=c_time_e,
                linewidth=0.5, zorder=3, label='成功任务平均耗时',
                yerr=err_time_pct, capsize=3,
                error_kw={'lw': 0.8, 'color': '#555', 'capthick': 0.8})

for i in range(n_groups):
    h = time_pct[i]
    ax.text(pos3[i], h + err_time_pct[i] + 1.0,
            f'{avg_time[i]:.1f}±{std_time[i]:.1f}s',
            ha='center', va='bottom',
            fontsize=6.5, fontweight='bold', color='#444')

# ── 坐标轴 ──
ax.set_xticks(x)
ax.set_xticklabels(tags, fontsize=12, fontweight='bold')
ax.set_ylabel('成功-失败率', fontsize=11, fontweight='bold', color='#444')
ax.tick_params(axis='y', labelsize=9, colors='#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#999')
ax.spines['bottom'].set_color('#999')
ax.grid(axis='y', alpha=0.15, linestyle='--', zorder=0)
ax.set_ylim([0, 115])
ax.axhline(y=100, color='#CCC', linewidth=0.5, linestyle='-', zorder=1)

# 右侧副Y轴
ax_r = ax.twinx()
ax_r.set_ylim([0, reward_max * 115 / 100])
ax_r.set_ylabel('平均奖励', fontsize=10, color=c_reward_e, labelpad=8)
ax_r.tick_params(axis='y', labelsize=9, colors=c_reward_e)
ax_r.spines['top'].set_visible(False)
ax_r.spines['left'].set_visible(False)
ax_r.spines['right'].set_color(c_reward_e)
ax_r.spines['right'].set_linewidth(0.6)

# ── 图例 ──
legend = ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92,
                   edgecolor='#CCC', borderpad=0.5, ncol=2,
                   handlelength=1.5, columnspacing=1.0)
legend.get_frame().set_linewidth(0.5)

# ── 底部配置说明 ──
fig.text(0.5, 0, config_labels, ha='center', va='top', fontsize=9.5,
         color='#555',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA',
                   edgecolor='#DDD', alpha=0.95))

# ── 保存 ──
for fmt in ['svg']:
    fig.savefig(os.path.join(save_dir, f'ablation_combined.{fmt}'), format=fmt)
plt.close()

print("Done. 分组柱状图已保存至 ./figures/ablation_combined.svg")