# ARCHIVAL COPY of the script the author used to produce `ppo_training_convergence.png` and
# `ppo_metric_0*.svg`.
#
# It expects five single-tag CSV exports from the TensorBoard UI:
#     PPO_98.csv      -> reward
#     PPO_98 (1).csv  -> success rate
#     PPO_98 (2).csv  -> approx KL
#     PPO_98 (3).csv  -> action std
#     PPO_98 (4).csv  -> avg episode steps
#
# Those five files are the SAME run, five different scalar tags -- the "(n)" suffixes are browser
# download duplicates, not separate experiments. They are not shipped; the equivalent data lives in
# benchmarks/tb_curves/, one row per step with every tag as a column. Adapt `file_mapping` below if
# you want to regenerate the panels from there.
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 设置工作目录为当前脚本所在目录，确保能找到CSV文件
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- Configuration ---
file_mapping = [
    {
        "file": "PPO_98.csv", 
        "label": "每幕平均奖励", 
        "ylabel": "平均奖励 (Average Reward)", 
        "filename_suffix": "01_reward"
    },
    {
        "file": "PPO_98 (1).csv", 
        "label": "最近100幕任务成功率", 
        "ylabel": "成功率 (Success Rate)", 
        "filename_suffix": "02_success_rate"
    },
    {
        "file": "PPO_98 (2).csv", 
        "label": "近似KL散度", 
        "ylabel": "近似 KL 散度 (Approx. KL Div)", 
        "filename_suffix": "03_kl_div"
    },
    {
        "file": "PPO_98 (3).csv", 
        "label": "动作采样标准差", 
        "ylabel": "标准差 (Std Dev)", 
        "filename_suffix": "04_std_dev"
    },
    {
        "file": "PPO_98 (4).csv", 
        "label": "每幕平均步长", 
        "ylabel": "平均步数 (Avg. Steps)", 
        "filename_suffix": "05_avg_steps"
    }
]

smoothing_factor = 0.9
HOUR_INTERVAL = 6  # X-axis tick interval: 6 hours

# --- Academic Plotting Style Settings ---
# Font settings for Chinese support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'STHeiti'] 
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

def smooth_data(data, weight):
    """TensorBoard style smoothing."""
    if len(data) == 0:
        return data
    alpha = 1 - weight
    return data.ewm(alpha=alpha, adjust=False).mean()

def main():
    # Verify file existence
    missing_files = [item['file'] for item in file_mapping if not os.path.exists(item['file'])]
    if missing_files:
        print(f"Error: The following files were not found: {missing_files}")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # --- 首先找到所有文件中最早的 Wall time 作为全局起始时间 ---
    global_start_time = float('inf')
    for info in file_mapping:
        try:
            df_temp = pd.read_csv(info['file'])
            df_temp.columns = df_temp.columns.str.strip()
            if 'Wall time' in df_temp.columns:
                global_start_time = min(global_start_time, df_temp['Wall time'].min())
        except Exception:
            pass
    
    if global_start_time == float('inf'):
        print("Error: Could not determine global start time from 'Wall time' column.")
        return
    
    print(f"Global start time (Unix): {global_start_time}")

    for i, info in enumerate(file_mapping):
        filename = info['file']
        
        try:
            # Read and process data
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()
            
            if 'Wall time' not in df.columns or 'Value' not in df.columns:
                print(f"Warning: {filename} missing 'Wall time' or 'Value' columns. Skipping.")
                continue

            df = df.sort_values('Wall time')
            
            # 将 Wall time 转换为相对于全局起始时间的小时数
            x_hours = (df['Wall time'] - global_start_time) / 3600.0
            
            y_raw = df['Value']
            y_smooth = smooth_data(y_raw, smoothing_factor)
            
            # Create a NEW square figure for EACH metric
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)  # Square aspect ratio
            
            # Plot
            ax.plot(x_hours, y_smooth, color=colors[i % len(colors)], linewidth=2.2, label=info['label'])
            
            # Labels
            ax.set_ylabel(info['ylabel'], fontsize=20, fontweight='bold')
            ax.set_xlabel('训练时间 (Training Time / hours)', fontsize=20, fontweight='bold')
            
            # Ticks Configuration
            ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
            
            # Custom X-axis ticks with 6-hour interval
            min_hour = x_hours.min()
            max_hour = x_hours.max()
            
            start_tick = int(np.floor(min_hour / HOUR_INTERVAL) * HOUR_INTERVAL)
            end_tick = int(np.ceil(max_hour / HOUR_INTERVAL) * HOUR_INTERVAL)
            
            ticks = np.arange(start_tick, end_tick + HOUR_INTERVAL, HOUR_INTERVAL)
            ax.set_xticks(ticks)
            
            # X轴标签格式：显示整数小时
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
            
            # Grid
            ax.grid(True)
            
            # Legend (Avoid occlusion)
            legend = ax.legend(loc='best', framealpha=0.95, fontsize=18)
            legend.get_frame().set_linewidth(0.8)

            # Save as SVG (Vector format for papers)
            output_svg = f"ppo_metric_{info['filename_suffix']}.svg"
            plt.savefig(output_svg, format='svg', bbox_inches='tight')
            print(f"Saved SVG: {output_svg}")

            plt.close(fig)  # Close figure to free memory

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()