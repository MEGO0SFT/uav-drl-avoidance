import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import math

# 设置中文字体和矢量图输出
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # 确保字体为矢量

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

def generate_smooth_flight_path():
    """生成S型飞行路径数据 - 从[0,0,0]到[30,0,20]，y变化±15，运动时长30s"""
    # 生成参数化路径，使用弧长参数化
    # 参数t从0到1
    num_points = 300  # 30s / 0.1s = 300点
    t = np.linspace(0, 1, num_points)
    
    # S型路径: x从0到30，y正弦变化±15，z从0到20
    x = 30 * t
    y = 15 * np.sin(2 * np.pi * t)  # y变化±15
    z = 20 * t  # z从0到20
    
    # 计算速度向量
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    dz_dt = np.gradient(z, t)
    
    # 为了使飞行速度更加平滑，对速度进行限制
    # 设定最大速度为5m/s
    speed = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
    max_speed = 5.0
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    dx_dt = dx_dt * scale_factor
    dy_dt = dy_dt * scale_factor
    dz_dt = dz_dt * scale_factor
    
    # 重新计算位置，确保起点正确
    dt_param = 1.0 / (num_points - 1)
    x = np.cumsum(dx_dt) * dt_param * 30  # 调整x范围到[0,30]
    y = np.cumsum(dy_dt) * dt_param * 15  # 调整y范围到±15
    z = np.cumsum(dz_dt) * dt_param * 20  # 调整z范围到[0,20]
    
    # 调整起点和终点
    x += 0 - x[0]  # 调整x到0
    y -= y[0]      # 调整y到0
    z -= z[0]      # 调整z到0
    
    # 计算速度向量
    dt = 0.1  # 假设采样时间间隔
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    
    # 限制速度不超过10m/s
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    max_speed = 10.0
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    vx = vx * scale_factor
    vy = vy * scale_factor
    vz = vz * scale_factor
    
    # 限制加速度不超过5m/s²
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)
    
    accel = np.sqrt(ax**2 + ay**2 + az**2)
    max_accel = 5.0
    accel_scale = np.where(accel > max_accel, max_accel / accel, 1.0)
    ax = ax * accel_scale
    ay = ay * accel_scale
    az = az * accel_scale
    
    # 重新积分得到修正的速度
    vx = np.cumsum(ax) * dt
    vy = np.cumsum(ay) * dt
    vz = np.cumsum(az) * dt
    
    # 再次限制速度
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    vx = vx * scale_factor
    vy = vy * scale_factor
    vz = vz * scale_factor
    
    # 确保开始速度为0（平滑起飞）
    vx[0] = 0
    vy[0] = 0
    vz[0] = 0
    
    # 重新积分位置以确保路径正确
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt
    z = np.cumsum(vz) * dt
    
    # 调整起点到[0,0,0]
    x -= x[0]  # 调整x到0
    y -= y[0]  # 调整y到0
    z -= z[0]  # 调整z到0
    
    # 创建状态队列
    stat_deque = deque(maxlen=10)
    for i in range(len(vx)):
        stat_deque.append({
            "velocity": [vx[i], vy[i], vz[i]]
        })
    
    return x, y, z, vx, vy, vz, stat_deque

def generate_rough_flight_path():
    """生成粗糙S型飞行路径数据 - 相同路径但运动更粗糙"""
    # 生成参数化路径，使用弧长参数化
    # 参数t从0到1
    num_points = 300  # 30s / 0.1s = 300点
    t = np.linspace(0, 1, num_points)
    
    # S型路径: x从0到30，y正弦变化±15，z从0到20
    x = 30 * t
    y = 15 * np.sin(2 * np.pi * t)  # y变化±15
    z = 20 * t  # z从0到20
    
    # 添加随机扰动使路径更粗糙
    x += np.random.normal(0, 0.5, size=x.shape)
    y += np.random.normal(0, 0.8, size=y.shape)
    z += np.random.normal(0, 0.3, size=z.shape)
    
    # 计算速度向量
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    dz_dt = np.gradient(z, t)
    
    # 为了使飞行速度更加平滑，对速度进行限制
    # 设定最大速度为5m/s
    speed = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
    max_speed = 5.0
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    dx_dt = dx_dt * scale_factor
    dy_dt = dy_dt * scale_factor
    dz_dt = dz_dt * scale_factor
    
    # 重新计算位置，确保起点正确
    dt_param = 1.0 / (num_points - 1)
    x = np.cumsum(dx_dt) * dt_param * 30  # 调整x范围到[0,30]
    y = np.cumsum(dy_dt) * dt_param * 15  # 调整y范围到±15
    z = np.cumsum(dz_dt) * dt_param * 20  # 调整z范围到[0,20]
    
    # 调整起点和终点
    x += 0 - x[0]  # 调整x到0
    y -= y[0]      # 调整y到0
    z -= z[0]      # 调整z到0
    
    # 计算速度向量
    dt = 0.1  # 假设采样时间间隔
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    
    # 添加更多随机噪声使运动更粗糙
    vx += np.random.normal(0, 1.0, size=vx.shape)
    vy += np.random.normal(0, 1.0, size=vy.shape)
    vz += np.random.normal(0, 0.6, size=vz.shape)
    
    # 限制速度不超过10m/s
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    max_speed = 10.0
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    vx = vx * scale_factor
    vy = vy * scale_factor
    vz = vz * scale_factor
    
    # 限制加速度不超过5m/s²
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)
    
    # 添加更多加速度噪声
    ax += np.random.normal(0, 0.8, size=ax.shape)
    ay += np.random.normal(0, 0.8, size=ay.shape)
    az += np.random.normal(0, 0.5, size=az.shape)
    
    accel = np.sqrt(ax**2 + ay**2 + az**2)
    max_accel = 5.0
    accel_scale = np.where(accel > max_accel, max_accel / accel, 1.0)
    ax = ax * accel_scale
    ay = ay * accel_scale
    az = az * accel_scale
    
    # 重新积分得到修正的速度
    vx = np.cumsum(ax) * dt
    vy = np.cumsum(ay) * dt
    vz = np.cumsum(az) * dt
    
    # 再次限制速度
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    scale_factor = np.where(speed > max_speed, max_speed / speed, 1.0)
    vx = vx * scale_factor
    vy = vy * scale_factor
    vz = vz * scale_factor
    
    # 确保开始速度为0（平滑起飞）
    vx[0] = 0
    vy[0] = 0
    vz[0] = 0
    
    # 重新积分位置以确保路径正确
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt
    z = np.cumsum(vz) * dt
    
    # 调整起点到[0,0,0]
    x -= x[0]  # 调整x到0
    y -= y[0]  # 调整y到0
    z -= z[0]  # 调整z到0
    
    # 创建状态队列
    stat_deque = deque(maxlen=10)
    for i in range(len(vx)):
        stat_deque.append({
            "velocity": [vx[i], vy[i], vz[i]]
        })
    
    return x, y, z, vx, vy, vz, stat_deque

def plot_flight_analysis():
    """绘制[2,4]图分析飞行平滑性"""
    # 生成平滑飞行数据
    smooth_x, smooth_y, smooth_z, smooth_vx, smooth_vy, smooth_vz, smooth_deque = generate_smooth_flight_path()
    
    # 计算平滑飞行的奖励
    smooth_reward = v_clip_reward_calc(smooth_deque, None, None, dt=0.1, mode='hard')
    
    # 生成粗糙飞行数据
    rough_x, rough_y, rough_z, rough_vx, rough_vy, rough_vz, rough_deque = generate_rough_flight_path()
    
    # 计算粗糙飞行的奖励
    rough_reward = v_clip_reward_calc(rough_deque, None, None, dt=0.1, mode='hard')
    
    # 计算平滑飞行的各方向加速度和jerk
    smooth_ax = np.gradient(smooth_vx, 0.1)
    smooth_ay = np.gradient(smooth_vy, 0.1)
    smooth_az = np.gradient(smooth_vz, 0.1)
    
    smooth_jx = np.gradient(smooth_ax, 0.1) if len(smooth_ax) > 1 else np.zeros_like(smooth_ax)
    smooth_jy = np.gradient(smooth_ay, 0.1) if len(smooth_ay) > 1 else np.zeros_like(smooth_ay)
    smooth_jz = np.gradient(smooth_az, 0.1) if len(smooth_az) > 1 else np.zeros_like(smooth_az)
    
    # 计算粗糙飞行的各方向加速度和jerk
    rough_ax = np.gradient(rough_vx, 0.1)
    rough_ay = np.gradient(rough_vy, 0.1)
    rough_az = np.gradient(rough_vz, 0.1)
    
    rough_jx = np.gradient(rough_ax, 0.1) if len(rough_ax) > 1 else np.zeros_like(rough_ax)
    rough_jy = np.gradient(rough_ay, 0.1) if len(rough_ay) > 1 else np.zeros_like(rough_ay)
    rough_jz = np.gradient(rough_az, 0.1) if len(rough_az) > 1 else np.zeros_like(rough_az)
    
    # 创建图形 [2,4] 布局
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('S型运动无人机飞行平滑性分析 - 优化版 (30s运动时长)', fontsize=16)
    
    # 时间轴
    time_smooth = np.linspace(0, len(smooth_vx)*0.1, len(smooth_vx))
    time_rough = np.linspace(0, len(rough_vx)*0.1, len(rough_vx))
    
    # 第一行：平滑飞行
    # 子图1: X,Y,Z方向Jerk
    ax1 = axes[0, 0]
    ax1.plot(time_smooth, smooth_jx, label='X方向Jerk', color='red', linewidth=1)
    ax1.plot(time_smooth, smooth_jy, label='Y方向Jerk', color='green', linewidth=1)
    ax1.plot(time_smooth, smooth_jz, label='Z方向Jerk', color='blue', linewidth=1)
    ax1.set_title(f'高分飞行 - X,Y,Z方向Jerk (奖励: {smooth_reward:.3f})', fontsize=12)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('Jerk (m/s$^3$)')  # 使用Unicode字符表示立方
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: X,Y,Z方向加速度
    ax2 = axes[0, 1]
    ax2.plot(time_smooth, smooth_ax, label='X方向加速度', color='red', linewidth=1)
    ax2.plot(time_smooth, smooth_ay, label='Y方向加速度', color='green', linewidth=1)
    ax2.plot(time_smooth, smooth_az, label='Z方向加速度', color='blue', linewidth=1)
    ax2.set_title('高分飞行 - X,Y,Z方向加速度', fontsize=12)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('加速度 (m/s$^2$)')  # 使用Unicode字符表示平方
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3: X,Y,Z方向速度
    ax3 = axes[0, 2]
    ax3.plot(time_smooth, smooth_vx, label='X方向速度', color='red', linewidth=1)
    ax3.plot(time_smooth, smooth_vy, label='Y方向速度', color='green', linewidth=1)
    ax3.plot(time_smooth, smooth_vz, label='Z方向速度', color='blue', linewidth=1)
    ax3.set_title('高分飞行 - X,Y,Z方向速度', fontsize=12)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度 (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 子图4: 3D路径图
    ax4 = axes[0, 3]
    ax4_3d = fig.add_subplot(2, 4, 4, projection='3d')
    ax4_3d.plot(smooth_x, smooth_y, smooth_z, label='平滑路径', color='blue', linewidth=2)
    ax4_3d.scatter([0], [0], [0], color='red', s=100, label='起点[0,0,0]')
    ax4_3d.set_title(f'高分S型路径 (奖励: {smooth_reward:.3f})', fontsize=12)
    ax4_3d.set_xlabel('X (m)')
    ax4_3d.set_ylabel('Y (m)')
    ax4_3d.set_zlabel('Z (m)')
    
    # 第二行：粗糙飞行
    # 子图5: X,Y,Z方向Jerk
    ax5 = axes[1, 0]
    ax5.plot(time_rough, rough_jx, label='X方向Jerk', color='red', linewidth=1)
    ax5.plot(time_rough, rough_jy, label='Y方向Jerk', color='green', linewidth=1)
    ax5.plot(time_rough, rough_jz, label='Z方向Jerk', color='blue', linewidth=1)
    ax5.set_title(f'低分飞行 - X,Y,Z方向Jerk (奖励: {rough_reward:.3f})', fontsize=12)
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('Jerk (m/s$^3$)')  # 使用Unicode字符表示立方
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 子图6: X,Y,Z方向加速度
    ax6 = axes[1, 1]
    ax6.plot(time_rough, rough_ax, label='X方向加速度', color='red', linewidth=1)
    ax6.plot(time_rough, rough_ay, label='Y方向加速度', color='green', linewidth=1)
    ax6.plot(time_rough, rough_az, label='Z方向加速度', color='blue', linewidth=1)
    ax6.set_title('低分飞行 - X,Y,Z方向加速度', fontsize=12)
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('加速度 (m/s$^2$)')  # 使用Unicode字符表示平方
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 子图7: X,Y,Z方向速度
    ax7 = axes[1, 2]
    ax7.plot(time_rough, rough_vx, label='X方向速度', color='red', linewidth=1)
    ax7.plot(time_rough, rough_vy, label='Y方向速度', color='green', linewidth=1)
    ax7.plot(time_rough, rough_vz, label='Z方向速度', color='blue', linewidth=1)
    ax7.set_title('低分飞行 - X,Y,Z方向速度', fontsize=12)
    ax7.set_xlabel('时间 (s)')
    ax7.set_ylabel('速度 (m/s)')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 子图8: 3D路径图
    ax8 = axes[1, 3]
    ax8_3d = fig.add_subplot(2, 4, 8, projection='3d')
    ax8_3d.plot(rough_x, rough_y, rough_z, label='粗糙路径', color='red', linewidth=2)
    ax8_3d.scatter([0], [0], [0], color='red', s=100, label='起点[0,0,0]')
    ax8_3d.set_title(f'低分S型路径 (奖励: {rough_reward:.3f})', fontsize=12)
    ax8_3d.set_xlabel('X (m)')
    ax8_3d.set_ylabel('Y (m)')
    ax8_3d.set_zlabel('Z (m)')
    
    plt.tight_layout()
    plt.savefig('flight_smoothness_analysis.svg')
    plt.show()
    

# 运行绘图程序
plot_flight_analysis()