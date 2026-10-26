import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
class ClampedSpline3D:
    def __init__(self, points, start_velocity, end_velocity):
        """
        3D夹紧样条曲线类
        
        参数:
        points: numpy数组, 形状为(n, 3), 表示3D点坐标
        start_velocity: 起点速度向量 (vx, vy, vz)
        end_velocity: 终点速度向量 (vx, vy, vz)
        """
        self.points = np.array(points)
        self.start_velocity = np.array(start_velocity)
        self.end_velocity = np.array(end_velocity)
        self.splines = []  # 存储三个坐标轴的样条函数
        self._fit()
    
    def _calculate_chord_length_parameterization(self):
        """计算弦长参数化"""
        n = len(self.points)
        chords = np.zeros(n)
        for i in range(1, n):
            chord = np.linalg.norm(self.points[i] - self.points[i-1])
            chords[i] = chords[i-1] + chord
        return chords
    
    def _fit(self):
        """拟合夹紧样条曲线"""
        # 弦长参数化
        t = self._calculate_chord_length_parameterization()
        
        # 为每个坐标轴创建夹紧样条
        self.splines = []
        for dim in range(3):
            # 计算端点速度在该坐标轴的分量
            # 注意：需要将速度从物理意义转换为参数空间的速度
            chord_start = np.linalg.norm(self.points[1] - self.points[0])
            chord_end = np.linalg.norm(self.points[-1] - self.points[-2])
            
            # 参数空间的速度 = 物理速度 * (dt/ds) 的近似
            # 这里简化处理：使用第一个弦长和最后一个弦长进行缩放
            start_vel_param = self.start_velocity[dim] * chord_start / np.linalg.norm(self.points[1] - self.points[0])
            end_vel_param = self.end_velocity[dim] * chord_end / np.linalg.norm(self.points[-1] - self.points[-2])
            
            # 创建夹紧样条
            cs = CubicSpline(t, self.points[:, dim], 
                            bc_type=((1, start_vel_param), (1, end_vel_param)))
            self.splines.append(cs)
        
        self.t_range = (t[0], t[-1])
    
    def evaluate(self, t_values):
        """在参数t处计算样条曲线上的点"""
        if isinstance(t_values, (int, float)):
            t_values = np.array([t_values])
        
        t_values = np.clip(t_values, self.t_range[0], self.t_range[1])
        
        points = np.zeros((len(t_values), 3))
        for dim in range(3):
            points[:, dim] = self.splines[dim](t_values)
        
        return points[0] if len(points) == 1 else points
    
    def evaluate_derivative(self, t_values, order=1):
        """计算样条曲线的导数"""
        if isinstance(t_values, (int, float)):
            t_values = np.array([t_values])
        
        t_values = np.clip(t_values, self.t_range[0], self.t_range[1])
        
        derivatives = np.zeros((len(t_values), 3))
        for dim in range(3):
            derivatives[:, dim] = self.splines[dim](t_values, order)
        
        return derivatives[0] if len(derivatives) == 1 else derivatives
    
    def get_curvature(self, t_values):
        """计算曲率"""
        if isinstance(t_values, (int, float)):
            t_values = np.array([t_values])
        
        first_deriv = self.evaluate_derivative(t_values, 1)
        second_deriv = self.evaluate_derivative(t_values, 2)
        
        curvatures = []
        for i in range(len(t_values)):
            fd = first_deriv[i]
            sd = second_deriv[i]
            
            cross_product_norm = np.linalg.norm(np.cross(fd, sd))
            fd_norm_cubed = np.linalg.norm(fd) ** 3
            
            if fd_norm_cubed < 1e-10:  # 避免除零
                curvature = 0.0
            else:
                curvature = cross_product_norm / fd_norm_cubed
            
            curvatures.append(curvature)
        
        return np.array(curvatures)

def demo_clamped_spline_3d():
    """演示3D夹紧样条曲线的使用"""
    # 示例数据：5个3D点
    points_3d = np.array([
        [0, 0, 0],
        [4, 2, 1],
        [8, 1, 2],
        [10, 3, 6],
        [0, 0, 0]
    ])
    
    # 起点和终点速度
    start_vel = np.array([1.0, 0.5, 0.5])  # 起点速度向量
    end_vel = np.array([0.5, -0.5, 1.0])   # 终点速度向量
    
    # 创建夹紧样条
    spline_3d = ClampedSpline3D(points_3d, start_vel, end_vel)
    
    # 生成密集点用于绘图
    t_plot = np.linspace(0, spline_3d.t_range[1], 200)
    curve_points = spline_3d.evaluate(t_plot)
    
    # 计算端点处的导数（应该接近给定的速度）
    start_deriv = spline_3d.evaluate_derivative(0, 1)
    end_deriv = spline_3d.evaluate_derivative(spline_3d.t_range[1], 1)
    
    print("起点计算导数:", start_deriv)
    print("起点给定速度:", start_vel)
    print("终点计算导数:", end_deriv)
    print("终点给定速度:", end_vel)
    
    # 可视化
    fig = plt.figure(figsize=(12, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], 
             'b-', linewidth=2, label='夹紧样条曲线')
    ax1.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
             'ro-', markersize=8, label='控制点')
    
    # 绘制速度向量
    arrow_length = 0.5
    ax1.quiver(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2],
               start_vel[0]*arrow_length, start_vel[1]*arrow_length, start_vel[2]*arrow_length,
               color='g', linewidth=3, label='起点速度')
    ax1.quiver(points_3d[-1, 0], points_3d[-1, 1], points_3d[-1, 2],
               end_vel[0]*arrow_length, end_vel[1]*arrow_length, end_vel[2]*arrow_length,
               color='r', linewidth=3, label='终点速度')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D夹紧样条曲线')
    
    # 三个坐标轴的分量图
    ax2 = fig.add_subplot(222)
    ax2.plot(t_plot, curve_points[:, 0], 'r-', label='X(t)')
    ax2.plot(t_plot, curve_points[:, 1], 'g-', label='Y(t)')
    ax2.plot(t_plot, curve_points[:, 2], 'b-', label='Z(t)')
    ax2.set_xlabel('参数 t')
    ax2.set_ylabel('坐标值')
    ax2.legend()
    ax2.set_title('各坐标轴分量')
    ax2.grid(True)
    
    # 曲率图
    ax3 = fig.add_subplot(223)
    curvature = spline_3d.get_curvature(t_plot)
    ax3.plot(t_plot, curvature, 'm-', linewidth=2)
    ax3.set_xlabel('参数 t')
    ax3.set_ylabel('曲率')
    ax3.set_title('样条曲线曲率')
    ax3.grid(True)
    
    # 速度大小图
    ax4 = fig.add_subplot(224)
    velocity = np.linalg.norm(spline_3d.evaluate_derivative(t_plot, 1), axis=1)
    ax4.plot(t_plot, velocity, 'c-', linewidth=2)
    ax4.set_xlabel('参数 t')
    ax4.set_ylabel('速度大小')
    ax4.set_title('参数速度大小')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return spline_3d

# 运行演示
if __name__ == "__main__":
    spline = demo_clamped_spline_3d()
    
    # 测试插值点
    test_t = 2.5
    test_point = spline.evaluate(test_t)
    test_velocity = spline.evaluate_derivative(test_t, 1)
    
    print(f"\n在 t={test_t} 处的点: {test_point}")
    print(f"在 t={test_t} 处的速度: {test_velocity}")
    print(f"在 t={test_t} 处的速度大小: {np.linalg.norm(test_velocity):.3f}")