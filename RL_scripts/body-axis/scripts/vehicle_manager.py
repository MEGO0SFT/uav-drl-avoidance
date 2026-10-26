import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from socket import *
import os,sys
import numpy as np
import time
import threading
import json
from collections import deque
import torch
import torch.nn.functional as F
from numba import jit, njit
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录的路径（根据你的目录结构向上跳转3级）
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# 将项目根目录添加到Python路径
if project_root not in sys.path:
    sys.path.append(project_root)
from methods.AirSim_ReinforceLearning import OctoMap_o3d as om_o3d
from methods.AirSim_ReinforceLearning import LidarProcess as lidar

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 基于激光雷达的无人机自主建图与避障

global OctoMap, device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获取本机ip地址
def get_real_ip():
    try:
        with socket(AF_INET, SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'

def load_json_to_nx3_matrix(json_file_path):
    """
    快速读取JSON文件并转换为Nx3的numpy矩阵
    
    参数:
    json_file_path: JSON文件路径
    
    返回:
    numpy矩阵 (Nx3) 或 None（如果出错）
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 直接转换为numpy数组
        matrix = np.array(data, dtype=np.float64)
        
        # 验证形状
        if len(matrix.shape) != 2 or matrix.shape[1] != 3:
            raise ValueError(f"数据格式不正确，期望Nx3格式，实际得到{matrix.shape}")
        
        return matrix
    
    except FileNotFoundError:
        print(f"错误: 文件 '{json_file_path}' 不存在")
        return None
    except json.JSONDecodeError:
        print(f"错误: '{json_file_path}' 不是有效的JSON文件")
        return None
    except ValueError as ve:
        print(f"数据格式错误: {ve}")
        return None
    except Exception as e:
        print(f"处理JSON文件时发生错误: {e}")
        return None


def matrix_global_to_target(vector_target, vector_global, epsilon=1e-10):
    """
    将全局坐标系下的向量转换到目标坐标系下
    
    Args:
        vector_target: 目标方向向量 (无需单位化)
        vector_global: 全局坐标系下的向量
        epsilon: 数值稳定系数
    
    Returns:
        目标坐标系下的向量
    
    数学原理：
        构造从全局到目标坐标系的旋转矩阵R，满足：v_target = R @ v_global
        R的列向量是全局坐标系基向量在目标坐标系下的表示
    """
    # 输入校验
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_global = np.asarray(vector_global, dtype=np.float64)
    
    # 归一化目标方向向量（x轴）
    target_norm = np.linalg.norm(vector_target)
    if target_norm < epsilon:
        raise ValueError("目标方向向量模长过小，无法归一化")
    x_axis = vector_target / target_norm
    
    # 计算y轴（与全局z轴和x轴正交）
    z_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis = np.cross(z_global, x_axis)
    y_norm = np.linalg.norm(y_axis)
    
    # 处理x轴与全局z轴平行的情况
    if y_norm < epsilon:
        # 选择与x轴不共线的任意向量
        if abs(x_axis[0]) > epsilon or abs(x_axis[1]) > epsilon:
            y_axis = np.array([-x_axis[1], x_axis[0], 0], dtype=np.float64)
        else:
            y_axis = np.array([0, 1, 0], dtype=np.float64)
        # 施密特正交化
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < epsilon:
            raise ValueError("无法构造正交的y轴")
    
    y_axis /= y_norm
    
    # 计算z轴（完成右手系）
    z_axis = np.cross(x_axis, y_axis)  # 注意叉积顺序
    z_norm = np.linalg.norm(z_axis)
    if abs(z_norm - 1) > epsilon:
        raise ValueError("z轴模长异常")
    
    # 构造旋转矩阵：全局→目标
    # 矩阵的列是目标坐标系的基向量在全局坐标系下的表示
    R_global_to_target = np.column_stack([x_axis, y_axis, z_axis]).T
    
    return R_global_to_target @ vector_global

def matrix_target_to_global(vector_target, vector_target_frame, epsilon=1e-10):
    """
    将目标坐标系下的向量转换到全局坐标系下
    
    Args:
        vector_target: 目标方向向量 (无需单位化)
        vector_target_frame: 目标坐标系下的向量
        epsilon: 数值稳定系数
    
    Returns:
        全局坐标系下的向量
    """
    # 使用相同的基向量构造方法
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_target_frame = np.asarray(vector_target_frame, dtype=np.float64)
    
    target_norm = np.linalg.norm(vector_target)
    if target_norm < epsilon:
        raise ValueError("目标方向向量模长过小，无法归一化")
    x_axis = vector_target / target_norm
    
    z_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis = np.cross(z_global, x_axis)
    y_norm = np.linalg.norm(y_axis)
    
    if y_norm < epsilon:
        if abs(x_axis[0]) > epsilon or abs(x_axis[1]) > epsilon:
            y_axis = np.array([-x_axis[1], x_axis[0], 0], dtype=np.float64)
        else:
            y_axis = np.array([0, 1, 0], dtype=np.float64)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < epsilon:
            raise ValueError("无法构造正交的y轴")
    
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    
    # 构造旋转矩阵：目标→全局（这是原矩阵的逆，即转置）
    R_target_to_global = np.column_stack([x_axis, y_axis, z_axis])
    
    return R_target_to_global @ vector_target_frame

class VehicleThread:
    """
    AirSim八叉树数据收集线程\n
    :parma uavname: 无人机名称
    """
    def __init__(self, uavname, stat_range=16, max_task_steps=500, start_point=np.array([0,0,-0.5], dtype=np.float64)):
        self.OctoMaps = om_o3d.OctreeMapManager(map_size=64, max_depth=7)  # O3D八叉树地图管理方法
        self.lidar = lidar.LidarModule(GlobalStartPoint=start_point)            # 激光雷达数据处理模块
        self.uavname = uavname                      # 无人机名称
        self.stat_range = stat_range              # 状态查询半径
        self.max_task_steps = max_task_steps         # 最大任务步数
        self.max_velocity = 10.0                   # 最大速度，用于速度归一化
        self.windowstat = [[], []]  # 滑动窗口状态空间
        self.RLstat = np.zeros((15,), dtype=np.float64)
        self.pos = start_point                      # 无人机初始位置
        self.position = self.pos.copy()             # 无人机当前位置(global)
        self.ReinForceLearnLock = threading.Lock()  # 状态更新锁，防止数据竞争
        self.target = np.zeros((0, 3), dtype=np.float64)  # 目标点位置
        self.alltargets = load_json_to_nx3_matrix('coordinates.json')      # 所有可选目标点位置
        self.stat_deque = deque(maxlen=40)          # 任务信息存储队列
        self.Target_Setting()                       # 设置任务目标点
        self.current_target_d_norm = 0              # 当前目标点欧式几何距离
        self.max_distance = 150                     # 最大任务距离，用于规范任务距离与进度解算
        self.start_time = time.time()
        self.ab_matrix = alpha_beta_matrix(stat_range, azimuth_step=2, elevation_step=2)  # 预计算角度矩阵
        self.Seqspace_directions = global_matrix_vector()  # 预计算方向向量矩阵
        # OctoMap模块预热
        self.running = True
        self.thread_lock = threading.Lock()
        
    def environment_reset(self, modesetting='easy'):
        """重置所有环境，包括八叉树地图"""        
        self.running = False                                                                                      
        self.current_target_d_norm = 0
        self.windowstat = [[], []]       
        self.RLstat = np.zeros((15,), dtype=np.float64)
        self.position = self.pos.copy()
        self.OctoMaps.reset()
        self.Target_Setting(mode=modesetting)
        self.stat_deque.clear()  
        self.start_time = time.time()
        

    def Target_Setting(self, mode='easy'):
        """设置任务目标点,从alltargets中随机选择1个目标点
           选择条件：
           1. 目标点与初始位置距离大于50米
           2. 目标点与初始位置距离小于100米
           """
        if mode=='easy':
            while True:
                x = 100*np.random.random()
                y = 100*np.random.random()
                z = -40+np.random.randint(-40,30)
                if 80 < np.linalg.norm([x, y, z]) < 150:
                    break
            self.target = np.array([x, y, z], dtype=np.float64) + self.pos
            self.max_distance = 150
            self.running = True
            print(f"{self.uavname}任务目标点设置为: {self.target}")
            return None
        if len(self.stat_deque)>0:
            start_point = self.position.copy() + self.stat_deque[-1]['position']
        else:
            start_point = self.pos.copy()
        # 计算所有目标点与初始位置的距离
        distances = np.linalg.norm(self.alltargets - start_point, axis=1)
        mask = (distances > 110) & (distances <= 150)
        valid_targets = self.alltargets[mask]
        # 随机选择目标点
        if len(valid_targets)>0:
            selected_indices = np.random.choice(len(valid_targets), size=1, replace=False)
            self.target = valid_targets[selected_indices]
            self.target = self.target.flatten()
            print(f"{self.uavname}任务目标点设置为: {self.target}")
        else:
            self.target = np.array([start_point[0]+40, start_point[1], start_point[2]], dtype=np.float64)
        self.max_distance = 150
        self.running = True

    def _Lidar_OctoMap_thread(self, lidar_data):
        """八叉树地图数据输入线程"""
        while self.OctoMaps.reset_index:
            time.sleep(0.005)
        else:
            try:
                self.thread_lock.acquire()
                timea = time.time()
                original_lidardata = self.lidar.LidarDataResolver(lidar_data)
                lidar_globaldata = self.lidar.LidarDataB2G(original_lidardata)
                self.OctoMaps.update_with_obstacles(lidar_globaldata['points_global'])
                timeb = time.time()
                self.thread_lock.release()
            except Exception as e:
                print(f'八叉树地图数据输入线程错误: {e}')
                self.thread_lock.release()
                            
    def windowstat_update(self, pos):
        if not self.OctoMaps.reset_index:
            try:
                with self.thread_lock:
                    probs, coords, prob_values = self.OctoMaps.get_region_probabilities(pos-self.stat_range, pos+self.stat_range, return_coords=True)
                if coords.size == 0:
                    self.windowstat = [[], []]
                else:
                    self.windowstat = [coords, prob_values]
            except Exception as e:
                print(f'八叉树地图状态更新错误: {e}')
                self.windowstat = [[], []]

    def _Lidar_RLstat_update(self, position, orientation, eular_angles, velocity, lidar_data):
        """强化学习状态空间更新\n
        连续状态空间MLpolicy:
            1. 无人机速度方向(1x3)\n
            2. 无人机速度大小(1x1)\n
            3. 无人机四元数(1x4)\n
            4. 当前任务目标点方向(1x3)\n
            5. 当前任务目标点距离归一化(1x1)\n
            6. 当前总体任务进度(1x1)\n
        八叉树地图空间Seqspace:\n
        描述无人机周围环境障碍物分布的角度查询矩阵，范围0-1归一化，障碍物越近值越小，无障碍为1
        """
        # 更新八叉树地图
        self.stat_deque.append({
            "position": position,
            "orientation": orientation,
            "eular_angles": eular_angles,
            "velocity": velocity,
            "target_vector": self.target - position - self.pos,
        })
        self.position = position + self.pos
        NewRLstat = np.zeros((15,), dtype=np.float64)
        self._Lidar_OctoMap_thread(lidar_data=lidar_data)
        # 计算水平和垂直方向角，确保水平角度在0-360范围内
        self.current_target_d_norm = np.linalg.norm(self.target - position - self.pos)
        NewRLstat[11:14] = np.clip((self.target - position)/np.linalg.norm(self.target - position), -1, 1)
        NewRLstat[14] = np.clip(self.current_target_d_norm/self.max_distance, 0, 1) 
        # 计算无人机状态
        # 将速度向量从全局坐标系转换到目标坐标系
        velocity_in_target = matrix_global_to_target(self.target - position - self.pos, velocity)
        v_norm = np.linalg.norm(velocity)
        NewRLstat[0:3] = velocity_in_target / v_norm if v_norm > 0 else np.zeros((3,), dtype=np.float64) # 速度方向归一化
        NewRLstat[3:6] = velocity/ v_norm if v_norm > 0 else np.zeros((3,), dtype=np.float64)  # 绝对坐标下速度方向归一化
        NewRLstat[6] = np.linalg.norm(velocity)/self.max_velocity  # 速度归一化
        NewRLstat[7:11] = orientation  # 四元数
        self.windowstat_update(position + self.pos)
        self.RLstat = NewRLstat
        yaw = np.degrees(eular_angles[2])  # 单位：弧度，0=朝北，顺时针为正（MAVROS convention）
        Seqspace = self.APF_SeqCalculate3D(yaw=yaw, cal_pos=position + self.pos)
        return self.RLstat.copy(), Seqspace

    def APF_SeqCalculate3D(self, 
                       azimuth_start=-180, 
                       azimuth_end=180, 
                       elevation_start=-90, 
                       elevation_end=90, 
                       yaw=0,  # 无人机当前偏航角（°，-180°~180°，0°=朝北）
                       cal_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                       azimuth_step=10, 
                       elevation_step=10):
        """
        基于无人机机体坐标系（0°指向无人机前方）计算最近障碍物分布。
        Args:
            azimuth_start: 机体坐标系方位角起点 (°)
            azimuth_end: 机体坐标系方位角终点 (°)
            elevation_start: 垂直高度角起点 (°)
            elevation_end: 垂直高度角终点 (°)
            yaw: 无人机当前偏航角 (°，-180°~180°，0°=朝北，90°=朝东)
            azimuth_step: 水平方位角步长 (°)
            elevation_step: 垂直高度角步长 (°)
        Returns:
            SeqMatrix: shape=(elevation_bins, azimuth_bins)，机体坐标系下最近障碍距离归一化值
        """
        # 获取新格式的windowstat数据
        coords_array, prob_array = self.windowstat
    
        # 1. 处理空障碍点情况
        if len(coords_array) == 0 or len(prob_array) == 0:
            # 计算bin数量
            elevation_bins = int((elevation_end - elevation_start) / elevation_step) 
            azimuth_bins = int((azimuth_end - azimuth_start) / azimuth_step) 
            return np.ones((elevation_bins, azimuth_bins), dtype=np.float64)
        
        # 2. 将yaw归一化到-180 - 180°范围
        yaw_normalized = np.round(((yaw + 180) % 360 - 180)/azimuth_step)*azimuth_step
        
        # 3. 筛选障碍点（概率>0.5）
        obstacle_mask = prob_array > 0.5
        if not np.any(obstacle_mask):
            # 计算bin数量
            elevation_bins = int((elevation_end - elevation_start) / elevation_step) 
            azimuth_bins = int((azimuth_end - azimuth_start) / azimuth_step)
            return np.ones((elevation_bins, azimuth_bins), dtype=np.float64)
        
        # 4. 提取障碍点坐标
        obstacle_coords = coords_array[obstacle_mask]
        
        # 5. 计算每个障碍点的球坐标（世界坐标系）
        x, y, z = obstacle_coords[:, 0] - cal_pos[0], obstacle_coords[:, 1] - cal_pos[1], obstacle_coords[:, 2] - cal_pos[2]
        
        # 计算水平距离（投影到XY平面）
        r_xy = np.sqrt(x**2 + y**2)
        
        # 计算世界坐标系方位角（alpha）：atan2(y, x)，范围[-180, 180]
        alpha_world = np.degrees(np.arctan2(y, x))
        
        # 计算高度角（beta）：arctan(z / r_xy)
        beta = np.degrees(np.arctan2(z, r_xy))
        
        # 6. 转换为机体坐标系方位角
        alpha_body = alpha_world - yaw_normalized
        alpha_body = ((alpha_body + 180) % 360) - 180  # 归一化到[-180,180]

        # 7. 计算障碍物距离
        dists = np.sqrt(x**2 + y**2 + z**2)
        max_dist = self.stat_range
        
        # 8. 计算bin数量
        azimuth_bins = int((azimuth_end - azimuth_start) / azimuth_step)
        elevation_bins = int((elevation_end - elevation_start) / elevation_step)
        
        # 9. 判断障碍点是否在无人机检测锥体内（使用机体坐标系角度）
        in_azimuth = (alpha_body >= azimuth_start) & (alpha_body <= azimuth_end)
        in_elevation = (beta >= elevation_start) & (beta <= elevation_end)
        in_cone = in_azimuth & in_elevation

        if not np.any(in_cone):
            return np.ones((elevation_bins, azimuth_bins), dtype=np.float64)
        
        # 10. 提取在锥体内的障碍点
        alpha_in = alpha_body[in_cone]
        beta_in = beta[in_cone]
        dists_in = dists[in_cone]
        
        # 11. 计算bin边界
        azimuth_bounds = np.linspace(azimuth_start, azimuth_end, azimuth_bins + 1)
        elevation_bounds = np.linspace(elevation_start, elevation_end, elevation_bins + 1)
        
        # 12. 分配到bin
        alpha_bin_indices = np.digitize(alpha_in, azimuth_bounds) - 1
        beta_bin_indices = np.digitize(beta_in, elevation_bounds) - 1
        
        # 防止索引越界（digitize已经处理边界，但为安全起见保留）
        alpha_bin_indices = np.clip(alpha_bin_indices, 0, azimuth_bins - 1)
        beta_bin_indices = np.clip(beta_bin_indices, 0, elevation_bins - 1)
        
        # 13. 初始化并填充最近距离
        SeqMatrix = np.ones((elevation_bins, azimuth_bins), dtype=np.float64) * max_dist

        # 14. 计算各方向上最小距离并归一化
        np.minimum.at(SeqMatrix, (beta_bin_indices, alpha_bin_indices), dists_in)
        SeqMatrix = np.clip(SeqMatrix, 0, max_dist) / max_dist
        return SeqMatrix

def alpha_beta_matrix(stat_range, azimuth_step, elevation_step):
    """
    Transform the Query message into Angle matrix
    Args:
        stat_range: the Query range
    Returns:
        Alpha-Beta Matrix step°/pixel
    """
    matrix = np.zeros((stat_range*2+1, stat_range*2+1, stat_range*2+1, 2))
    x, y, z = np.mgrid[-stat_range:stat_range+1,
                       -stat_range:stat_range+1,
                       -stat_range:stat_range+1]
    index = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).reshape(stat_range*2+1, stat_range*2+1, stat_range*2+1,3)
    matrix[:,:,:,0] = np.degrees(np.arctan2(index[:,:,:,1],index[:,:,:,0]))%360                               # alpha
    matrix[:,:,:,1] = np.degrees(np.arctan2(index[:,:,:,2],(np.sqrt(index[:,:,:,0]**2+index[:,:,:,1]**2))))   # beta
    # convert to azimuth and elevation
    matrix[:,:,:,0] = np.round(matrix[:,:,:,0]/azimuth_step)*azimuth_step
    matrix[:,:,:,1] = np.round(matrix[:,:,:,1]/elevation_step)*elevation_step
    matrix[matrix[:,:,:,0]==360,0] = 0
    return matrix

def global_matrix_vector(horizon_start=0, horizon_end=360, elevation_start=-90, elevation_end=90, azimuth_step=10, elevation_step=10):
    """
    计算APF_SeqCalculate3D函数输出矩阵对应的方向上的世界坐标下下单位向量
    """
    azimuth_angles = np.arange(horizon_start, horizon_end, azimuth_step)
    azimuth_angles = np.append(azimuth_angles, horizon_end)
    elevation_angles = np.arange(elevation_start, elevation_end, elevation_step)
    elevation_angles = np.append(elevation_angles, elevation_end)
    azimuth_bins = len(azimuth_angles) - 1
    elevation_bins = len(elevation_angles) - 1

    direction_vectors = np.zeros((elevation_bins, azimuth_bins, 3), dtype=np.float64)
    print(direction_vectors.shape)

    for i in range(elevation_bins):
        for j in range(azimuth_bins):
            azimuth_rad = np.radians((azimuth_angles[j] + azimuth_angles[j+1]) / 2)
            elevation_rad = np.radians((elevation_angles[i] + elevation_angles[i+1]) / 2)

            x = np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = np.sin(elevation_rad)

            direction_vectors[i, j, :] = np.array([x, y, z])

    return direction_vectors


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = VehicleThread("UAV01")
    # directions = global_matrix_vector()
    # print(directions[0:3,0,:])
    # w, x, y, z = 0.7071, 0, 0, 0.7071
    # Rg2b = np.array([[1-2*(y**2+z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
    #                 [2*(x*y + w*z), 1-2*(x**2 + z**2), 2*(y*z - w*x)],
    #                 [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x**2 + y**2)]], dtype=np.float64)
    # new_dir = directions @ Rg2b.T
    # print(new_dir[0:3,0,:])
    print(test.alltargets.shape)