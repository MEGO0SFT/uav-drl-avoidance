import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from socket import *
import pickle
from . import Octomap as om
from . import OctoMap_o3d as om_o3d
from . import DepthImageProcess as dip
import numpy as np
import time
import threading
import json
import os
from collections import deque
import torch
import torch.nn.functional as F
from numba import jit, njit

# 最后更新时间 2025年8月28日
# 作者： 石辅天
# 版本： stable v0.12.3

global OctoMap, device
global camara_space
camara_space = ["front_center", "bottom_center", "back_center"]
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

def matrix_Rg2t(vector_target, vector_global):
    """
    将全局坐标系下的速度向量转换到目标坐标系下
    
    Args:
        vector_target: 目标方向单位向量 (3D vector, [x,y,z])
        vector_global: 全局坐标系下的速度向量 (3D vector, [vx,vy,vz])
    
    Returns:
        vector_transformed: 目标坐标系下的速度向量 (3D vector)
    
    数学原理：
        1. 构建目标坐标系三个基向量：
           - x轴：目标方向向量（归一化）
           - y轴：全局z轴与x轴的叉积（确保垂直于x轴）
           - z轴：x轴与y轴的叉积（完成右手系）
        2. 构造旋转矩阵 R = [x_axis, y_axis, z_axis].T
        3. 应用旋转矩阵：v_target = R @ v_global
    """
    # 输入校验
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_global = np.asarray(vector_global, dtype=np.float64)
    
    # 1. 归一化目标方向向量
    x_axis = vector_target  # 输入已经是单位向量
    
    # 2. 计算临时y轴（与全局z轴和x轴正交）
    z_axis_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis_temp = np.cross(z_axis_global, x_axis)
    
    # 处理特殊情况：当x轴与全局z轴平行时
    if np.linalg.norm(y_axis_temp) < 1e-6:
        # 使用全局y轴作为备选
        y_axis_temp = np.array([0, 1, 0], dtype=np.float64)
        y_axis_temp = y_axis_temp - np.dot(y_axis_temp, x_axis) * x_axis
    
    # 归一化y轴
    y_axis = y_axis_temp / np.linalg.norm(y_axis_temp)
    
    # 3. 计算z轴（完成右手坐标系）
    z_axis = np.cross(y_axis, x_axis)
    
    # 构造旋转矩阵 (全局→目标的转换矩阵)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    
    # 应用旋转
    vector_transformed = rotation_matrix @ vector_global
    
    return vector_transformed


def matrix_Rt2g(vector_target, vector_velocity_target, epsilon=1e-10):
    """
    目标坐标系到全局坐标系的转换

    Args:
        vector_target: 目标方向向量(无需单位化)
        vector_velocity_target: 目标坐标系下的速度向量
        epsilon: 数值稳定系数

    Returns:
        全局坐标系下的速度向量
    """
    # 输入校验和归一化
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_velocity_target = np.asarray(vector_velocity_target, dtype=np.float64)
    
    x_axis = vector_target / (np.linalg.norm(vector_target) + epsilon)
    
    # 处理z轴平行情况
    z_axis_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis_temp = np.cross(z_axis_global, x_axis)
    
    if np.linalg.norm(y_axis_temp) < epsilon:
        # 更稳健的备选方案
        if abs(x_axis[0]) > epsilon or abs(x_axis[1]) > epsilon:
            y_axis_temp = np.array([-x_axis[1], x_axis[0], 0], dtype=np.float64)
        else:
            y_axis_temp = np.array([0, 1, 0], dtype=np.float64)
        y_axis_temp = y_axis_temp / (np.linalg.norm(y_axis_temp) + epsilon)
    
    y_axis = y_axis_temp / (np.linalg.norm(y_axis_temp) + epsilon)
    z_axis = np.cross(y_axis, x_axis)
    
    # 构造旋转矩阵
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
    
    return rotation_matrix @ vector_velocity_target

class VehicleThread:
    """
    AirSim八叉树数据收集线程\n
    :parma uavname: 无人机名称
    """
    def __init__(self, uavname, state_range=16, max_task_time=120, vertical_range=45, horizon_range=180, azimuth_step=15, elevation_step=15, simmode='traject', octomap_mode='o3d'):
        self.octomap_mode = octomap_mode
        if self.octomap_mode == 'o3d':
            self.OctoMaps = om_o3d.OctreeMapManager(map_size=64, max_depth=6)  # O3D八叉树地图管理方法
        else:
            self.OctoMaps = om.OctoMaps()  # OctoMap管理方法
        self.simmode = simmode                      # 仿真模式管理
        self.dip = dip.DepthImageProcessor()        # 深度图像处理
        self.uavname = uavname                      # 无人机名称
        self.state_range = state_range              # 状态查询半径
        self.max_task_time = max_task_time
        self.windowstate = np.zeros((state_range*2+1, state_range*2+1, state_range*2+1), dtype=np.float64)  # 滑动窗口状态空间
        self.RLstate = np.zeros((12,), dtype=np.float64)
        self.pos = np.zeros((3,), dtype=np.float64)  # 无人机位置
        self.ReinForceLearnLock = threading.Lock()  # 状态更新锁，防止数据竞争
        self.target = np.array([200,np.random.randint(-10,10),-4.0-np.random.randint(0,2)])
        self.target_norm = 0
        self.max_distance = 400
        self.octomap_img = None
        self.start_time = time.time()
        self.finished_target = 0   # 目标点计数器初始化
        self.state_deque = deque(maxlen=10)  # 速度平滑队列
        self.vertical_range = vertical_range
        self.horizon_range = horizon_range
        self.azimuth_step = azimuth_step
        self.elevation_step = elevation_step
        # 修改octomap_shape以符合高度角在前，方位角在后的顺序
        self.octomap_shape = (int(self.vertical_range*2/self.elevation_step)+1, int(self.horizon_range*2/self.azimuth_step)+1)
        # 使用固定步长提高角度精度
        self.alpha_beta_matrix = alpha_beta_matrix(self.state_range, self.azimuth_step, self.elevation_step)
        # OctoMap模块预热
        self.running = True
        self.thread_lock = threading.Lock()
        
    def environment_reset(self):
        """重置所有环境，包括八叉树地图"""
        self.finished_target = 0                                                                                                  # 目标点计数器初始化
        self.target_norm = 0
        self.windowstate = np.zeros((self.state_range*2+1, self.state_range*2+1, self.state_range*2+1), dtype=np.float64)         # 滑动窗口状态空间初始化
        self.RLstate = np.zeros((12,), dtype=np.float64)
        self.pos = np.zeros((3,), dtype=np.float64)                                                                               # 无人机位置初始化
        self.state_deque.clear()
        # OctoMap模块预热
        self.target = np.array([200+np.random.randint(-10,0),np.random.randint(-100,100),-4.0-np.random.randint(0,2)])
        self.start_time = time.time()
        if self.octomap_mode == 'o3d':
            self.OctoMaps.reset()
        else:
            self.OctoMaps.global_reset()

    def OctoMap_thread(self, depth_image, depth_img_pos, eular_angles):
        """八叉树地图数据输入线程"""
        if self.octomap_mode == 'o3d':
            while self.OctoMaps.reset_index:
                time.sleep(0.005)
            else:
                try:
                    self.thread_lock.acquire()
                    timea = time.time()
                    obstacle_points, free_points = self.dip._point_to_global(depth_image, depth_img_pos, eular_angles, "front_center")
                    obstacle_points[:,2], free_points[:,2] = -obstacle_points[:,2], -free_points[:,2]
                    self.OctoMaps.update_with_mixed_points(obstacle_points, free_points)
                    timeb = time.time()
                    if self.uavname == 'UAV01':
                        print(f'{self.uavname}八叉树地图数据输入线程处理时间: {timeb-timea:.4f}s, 点云数: {obstacle_points.shape[0]}, 空闲点云数:{free_points.shape[0]}')
                    self.thread_lock.release()
                except Exception as e:
                    print(f'八叉树地图数据输入线程错误: {e}')
                    self.thread_lock.release()
                            
    def windowstate_update(self, pos):
        if self.octomap_mode == 'o3d' and not self.OctoMaps.reset_index:
            try:
                new_window = np.ones((self.state_range*2+1, self.state_range*2+1, self.state_range*2+1), dtype=np.float64)*0.5
                pos[2] = -pos[2]
                with self.thread_lock:
                    probs, coords, prob_values = self.OctoMaps.get_region_probabilities(pos-self.state_range, pos+self.state_range, return_coords=True)
                if coords.size == 0:
                    self.windowstate = new_window
                else:
                    index = np.clip(np.round(coords - pos + self.state_range).astype(np.int32), 0, self.state_range*2)
                    new_window[index[:, 0], index[:, 1], index[:, 2]] = prob_values
                    self.windowstate = new_window
                    
            except Exception as e:
                print(f'八叉树地图状态更新错误: {e}')
        else:
            if self.OctoMaps.reset_index:
                return
            try:
                new_window = np.ones((self.state_range*2+1, self.state_range*2+1, self.state_range*2+1), dtype=np.float64)*0.5
                points_p = self.OctoMaps.circle_query_points(self.pos, self.azimuth_step, self.elevation_step, self.state_range)
                if points_p.size == 0:
                    self.windowstate = new_window
                else:
                    index = np.clip((points_p[:,0:3] - self.pos + self.state_range).astype(np.int32),0,self.state_range*2)
                    new_window[index[:,0], index[:,1], index[:,2]] = points_p[:,3]
                    self.windowstate = new_window
            except Exception as e:
                print(f'八叉树地图状态更新错误: {e}')

    def RLstate_update(self, position, orientation, eular_angles, velocity, depth_image, depth_img_pos):
        """强化学习状态空间更新\n
        连续状态空间MLpolicy:
        1. 无人机位置\n
        2. 无人机速度\n
        3. 无人机欧拉角\n
        4. 人工引力场状态\n
        八叉树地图空间CNN:\n
        0-2:x,y,z三方向求和融合空间
        3-5:yz,zx,xy二阶导求和融合空间
        """
        # 更新八叉树地图
        self.OctoMap_thread(depth_image=depth_image, depth_img_pos=depth_img_pos, eular_angles=orientation)
        # 计算水平和垂直方向角，确保水平角度在0-360范围内
        self.state_deque.append(velocity)
        self.target_norm = np.linalg.norm(self.target - position)
        self.pos = position
        self.RLstate[0:3] = (self.target - position)/(self.target_norm+1e-6)
        # 转换到目标坐标系下计算速度状态
        velocity_t = matrix_Rg2t(self.RLstate[0:3], velocity)
        self.RLstate[3:6] = velocity_t/(np.linalg.norm(velocity)+1e-6)
        self.RLstate[6:9] = velocity/np.linalg.norm(velocity+1e-6)
        self.RLstate[9] = eular_angles[2]/np.pi  # 偏航角归一化
        self.RLstate[11] = 1 -  self.target_norm/self.max_distance
        self.RLstate[10] = np.linalg.norm(velocity)/10  # 速度归一化
        vxy_theta = np.arctan2(velocity[1], velocity[0])
        self.windowstate_update(position)
        # 地图融合空间
        Seqspace = self.APF_SeqCalculate3D(0, 0, self.horizon_range, self.vertical_range, self.azimuth_step, self.elevation_step)
        Prob_area_distribution = self.Area_prob_calculate()
        if self.uavname == 'UAV01':
            print(f'{self.uavname}:{np.mean(Seqspace)},{velocity}')
        return self.RLstate.copy(), Seqspace, Prob_area_distribution


    def Area_prob_calculate(self):
        """
        计算根据角度划分的区域内障碍物概率分布\n
        划分规则：
        一共分为8x2个区域，其中按照水平角度划分(0-±45),(±45-±90),(±90-±135),(±135-180)\n
        垂直角度划分为(-45-0),(0-45)    
        """
        windowstate = self.windowstate
        azimuth_bins = [(-180,-165), (-165,-150), (-150,-135), (-135,-120), (-120,-105), (-105,-90), (-90,-75), (-75,-60),
                        (-60,-45), (-45,-30), (-30,-15), (-15,0), (0, 15), (15, 30), (30, 45), (45, 60),
                        (60, 75), (75, 90), (90, 105), (105, 120), (120, 135), (135, 150), (150, 165), (165, 180)]
        elevation_bins = [(-45, -30), (-30, -15),(-15,0),(0, 15), (15, 30),(30,45)]
        # 计算每个点与vector_v的夹角（方位角差）
        angles = self.alpha_beta_matrix  # (N,2): alpha, beta
        alpha = angles[:,:,:, 0]  # 方位角
        beta = angles[:,:,:, 1]   # 高度角
        # 选取azimuth_bins和elevation_bins范围内的点
        area_accuracy = np.zeros((7,24), dtype=np.float64)
        try:
            for i,azimuth in zip(range(24),azimuth_bins):
                for j,elevation in zip(range(7),elevation_bins):
                    in_azimuth = (alpha >= azimuth[0]) & (alpha < azimuth[1])
                    in_elevation = (beta >= elevation[0]) & (beta < elevation[1])
                    in_cone = in_azimuth & in_elevation
                    # 描述区域感知可靠性
                    area = windowstate[in_cone]-0.5
                    # 如果区域内全不可知，则说明该区域感知过弱，感知度为-1
                    if not np.any(area != 0):
                        area_accuracy[j,i] = 0
                    # 如果区域内并非全不可知，则计算感知度
                    else:
                        area_mean = np.mean(np.abs(area))*2
                        area_accuracy[j,i] = area_mean
            return area_accuracy
        except Exception as e:
            print(F"Area_prob_calculate error:{e}")
            return area_accuracy

    def Ground_alarm(self):
        """
        近地警报，判断无人机是否过低/过于贴近顶端
        """
        windowstate,min_h,max_h= self.windowstate,0,2*self.state_range
        if np.any(windowstate[self.state_range-1:self.state_range+2,self.state_range-1:self.state_range+2,0:self.state_range+1] > 0.5):
            # 如果在z轴正方向上有障碍物，表示无人机过低
            min_h = np.max(np.where(windowstate[self.state_range-1:self.state_range+2,self.state_range-1:self.state_range+2,0:self.state_range+1] > 0.5)[2])
        if np.any(windowstate[self.state_range-1:self.state_range+2,self.state_range-1:self.state_range+2,self.state_range+1:] > 0.5):
            max_h = np.min(np.where(windowstate[self.state_range-1:self.state_range+2,self.state_range-1:self.state_range+2,self.state_range+1:] > 0.5)[2])+self.state_range
            # 将高度归一化到0-1之间
        ground_alarm =  (self.state_range - min_h) / (self.state_range)-1
        up_alarm = (2*self.state_range-max_h)/self.state_range
        if abs(up_alarm) > abs(ground_alarm):
            return up_alarm
        else:
            return ground_alarm

    def APF_SeqCalculate3D(self, vector_v_horizon, vector_v_vertical, horizon_range=60, vertical_range=30, azimuth_step=10, elevation_step=10):
        """
        基于输入的速度方向角vector_v，输出以该方向为中心的锥体射线障碍查询矩阵。
        Args:
            vector_v: 水平速度向量方向角度(0-360°)，作为锥体中心
            vertical_range: 方位角锥体范围(°)
            horizon_range: 高度角锥体范围(°)
            azimuth_step: 方位角步长
            elevation_step: 高度角步长
        Returns:
            SeqMatrix: shape=(azimuth_bins, elevation_bins)，锥体内为最近障碍距离归一化值，锥体外为1
        """
        windowstate = self.windowstate
        obstacle_mask = windowstate > 0.5
        azimuth_bins = int(horizon_range*2 / azimuth_step) + 1 if horizon_range*2%360 != 0 else int(horizon_range*2 / azimuth_step)
        elevation_bins = int(vertical_range*2 / elevation_step) + 1
        max_dist = self.state_range
        angles = self.alpha_beta_matrix  # (N,2): alpha, beta
        # 计算每个点与vector_v的夹角（方位角差）
        alpha = angles[:,:,:, 0]  # 方位角
        beta = angles[:,:,:, 1]   # 高度角
        # 计算锥体内mask
        # 方位角差（周期性）
        vector_v_horizon= np.round(vector_v_horizon)//azimuth_step*azimuth_step
        vector_v_vertical = np.round(vector_v_vertical)//elevation_step*elevation_step
        alpha_diff = (alpha - vector_v_horizon + 180) % 360 - 180  # [-180,180]
        beta_diff = beta - vector_v_vertical
        in_azimuth = np.abs(alpha_diff) <= horizon_range
        in_elevation = np.abs(beta_diff) <= vertical_range
        in_cone = in_azimuth & in_elevation
        # 没有障碍物，全部为最大距离
        if not np.any(obstacle_mask&in_cone):
            return np.ones((elevation_bins, azimuth_bins), dtype=np.float64)

        coords = np.argwhere(obstacle_mask&in_cone)
        dists = np.linalg.norm(coords - self.state_range, axis=1)
        
        alpha_bins = np.floor((alpha[obstacle_mask&in_cone] - (vector_v_horizon - horizon_range)) / azimuth_step).astype(int)
        beta_bins = np.floor((beta[obstacle_mask&in_cone] - (vector_v_vertical - vertical_range)) / elevation_step).astype(int)
        # 防止越界
        alpha_bins = np.clip(alpha_bins, 0, azimuth_bins-1)
        beta_bins = np.clip(beta_bins, 0, elevation_bins-1)
        # 初始化为最大距离
        SeqMatrix = np.ones((elevation_bins, azimuth_bins), dtype=np.float64) * max_dist
        # 用np.minimum.at矢量化投影，注意交换索引顺序使得elevation_bins在前（高度角索引0），azimuth_bins在后（方位角索引1）
        np.minimum.at(SeqMatrix, (beta_bins, alpha_bins), dists)
        # 归一化到[0,1]，障碍物近为小值，无障碍为1
        SeqMatrix = np.clip(SeqMatrix, 0, max_dist) / max_dist
        return SeqMatrix

    def Octomap_query_update(self):
        while self.running:
            self.windowstate = self.OctoMaps.visualize_octomap_query_nopltshow(self.pos, self.state_range, self.start_time)
            # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # self.octomap_img = img
            time.sleep(0.1)
            # self.PygameDataManager.octomapfig2pygame(self.octomap_img)

    def OctoMapstate_buffer(self):
        """
        通过经验池存储采集到的状态空间，以用于后续训练
        """
        while self.running:
            if self.windowstate_buffer_is_available:
                self.windowstate_buffer.append({
                    "timestamp": time.time() - self.start_time,
                    "state": self.windowstate.tolist(),
                    "value": grad_cal(self.windowstate).tolist(),
                })
            print(f"状态缓冲区大小: {len(self.windowstate_buffer)}")
            time.sleep(1.5)
        filename = f"state_buffer_{self.uavname}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(list(self.windowstate_buffer), f, indent=4)
        print("状态缓冲区已保存到state_buffer.json文件中")
    
    def AirSim_traject(self):
        """
        无人机飞行轨迹规划\n
        该轨迹点方法生成的点是固定的
        """
        targetpoints = np.array([[0, 0, -5],
                                 [0,-100,-5],
                                 [200,-100,-5],
                                 [200,100,-5],
                                 [0,-100,-5],
                                #  [5,-2,-5],
                                #  [5,0,-3],
                                #  [5,0, -6],
                                #  [5,3, -3],
                                #  [5,-2, -4]
                                #  [15,-7,-4],
                                #  [15,-15,-5],
                                #  [18,-20,-5],
                                #  [27,-21,-7],
                                #  [32,-20,-5]
                          ])
        return targetpoints
    
    def AirSim_targetfollowing(self):
        targetpoints = self.AirSim_traject()
        target_thread = True
        for target in targetpoints:
            if self.reset_flag.is_set():
                time.sleep(0.1)  # 等待重置完成
                break
            if target_thread == True:
                sim = 1
                self.target = target
            else:
                break
            while sim:
                if self.vehicle_state == 'collision':
                    target_thread = False
                    break
                self.position = self.ap.motion_buffer[-1]['position']
                self.pos = np.array([self.position.x_val, self.position.y_val, self.position.z_val])
                self.ap.movebyVelocity(self.target, self.pos, self.uavname, (1, 45))
                if np.linalg.norm(self.pos - target) < 2:
                    sim = 0
                    self.finished_target += 1
                time.sleep(0.1)
        if self.vehicle_state != 'collision':
            print("最终目标点到达")
            self.sender.sendto(pickle.dumps('next_episode'), (self.ip, 2000+int(self.uavname[3:])))
            self.vehicle_state = 'Finish'
            self.ap.exist = False
            self.running = False
            self.PygameDataManager.updating_thread_running = False

    def visualize_state_action(self):
        while True:
            print(f'State:\n{self.RLstate},\nAction:\n{self.action_space}')
            print(f'Query:{np.sum(self.windowstate)}')
            time.sleep(2)

def AirSim_datacatch(om):
    targetpoints = om.AirSim_traject()
    for target in targetpoints:
        sim = 1
        om.target = target
        while sim:
            om.position = om.ap.motion_buffer[-1]['position']
            om.pos = np.array([om.position.x_val, om.position.y_val, om.position.z_val])
            om.ap.movebyVelocity(om.target, om.pos, om.uavname, (1,0))
            if np.linalg.norm(om.pos - target) < 2:
                sim = 0
            time.sleep(0.1)
    om.ap.exist=False
    octomap = om.OctoMaps.OctoMap_dict[(0,0,0)]

def is_thread_running(thread_name: str) -> bool:
    """判断是否存在指定名称且处于运行状态的线程"""
    for thread in threading.enumerate():
        if thread.name == thread_name and thread.is_alive():
            return True
    return False

def grad_cal(state_block):
    global device
    b = torch.tensor(state_block, dtype=torch.float64, device=device)  # shape: (17,17,17)
    
    # --- 严格按顺序求二阶偏导 ---
    # 1. yz方向：先对y求导，再对z求导
    dy = b[:, 1:, :] - b[:, :-1, :]       # 一阶y导数 (17,16,17)
    dy_dz = dy[:, :, 1:] - dy[:, :, :-1]   # 二阶yz导数 (17,16,16)
    grad_yz = dy_dz      # 取绝对值幅度
    
    # 2. zx方向：先对z求导，再对x求导
    dz = b[:, :, 1:] - b[:, :, :-1]       # 一阶z导数 (17,17,16)
    dz_dx = dz[1:, :, :] - dz[:-1, :, :]   # 二阶zx导数 (16,17,16)
    grad_zx = dz_dx
    
    # 3. xy方向：先对x求导，再对y求导
    dx = b[1:, :, :] - b[:-1, :, :]       # 一阶x导数 (16,17,17)
    dx_dy = dx[:, 1:, :] - dx[:, :-1, :]   # 二阶xy导数 (16,16,17)
    grad_xy = dx_dy
    
    # --- 沿未求导的轴求和 ---
    # yz梯度已对y,z求导 → 沿x求和
    yz_sum = torch.sum(grad_yz, dim=0)     # (1,16,16) → 需补零到17x17
    yz_sum = F.pad(yz_sum, (0,1,0,1)).squeeze(0)     # (17,17)
    
    # zx梯度已对z,x求导 → 沿y求和
    zx_sum = torch.sum(grad_zx, dim=1)     # (16,1,16) → 补零
    zx_sum = F.pad(zx_sum, (0,1,0,1)).squeeze(1)      # (17,17)
    
    # xy梯度已对x,y求导 → 沿z求和
    xy_sum = torch.sum(grad_xy, dim=2)     # (16,16,1) → 补零
    xy_sum = F.pad(xy_sum, (0,1,0,1))          # (17,17)
    
    # 按x,y,z顺序堆叠
    if device == 'cpu':
        return torch.stack([yz_sum, zx_sum, xy_sum], dim=-1).numpy()  # (17,17,3)
    else:
        return torch.stack([yz_sum, zx_sum, xy_sum], dim=-1).cpu().numpy()  # (17,17,3)

def test_grad_cal():
    # 创建测试输入 (17x17x17)
    test_input = np.random.rand(17, 17, 17)
    
    # 运行函数
    result = grad_cal(test_input)
    
    # 验证输出形状
    print("输出形状:", result.shape)  # 应为 (17,17,3)
    
    # 验证填充效果
    print("yz_sum右下角值:", result[16, 16, 0])  # 应为0（填充值）
    print("zx_sum右下角值:", result[16, 16, 1])  # 应为0
    print("xy_sum右下角值:", result[16, 16, 2])  # 应为0

def alpha_beta_matrix(state_range, azimuth_step, elevation_step):
    """
    Transform the Query message into Angle matrix
    Args:
        state_range: the Query range
    Returns:
        Alpha-Beta Matrix 15°/pixel
    """
    matrix = np.zeros((state_range*2+1, state_range*2+1, state_range*2+1, 2))
    x, y, z = np.mgrid[-state_range:state_range+1,
                       -state_range:state_range+1,
                       -state_range:state_range+1]
    index = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).reshape(state_range*2+1, state_range*2+1, state_range*2+1,3)
    matrix[:,:,:,0] = np.degrees(np.arctan2(index[:,:,:,1],index[:,:,:,0]))%360                   # alpha
    matrix[:,:,:,1] = np.arctan2(index[:,:,:,2],(np.sqrt(index[:,:,:,0]**2+index[:,:,:,1]**2)))   # beta
    # convert to azimuth and elevation
    matrix[:,:,:,0] = np.round(matrix[:,:,:,0]/azimuth_step)*azimuth_step
    matrix[:,:,:,1] = np.round(matrix[:,:,:,1]/np.pi*180/elevation_step)*elevation_step
    matrix[matrix[:,:,:,0]==360,0] = 0
    return matrix



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = VehicleThread("UAV01")
    threading.Thread(target=test.capture_scene_thread, daemon=True).start()
    threading.Thread(target=test.PygameDataManager.updating_thread, daemon=True, args=(test.uavname,)).start()
    vthread = threading.Thread(target=AirSim_datacatch, args=(test,), daemon=False)
    octomapqueryupdate_thread = threading.Thread(target=test.Octomap_query_update, daemon=True)
    vthread.start()
    test.windowstate_buffer_is_available= True
    #threading.Thread(target=test.OctoMapstate_buffer, daemon=False).start()
    octomapqueryupdate_thread.start()
    
    # 可视化示例 - 等待一段时间后再可视化，确保有数据
    time.sleep(5)
    
    # 创建保存目录
    viz_dir = "windowstate_viz"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # 可视化当前windowstate
    print("正在生成3D可视化...")
    test.visualize_windowstate_3d(save_path=os.path.join(viz_dir, "windowstate_3d.png"))
    
    print("正在生成切片可视化...")
    test.visualize_windowstate_slices(save_path=os.path.join(viz_dir, "windowstate_slices.png"))
    
    print(f"可视化结果已保存到 {viz_dir} 目录")
    
    # 如果需要交互式可视化，取消下面的注释
    # test.visualize_windowstate_interactive(save_dir=viz_dir)

    test.running=False
