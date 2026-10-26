import numpy as np
import airsim
from . import Octomap as OM
from airsim.types import Pose, Vector3r, Quaternionr
from collections import deque
import threading
import time
from scipy.spatial.transform import Rotation, Slerp
from simple_pid import PID
import cv2
# import pygame
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import queue

# 最后更新时间 2025年4月28日 16:21:45
# 作者： 石辅天
# 版本： stable v1.2.0
# 待更新：
global AirSim_Lock,MOTION_BUFFER_SIZE  # AirSim锁,防止线程冲突
AirSim_Lock = threading.Lock()
MOTION_BUFFER_SIZE = 50   # 运动数据缓冲区大小
default_client = airsim.MultirotorClient()
__all__ = ['_eular_angles_to_rotation_matrix', 'rotation_point', 'AirSim_Processor']

def _eular_angles_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角转换为旋转矩阵
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    return R

def rotation_point(eular_angles, points):
    # 将相机坐标系下的点转换到全局坐标系
    R = _eular_angles_to_rotation_matrix(*eular_angles)
    points_global = points @ R.T
    return points_global

class AirSim_Processor:
    def __init__(self, vehicle_name, ip="127.0.0.1", port=41451):
        if ip == "":
            self.Client = airsim.MultirotorClient()
        else:
            try:
                self.Client = airsim.MultirotorClient(ip=ip,port=port)
            except:
                print("AirSim进程管理器声明失败，请确认ip和port都正确输入！")
        self.Client.confirmConnection()
        self.motion_buffer = deque(maxlen=MOTION_BUFFER_SIZE)  # 运动数据缓冲区
        self.vehicle_name = vehicle_name
        self.exist = True
        # 获取相机参数
        self.K = np.array([[640, 0, 640],
                           [0, 640, 400],
                           [0, 0, 1]], dtype=np.float32)
        self.motionthread = threading.Thread(target=self.motion_buffer_update,name=self.vehicle_name+"motion_buffer", daemon=True)
    
    def motion_buffer_update(self):
        global AirSim_Lock
        """
        用于更新无人机的状态信息(测量值)
        """
        # 定义异步方法
        try:
            while self.exist:
                with AirSim_Lock:
                    new_state = self.Client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
                self.motion_buffer.append({
                    "timestamp": time.time(),
                    "position": new_state.position,
                    "orientation": new_state.orientation,
                    "velocity": new_state.linear_velocity
                })
                time.sleep(0.1)
            self.motion_buffer.clear()
        except Exception as e:
            print(f"motion_buffer_update thread is crashed!")


    def get_depth_image(self, vehicle_name, which_camara):
        # 获取深度图像
        global AirSim_Lock # AirSim锁,防止线程冲突
        if which_camara == "front_center":
            AirSim_Lock.acquire()
            responses = self.Client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], vehicle_name=vehicle_name)
            time_stamp = float(responses[0].time_stamp)/1e9
            AirSim_Lock.release()
        elif which_camara == "bottom_center":
            AirSim_Lock.acquire()
            responses = self.Client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], vehicle_name=vehicle_name)
            time_stamp = float(responses[0].time_stamp)/1e9
            AirSim_Lock.release()
        elif which_camara == "back_center":
            AirSim_Lock.acquire()
            responses = self.Client.simGetImages([
                airsim.ImageRequest("4", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], vehicle_name=vehicle_name)
            time_stamp = float(responses[0].time_stamp)/1e9
            AirSim_Lock.release()
        else:
            print(f"Invalid camera type: {which_camara}. Use 'front_center' to return img.")
            AirSim_Lock.acquire()
            responses = self.Client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], vehicle_name=vehicle_name)
            time_stamp = float(responses[0].time_stamp)/1e9
            AirSim_Lock.release()
        return responses, time_stamp
    
    def motion_compensation(self, responses, capture_time):
        """运动补偿核心算法\n
        :param responses: 深度图像源数据
        :param capture_time: 捕获时间戳"""
        img_position = responses[0].camera_position
        depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(responses[0].height, responses[0].width)
        if len(self.motion_buffer) < 2:
            motion_buffer = self.motion_buffer[-1]
            motion_buffer['position'] = img_position
            return depth_img, motion_buffer
        # 精确查找时间窗口
        timestamps = np.array([p['timestamp'] for p in self.motion_buffer])
        current_time = time.time()
        validtimestamps = [t for t in self.motion_buffer if current_time - t['timestamp'] < 0.1]  # 200ms有效窗口
        if len(validtimestamps) < 2:
            motion_buffer = self.motion_buffer[-1]
            motion_buffer['position'] = img_position
            return depth_img, motion_buffer
        # 找到capture_time的插入位置
        idx = np.searchsorted(timestamps, capture_time)
        if idx == 0 or idx == len(timestamps):
            print(idx, len(timestamps), len(self.motion_buffer))
            return (depth_img, self.motion_buffer[0]) if idx == 0 else (depth_img, self.motion_buffer[-1]) # 无法插值
        alpha = (capture_time - timestamps[idx-1]) / (timestamps[idx] - timestamps[idx-1])

        # 线性插值位姿
        start_pose = interpolate_pose(self.motion_buffer[idx-1], self.motion_buffer[idx], alpha)
        end_pose = self.motion_buffer[-1]  # 当前最新位姿
        start_pos = np.array([start_pose['position'].x_val, start_pose['position'].y_val, start_pose['position'].z_val], dtype=np.float32)
        end_pos = np.array([end_pose['position'].x_val, end_pose['position'].y_val, end_pose['position'].z_val], dtype=np.float32)
        # 计算平移变化
        # 计算旋转变化（四元数转换为旋转矩阵）
        q_start = start_pose["orientation"]
        q_end = end_pose["orientation"]
        R_start = Rotation.from_quat([q_start.x_val, q_start.y_val, 
                                    q_start.z_val, q_start.w_val]).as_matrix()
        R_end = Rotation.from_quat([q_end.x_val, q_end.y_val,
                                  q_end.z_val, q_end.w_val]).as_matrix()
        
        # 应用补偿变换
        return apply_transform(self.K, depth_img, R_start, start_pos, R_end, end_pos), end_pose

    def movebyAngleThrottle(self, action, vehicle_name):
        """以俯仰偏航滚转角和油门控制移动"""
        global AirSim_Lock # AirSim锁,防止线程冲突
        AirSim_Lock.acquire()
        self.Client.moveByAngleThrottleAsync(
            float(action[0]),  # pitch angle
            float(action[1]),  # row angle
            float(action[3]),  # throttle
            float(action[2]),  # yaw
            duration=0.5,
            vehicle_name=vehicle_name
        )
        AirSim_Lock.release()

    def movebyVelocity(self, target, pos, vehicle_name :str, Mode_and_Yaw :tuple, v_max=7):
        """以偏航角和期望速度控制移动
        Args:
            target: 目标位置
            pos: 当前无人机位置
            vehicle_name: 无人机名称
            Mode_and_Yaw: 模式和期望偏航角
            v_max: 最大速度"""
        global AirSim_Lock # AirSim锁,防止线程冲突
        pid_x = PID(0.8, 0.2, 0.1, setpoint=target[0])
        pid_y = PID(0.8, 0.2, 0.1, setpoint=target[1])
        pid_z = PID(1.2, 0.3, 0.2, setpoint=target[2])
        mode, Yaw = Mode_and_Yaw[0], Mode_and_Yaw[1]
        v = np.zeros((3,1))
        v[0] = pid_x(pos[0]).astype(np.float32)
        v[1] = pid_y(pos[1]).astype(np.float32)
        v[2] = pid_z(pos[2]).astype(np.float32)
        if np.linalg.norm(v) > v_max:
            Vector = v/np.linalg.norm(v)
            v = v_max*Vector
        if mode == 0:
            # mode=0，表示无人机始终朝向当前速度方向
            with AirSim_Lock:
                self.Client.moveByVelocityAsync(float(v[0]),
                                                float(v[1]),
                                                float(v[2]),
                                            0.5,
                                            vehicle_name=vehicle_name,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0))
        else:
            # mode !=0，表示无人机朝向由Yaw期望决定
            with AirSim_Lock:
                self.Client.moveByVelocityAsync(float(v[0]),
                                                float(v[1]),
                                                float(v[2]),
                                            0.5,
                                            vehicle_name=vehicle_name,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=False,yaw_or_rate=Yaw))
                
    def RL_movebyVelocity(self, v, vehicle_name :str, Yaw):
        """以偏航角和期望速度控制移动
        Args:
            target: 目标位置
            pos: 当前无人机位置
            vehicle_name: 无人机名称
            Mode_and_Yaw: 模式和期望偏航角
            v_max: 最大速度"""
        global AirSim_Lock # AirSim锁,防止线程冲突
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        with AirSim_Lock:
            self.Client.moveByVelocityAsync(vx,
                                            vy,
                                            vz,
                                            0.5,
                                            vehicle_name=vehicle_name,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=False,yaw_or_rate=Yaw))
                
    def reset(self, vehicle_name):
        # 重置无人机状态到初始位置
        global AirSim_Lock
        pose = Pose(Vector3r(0, 0, -0.8), Quaternionr(0, 0, 0, 1))
        pose2 = Pose(Vector3r(np.random.randint(0,2),np.random.randint(-50,50),-2-np.random.randint(3,5)), Quaternionr(0, 0, 0, 1))
        AirSim_Lock.acquire()
        self.Client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        self.Client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=vehicle_name).join()
        self.Client.simSetVehiclePose(pose2, True, vehicle_name=vehicle_name)
        self.Client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=vehicle_name).join()
        self.Client.simSetVehiclePose(pose2, True, vehicle_name=vehicle_name)
        AirSim_Lock.release()

    def setPostion(self, position, vehicle_name, quaternion=(0, 0, 0, 1)):
        # 设置无人机位置
        global AirSim_Lock
        pose = Pose(Vector3r(*position), Quaternionr(*quaternion))
        AirSim_Lock.acquire()
        self.Client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        self.Client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=vehicle_name)
        AirSim_Lock.release()

    def Api_control_enable(self, vehicle_name):
        # 允许AirSim启用API控制，并使得被控制的无人机起飞
        global AirSim_Lock
        AirSim_Lock.acquire()
        self.Client.enableApiControl(True, vehicle_name=vehicle_name)
        self.Client.armDisarm(True, vehicle_name=vehicle_name)
        self.Client.takeoffAsync(vehicle_name=vehicle_name)
        AirSim_Lock.release()

    def DepthImage_Processing(self, vehicle_name, which_camara):
        # 获取无人机前置摄像机图像信息，分辨率为256x144，0为rgb，1为depthplanar
        responses, depth_time= self.get_depth_image(vehicle_name, which_camara)
        # 处理深度图像
        # img_position = responses[0].camera_position
        # depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        # depth_image = depth_image.reshape(responses[0].height, responses[0].width)
        depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(responses[0].height, responses[0].width)
        motion_buffer = self.motion_buffer[-1]
        return depth_img, motion_buffer
        # return self.motion_compensation(responses, depth_time)
    
    def AirSim_simGetGroundTruthKinematics(self, vehicle_name, mode='eular'):
        """
        处理无人机状态信息，输出位置、姿态角、速度
        pos, orientanion, velocity = output
        """
        global AirSim_Lock
        AirSim_Lock.acquire()
        value = self.Client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
        AirSim_Lock.release()
        pos, orientation, velocity = value.position, value.orientation, value.linear_velocity
        pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        q = np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val], dtype=np.float32)
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        eular_angles = np.array([roll, pitch, yaw], dtype=np.float32)
        velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val], dtype=np.float32)
        if mode == 'eular':
            return pos, eular_angles, velocity
        else:
            return pos, q, velocity
    
    def OctoMap2State(self, OctoMaps: OM.OctoMaps, vehicle_name ,last_location=0):
        """
        废弃\n
        :param OctoMaps: OM.OctoMaps类
        :param vehicle_name: 无人机名称,例如"UAV01"
        """
        state = np.zeros((1, 32, 32, 32), dtype=np.float64)
        pos, eular_angles, velocity = self.AirSim_simGetGroundTruthKinematics(vehicle_name)
        if last_location == 0:
            location, Octomap = OctoMaps.UAVinWhichOctomap(pos)
        else:   
            location, Octomap = OctoMaps.UAVinWhichOctomap(pos, last_location=last_location)
        points, prob = Octomap.traverse()
        center = Octomap.root.center
        halfsize = Octomap.root.size/2
        for i in range(len(prob)):
            # 如果概率大于0.5，就将该节点的位置信息写入state，否则跳过
            if prob[i] > 0.5:
                point = points[i]-center
                x = np.clip(0, 31, int(pos[0]-center[0]+halfsize))
                y = np.clip(0, 31, int(pos[1]-center[1]+halfsize))
                z = np.clip(0, 31, int(pos[2]-center[2]+halfsize))
                state[0, x, y, z] = max(state[0,x,y,z], prob[i])
        # 查询无人机的位置信息，将其转换为state的坐标，将该位置的值设置为2
        x = np.clip(0, 31, int(pos[0]-center[0]+halfsize))
        y = np.clip(0, 31, int(pos[1]-center[1]+halfsize))
        z = np.clip(0, 31, int(pos[2]-center[2]+halfsize))
        state[0, x, y, z] = 2
        return state, location
    # def Position2Octomap(self,)

def interpolate_pose(pose_a, pose_b, alpha):
    """
    在两组位姿之间进行线性插值
    参数:
        pose_a: 起始位姿字典，包含 position(Vector3r), orientation(Quaternionr), timestamp
        pose_b: 结束位姿字典，格式同pose_a
        alpha: 插值系数 (0.0-1.0)
    返回:
        插值后的位姿字典（包含 position, orientation, timestamp）
    """
    # 边界处理
    alpha = np.clip(alpha, 0.0, 1.0)
    if alpha == 0.0:
        return pose_a.copy()
    if alpha == 1.0:
        return pose_b.copy()

    # 位置线性插值
    pos_a = np.array([pose_a['position'].x_val, 
                     pose_a['position'].y_val,
                     pose_a['position'].z_val])
    pos_b = np.array([pose_b['position'].x_val,
                     pose_b['position'].y_val,
                     pose_b['position'].z_val])
    interp_pos = pos_a + alpha * (pos_b - pos_a)
    
    # 四元数球面线性插值 (SLERP)
    q_a = [
        pose_a['orientation'].x_val,
        pose_a['orientation'].y_val,
        pose_a['orientation'].z_val,
        pose_a['orientation'].w_val
    ]
    q_b = [
        pose_b['orientation'].x_val,
        pose_b['orientation'].y_val,
        pose_b['orientation'].z_val,
        pose_b['orientation'].w_val
    ]
    rotations = Rotation.from_quat([q_a, q_b])
    slerp = Slerp([0, 1], rotations)
    interp_rot = slerp([alpha])[0]
    # interp_rot = rot_a.slerp(rot_b, alpha)[0]  # Rotation对象
    
    # 时间戳插值
    interp_time = pose_a['timestamp'] + alpha * (pose_b['timestamp'] - pose_a['timestamp'])
    
    # 组装返回结果
    return {
        'position': Vector3r(interp_pos[0], interp_pos[1], interp_pos[2]),
        'orientation': Quaternionr(
            x_val=interp_rot.as_quat()[0],
            y_val=interp_rot.as_quat()[1],
            z_val=interp_rot.as_quat()[2],
            w_val=interp_rot.as_quat()[3]
        ),
        'timestamp': interp_time
    }

def apply_transform(K, depth_img, start_rot, start_pos, end_rot, end_pos):
    """应用几何变换到深度图，优化矩阵运算和向量化赋值"""
    h, w = depth_img.shape
    comp_depth = np.zeros_like(depth_img, dtype=np.float32)
    
    # 生成像素坐标网格（优化实现）
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.vstack([u.ravel(), v.ravel(), np.ones(w*h)])
    
    # 反投影到3D空间
    inv_K = np.linalg.inv(K)
    cam_points = (inv_K @ pixels) * depth_img.ravel()
    
    # 1. 相机坐标系 -> 世界坐标系（使用图像捕获时位姿）
    world_points = start_rot @ cam_points + start_pos.reshape(3,1)
    
    # 2. 世界坐标系 -> 当前相机坐标系（使用当前位姿）
    current_cam_points = end_rot.T @ (world_points - end_pos.reshape(3,1))
    
    # 重新投影到图像平面
    reproj = K @ current_cam_points
    reproj = reproj / (reproj[2, :] + 1e-6)  # 防止除零
    
    # 有效点过滤（向量化实现）
    valid_mask = (reproj[0] >= 0) & (reproj[0] < w) & (reproj[1] >= 0) & (reproj[1] < h)
    
    # 坐标映射与赋值（优化内存访问）
    x_new = np.clip(reproj[0, valid_mask], 0, w-1).astype(int)
    y_new = np.clip(reproj[1, valid_mask], 0, h-1).astype(int)
    valid_z = current_cam_points[2, valid_mask]
    
    # 使用高效散射操作
    np.maximum.at(comp_depth, (y_new, x_new), valid_z)
    
    return comp_depth

# 使用示例
if __name__ == "__main__":
    # 创建八叉树地图
    AP = AirSim_Processor("UAV01")
    for i in range(10):
        time_a = time.time()
        img= AP.DepthImage_Processing("UAV01")
        print(AP.motion_buffer[-1]['position'])
        time_b = time.time()
        print(time_b-time_a)
        AP.exist = False
