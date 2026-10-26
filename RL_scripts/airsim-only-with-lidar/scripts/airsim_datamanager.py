import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from collections import deque
import numpy as np
import os
import socket
import struct
import threading
import time
import pickle
import math
import json
from typing import List, Dict, Any

MAX_PACKET_SIZE = 62000

# 获取本机ip地址
def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'

def target_points_read():
    """读取点文件"""
    try:
        with open('coordinates.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        points_list = []
        for point in data['points']:
            points_list.append([point['x'], point['y'], point['z']])
        points_matrix = np.array(points_list)
        print(f"读取到{len(points_list)}个坐标点，矩阵形状: {points_matrix.shape}")
        return points_matrix
    except FileNotFoundError:
        print("警告: 未找到coordinates.json文件，使用默认坐标点")
        return None
    except Exception as e:
        print(f"读取coordinates.json文件失败: {e}")
        return None

class AirSim_Data_Manager:
    def __init__(self, server_port=5812, airsim_ip='', airsim_port=41451, trainmanager_ip='', uav_number=6):
        self.client = airsim.MultirotorClient(ip=airsim_ip, port=airsim_port)
        self.ip = get_real_ip()
        self.port = server_port
        self.udp_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sender.bind((self.ip, self.port))
        self.udp_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_recv.bind((self.ip, self.port + 1))
        self.trainmanager_ip = trainmanager_ip if trainmanager_ip != '' else self.ip
        self.uav_number = uav_number
        
        # 初始化重生点系统（只读取一次）
        self.origin_pos = np.array([0, 0, 0])  # 原点位置
        self.global_points = self._load_spawn_points()  # 一次性读取所有点
        self.used_spawn_points = deque(maxlen=uav_number)  # 已使用的重生点
        
        # 数据队列和无人机设置
        self.Data_Queue = [deque(maxlen=100) for _ in range(self.uav_number)]
        self.task_deque = deque(maxlen=10)
        
        # 初始化无人机设置，使用获取的重生点
        self.uav_settings = []
        for i in range(self.uav_number):
            self.uav_settings.append({
                "uavname": f"UAV{i+1:02d}", 
                "start_position": np.array([0, 4*i, -0.5]), 
                "max_velocity": 10.0
            })
        
        self.uav_velocity = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.uav_number)]
        self.uav_yaw_speed = [0.0 for _ in range(self.uav_number)]
        self.AIRSIM_LOCK = threading.Lock()
        self.state = "Ready"
        
        # 批处理缓冲区
        self.batch_buffer = [None] * self.uav_number
        self.last_send_time = time.time()
        
        # 启动系统
        self.AirSim_start()
        
        # 启动线程
        threads = [
            threading.Thread(target=self.data_fetch, daemon=True),
            threading.Thread(target=self.communication_sendcenter, name="AirSimDataManager_send", daemon=True),
            threading.Thread(target=self.communication_recvcenter, name="AirSimDataManager_recv", daemon=True),
            threading.Thread(target=self.communication_taskcenter, name="AirSimDataManager_task", daemon=True),
            threading.Thread(target=self.RL_movebyVelocity, name="AirSimDataManager_Velocity_Control", daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        self.state = "Running"
        self._monitor_system()
    
    def _load_spawn_points(self):
        """加载重生点坐标（只执行一次）"""
        points_matrix = target_points_read()
        
        if points_matrix is None or len(points_matrix) == 0:
            # 生成默认的重生点网格
            n_per_side = int(math.ceil(math.sqrt(self.uav_number * 2)))
            points = []
            for i in range(n_per_side):
                for j in range(n_per_side):
                    points.append([i * 4.0, j * 4.0, -0.5])
            points_matrix = np.array(points)
            print(f"使用默认重生点，共{len(points_matrix)}个点")
        
        # 确保有足够的重生点
        if len(points_matrix) < self.uav_number:
            print(f"警告: 重生点数量({len(points_matrix)})少于无人机数量({self.uav_number})，将复制部分点")
            # 复制点直到足够
            while len(points_matrix) < self.uav_number:
                points_matrix = np.vstack([points_matrix, points_matrix[:self.uav_number - len(points_matrix)]])
        
        return points_matrix
    
    def _get_spawn_point(self, uav_id):
        """为无人机获取一个未使用的重生点"""
        # 将numpy数组转换为元组列表以便比较
        all_points_tuples = [tuple(point) for point in self.global_points]
        
        # 过滤掉已使用的点
        available_points = [point for point in all_points_tuples 
                           if point not in self.used_spawn_points]
        
        if not available_points:
            # 如果没有可用点，清空已使用列表
            print("所有重生点已使用，清空使用记录")
            self.used_spawn_points.clear()
            available_points = all_points_tuples
        
        # 选择距离原点最近的点
        distances = [np.linalg.norm(np.array(point) - self.origin_pos) 
                    for point in available_points]
        selected_point = available_points[np.argmin(distances)]
        
        # 转换为相对坐标
        relative_point = np.array(selected_point) - self.uav_settings[uav_id]["start_position"]
        
        # 记录已使用的点（使用原始坐标）
        self.used_spawn_points.append(selected_point)
        
        print(f"UAV{uav_id+1:02d} 重生点: {relative_point} (原始: {selected_point})")
        return relative_point
    
    def AirSim_start(self):
        """批量启动所有无人机"""
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        
        # 批量启用API控制
        for name in vehicle_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            self.client.takeoffAsync(vehicle_name=name)
        
        time.sleep(3)
        print(f"AirSim: {self.uav_number}架无人机已准备就绪")
        print(f"使用重生点数量: {len(self.used_spawn_points)}/{len(self.global_points)}")
    
    def _monitor_system(self):
        """系统监控"""
        while True:
            print(f"状态:{self.state}, 数据队列:{[len(q) for q in self.Data_Queue]}, 任务队列:{len(self.task_deque)}, 已使用重生点:{len(self.used_spawn_points)}")
            time.sleep(0.5)
    
    def _batch_get_kinematics(self, vehicle_names):
        """批量获取无人机运动学信息"""
        kinematics_data = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    value = self.client.simGetGroundTruthKinematics(vehicle_name=name)
                pos = value.position.to_numpy_array().astype(np.float32)
                q = np.array([value.orientation.w_val, value.orientation.x_val, 
                             value.orientation.y_val, value.orientation.z_val], dtype=np.float32)
                pitch, roll, yaw = airsim.to_eularian_angles(value.orientation)
                eular_angles = np.array([roll, pitch, yaw], dtype=np.float32)
                velocity = value.linear_velocity.to_numpy_array().astype(np.float32)
                
                kinematics_data.append((pos, q, eular_angles, velocity))
            except Exception as e:
                print(f"获取{name}运动学数据失败: {e}")
                kinematics_data.append((np.zeros(3), np.array([1,0,0,0]), np.zeros(3), np.zeros(3)))
        
        return kinematics_data
    
    def _batch_get_lidar(self, vehicle_names):
        """批量获取激光雷达数据"""
        lidar_data = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
                lidar_data.append(data)
            except:
                lidar_data.append(None)
        return lidar_data
    
    def _batch_get_collision(self, vehicle_names):
        """批量获取碰撞信息"""
        collisions = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    collision = self.client.simGetCollisionInfo(vehicle_name=name).has_collided
                collisions.append(collision)
            except:
                collisions.append(False)
        return collisions
    
    def data_fetch(self):
        """批量获取无人机数据"""
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        
        while True:
            if self.state == "Running":
                try:
                    timestamp = time.time()
                    
                    # 批量获取数据
                    kinematics = self._batch_get_kinematics(vehicle_names)
                    lidar_data = self._batch_get_lidar(vehicle_names)
                    collisions = self._batch_get_collision(vehicle_names)
                    
                    # 批量存储数据
                    for i in range(self.uav_number):
                        pos, q, eular, vel = kinematics[i]
                        self.Data_Queue[i].append({
                            "timestamp": timestamp,
                            "position": pos,
                            "orientation": q,
                            "eular_angle": eular,
                            "velocity": vel,
                            "Lidar": lidar_data[i],
                            "collision": collisions[i]
                        })
                    
                    # 更新批处理缓冲区
                    for i in range(self.uav_number):
                        if self.Data_Queue[i]:
                            self.batch_buffer[i] = self.Data_Queue[i][-1]
                    
                except Exception as e:
                    print(f"批量数据获取异常: {e}")
            
            time.sleep(0.02)  # 50Hz
    
    def RL_movebyVelocity(self):
        """批量控制无人机速度"""
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        
        while True:
            if self.state == "Running":
                
                for i, name in enumerate(vehicle_names):
                    vx, vy, vz = float(self.uav_velocity[i][0]), float(self.uav_velocity[i][1]), float(self.uav_velocity[i][2])
                    yaw = float(self.uav_yaw_speed[i])
                    with self.AIRSIM_LOCK:
                        self.client.moveByVelocityAsync(
                            vx, vy, vz, 0.5,
                            vehicle_name=name,
                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
                        )
            
            time.sleep(0.05)  # 40Hz控制频率
    
    def reset(self, vehicle_name):
        """重置无人机到新的重生点"""
        uav_id = int(vehicle_name[3:]) - 1
        print(f"重置 UAV{uav_id+1:02d}...")
        
        # 停止当前运动
        with self.AIRSIM_LOCK:
            self.client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=vehicle_name)
        
        time.sleep(0.1)
        
        # 获取新的重生点
        new_spawn_point = self._get_spawn_point(uav_id)
        
        # 重置速度
        self.uav_velocity[uav_id] = np.array([0, 0, 0], dtype=np.float32)
        self.uav_yaw_speed[uav_id] = 0.0
        
        # 设置位置
        pose = Pose(
            Vector3r(new_spawn_point[0], new_spawn_point[1], new_spawn_point[2]-np.random.randint(-10,10)),
            airsim.to_quaternion(0, 0, np.random.uniform(-np.pi, np.pi))
        )
        
        with self.AIRSIM_LOCK:
            self.client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        time.sleep(0.1)
        with self.AIRSIM_LOCK:
            self.client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        
        time.sleep(0.1)
        
        # 获取重置后的状态
        try:
            kinematics = self._batch_get_kinematics([vehicle_name])[0]
            lidar = self._batch_get_lidar([vehicle_name])[0]
            collision = self._batch_get_collision([vehicle_name])[0]
            
            pos, q, eular, vel = kinematics
            self.Data_Queue[uav_id].clear()  # 清空旧数据
            self.Data_Queue[uav_id].append({
                "timestamp": time.time(),
                "position": pos,
                "orientation": q,
                "eular_angle": eular,
                "velocity": vel,
                "Lidar": lidar,
                "collision": collision
            })
            self.batch_buffer[uav_id] = self.Data_Queue[uav_id][-1]
            
            print(f"UAV{uav_id+1:02d} 重置完成，新位置: {pos}")
            
        except Exception as e:
            print(f"reset后获取状态失败: {e}")
    
    def communication_recvcenter(self):
        """接收控制指令"""
        buffers = {}
        
        while True:
            if self.state == "Running":
                try:
                    packet, _ = self.udp_recv.recvfrom(MAX_PACKET_SIZE + 16)
                    pkt_id, total_pkts = struct.unpack('!II', packet[:8])
                    seq = struct.unpack('!I', packet[8:12])[0]
                    chunk = packet[12:]
                    
                    if pkt_id not in buffers:
                        buffers[pkt_id] = {}
                    buffers[pkt_id][seq] = chunk
                    
                    # 检查是否收齐
                    if len(buffers[pkt_id]) == total_pkts:
                        full_data = b''.join([buffers[pkt_id][i] for i in sorted(buffers[pkt_id])])
                        control, data = pickle.loads(full_data)
                        del buffers[pkt_id]
                        self.task_deque.append([control, data])
                        
                except Exception as e:
                    print(f"接收异常: {e}")
            
            time.sleep(0.001)
    
    def communication_taskcenter(self):
        """处理任务队列"""
        while True:
            if self.state == "Running" and self.task_deque:
                try:
                    control, data = self.task_deque.popleft()
                    
                    if control == "action":
                        uavname, action = data
                        uav_id = int(uavname[3:]) - 1
                        self.uav_velocity[uav_id] = action[0:3].astype(np.float32)
                        self.uav_yaw_speed[uav_id] = float(action[3])
                        
                    elif control == "reset":
                        uavname = data[0]
                        self.reset(uavname)
                        
                except Exception as e:
                    print(f"任务处理异常: {e}")
            
            time.sleep(0.001)
    
    def communication_sendcenter(self):
        """批量发送数据到训练中心"""
        packet_counter = 0
        
        while True:
            if self.state == "Running":
                # 批量准备数据
                send_data = []
                for i in range(self.uav_number):
                    if self.batch_buffer[i]:
                        data = pickle.dumps(self.batch_buffer[i], protocol=pickle.HIGHEST_PROTOCOL)
                        send_data.append((i, data))
                
                # 批量发送
                for uav_id, data in send_data:
                    try:
                        total_size = len(data)
                        num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                        header = struct.pack('!II', packet_counter % 1000, num_packets)
                        
                        for seq in range(num_packets):
                            start = seq * MAX_PACKET_SIZE
                            end = start + MAX_PACKET_SIZE
                            chunk = data[start:end]
                            packet = header + struct.pack('!I', seq) + chunk
                            
                            self.udp_sender.sendto(packet, (self.trainmanager_ip, 2000 + uav_id))
                        
                    except Exception as e:
                        print(f"发送UAV{uav_id:02d}数据失败: {e}")
                
                packet_counter += 1
                if packet_counter >= 10000:
                    packet_counter = 0
            
            time.sleep(0.02)  # 50Hz发送频率


if __name__ == "__main__":
    ADM = AirSim_Data_Manager(airsim_ip='172.21.224.1', uav_number=8)