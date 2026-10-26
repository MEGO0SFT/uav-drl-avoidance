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
import json
from typing import List, Dict, Any

MAX_PACKET_SIZE = 62000
import json
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
client = airsim.MultirotorClient()
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

# 获取本机ip地址
def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'

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
        self.vaild_points = load_json_to_nx3_matrix('respawn_points.json')
        
        # 数据队列和无人机设置
        self.Data_Queue = [deque(maxlen=50) for _ in range(self.uav_number)]
        self.task_deque = deque(maxlen=50)
        
        # 初始化无人机设置，使用获取的重生点
        self.uav_settings = []
        for i in range(self.uav_number):
            self.uav_settings.append({
                "uavname": f"UAV{i+1:02d}", 
                "start_position": np.array([0, 0, -0.5]), 
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
    
    def _monitor_system(self):
        """系统监控"""
        while True:
            print(f"状态:{self.state}, 数据队列:{[len(q) for q in self.Data_Queue]}, 任务队列:{len(self.task_deque)}")
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
                        if self.Data_Queue[i]:
                            self.batch_buffer[i] = self.Data_Queue[i][-1]
                        time.sleep(0.005)

                    
                except Exception as e:
                    print(f"批量数据获取异常: {e}")
            

    
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
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
                        )
                    time.sleep(0.005)
    
    def reset(self, vehicle_name):
        """重置无人机到新的重生点"""
        uav_id = int(vehicle_name[3:]) - 1
        print(f"重置 UAV{uav_id+1:02d}...")
        
        # 停止当前运动
        with self.AIRSIM_LOCK:
            self.client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=vehicle_name)
        # 重置速度
        self.uav_velocity[uav_id] = np.array([0, 0, 0], dtype=np.float32)
        
        self.uav_yaw_speed[uav_id] = 0.0
        # 从有效重生点中随机选择一个
        # 设置位置
        x = np.random.uniform(-18, 18)
        y = np.random.uniform(-18, 18)
        pose = Pose(
            Vector3r(float(x), float(y), float((np.random.random()+1)*-20)),
            airsim.to_quaternion(0, 0, 0)
        )
        pose_zero = Pose(
            Vector3r(0, 0, -11.5),
            airsim.to_quaternion(0, 0, 0)
        )
        with self.AIRSIM_LOCK:
            for i in range(2):
                self.client.simSetVehiclePose(pose_zero, True, vehicle_name=vehicle_name)
        time.sleep(0.02)
        with self.AIRSIM_LOCK:
            self.client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)

        
        
        # 获取重置后的状态
        try:
            self.Data_Queue[uav_id].clear()  # 清空旧数据
            
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
            
    
    def communication_taskcenter(self):
        """处理任务队列"""
        while True:
            if self.state == "Running" and self.task_deque:
                while len(self.task_deque) > 0:
                    try:
                        control, data = self.task_deque.popleft()
                        
                        if control == "action":
                            uavname, action = data
                            action = action
                            uav_id = int(uavname[3:]) - 1
                            self.uav_velocity[uav_id] = action[0:3].astype(np.float32)
                            self.uav_yaw_speed[uav_id] = float(action[3])
                            
                        elif control == "reset":
                            uavname = data[0]
                            self.reset(uavname)
                        
                    except Exception as e:
                        print(f"任务处理异常: {e}")
                    time.sleep(0.003)
            time.sleep(0.01)
    
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
                    time.sleep(0.005)  # 10Hz发送频率
                packet_counter += 1
                if packet_counter >= 10000:
                    packet_counter = 0
            
                


if __name__ == "__main__":
    ADM = AirSim_Data_Manager(airsim_ip='172.21.224.1', uav_number=4)