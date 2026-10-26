"""
AirSim_Data_Manager — hardened reset logic.

Key fixes over the previous version:
  1. start_ned[2] is now actually used (was silently replaced with a random z).
  2. Two-stage "safe park" pose is cleared via moveByVelocity + hover, ensuring
     the physics engine does not drag the UAV away from the target pose.
  3. After simSetVehiclePose, we read back the pose and retry up to N times
     until the reported position matches the commanded position within tol.
  4. data_fetch is paused per-UAV during reset so no stale observation is
     pushed to the training side.
  5. The final verified pose is stored as uav_reset_pose[uav_id] so the
     training side can consult it for alignment.
"""

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
from typing import List

MAX_PACKET_SIZE = 62000

os.chdir(os.path.dirname(os.path.abspath(__file__)))
client = airsim.MultirotorClient()


def load_json_to_nx3_matrix(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        matrix = np.array(data, dtype=np.float64)
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


def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'


# ─── Reset tuning constants ──────────────────────────────────────────────────
_RESET_POS_TOL          = 0.30     # m, accepted position error after reset
_RESET_MAX_RETRIES      = 4        # how many simSetVehiclePose attempts
_RESET_SETTLE_TIME      = 0.08     # s, pause between pose-set and readback
_RESET_PARK_Z           = -60.0    # NED z for parking pose (well above scene)
_RESET_STALL_TIME       = 0.15     # s, hold zero velocity before teleport
_RESET_POST_SLEEP       = 0.20     # s, total blackout after final pose set


class AirSim_Data_Manager:
    def __init__(self, server_port=5812, airsim_ip='', airsim_port=41451,
                 trainmanager_ip='', uav_number=6):
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

        self.Data_Queue = [deque(maxlen=50) for _ in range(self.uav_number)]
        self.task_deque = deque(maxlen=50)

        self.uav_settings = []
        for i in range(self.uav_number):
            self.uav_settings.append({
                "uavname": f"UAV{i+1:02d}",
                "start_position": np.array([0, 0, -0.5]),
                "max_velocity": 10.0
            })

        self.uav_velocity     = [np.array([0, 0, 0], dtype=np.float32)
                                 for _ in range(self.uav_number)]
        self.uav_yaw_speed    = [0.0 for _ in range(self.uav_number)]

        # Per-UAV reset coordination — when True, data_fetch and
        # RL_movebyVelocity skip that UAV until reset is confirmed complete.
        self.uav_resetting    = [False] * self.uav_number
        # Final verified pose after the most recent reset
        self.uav_reset_pose   = [None]  * self.uav_number

        self.AIRSIM_LOCK = threading.Lock()
        self.state = "Ready"

        self.batch_buffer  = [None] * self.uav_number
        self.last_send_time = time.time()

        self.AirSim_start()

        threads = [
            threading.Thread(target=self.data_fetch,              name="ADM_fetch",    daemon=True),
            threading.Thread(target=self.communication_sendcenter, name="ADM_send",     daemon=True),
            threading.Thread(target=self.communication_recvcenter, name="ADM_recv",     daemon=True),
            threading.Thread(target=self.communication_taskcenter, name="ADM_task",     daemon=True),
            threading.Thread(target=self.RL_movebyVelocity,        name="ADM_velocity", daemon=True),
        ]
        for t in threads:
            t.start()

        self.state = "Running"
        self._monitor_system()

    # ─────────────────────────────────────────────────────────────────────
    #  Startup
    # ─────────────────────────────────────────────────────────────────────
    def AirSim_start(self):
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        for name in vehicle_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True,        vehicle_name=name)
            self.client.takeoffAsync(          vehicle_name=name)
        time.sleep(3)
        print(f"AirSim: {self.uav_number}架无人机已准备就绪")

    def _monitor_system(self):
        while True:
            print(
                f"状态:{self.state}, "
                f"数据队列:{[len(q) for q in self.Data_Queue]}, "
                f"任务队列:{len(self.task_deque)}, "
                f"resetting:{self.uav_resetting}"
            )
            time.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────
    #  Kinematics / lidar / collision batch reads
    # ─────────────────────────────────────────────────────────────────────
    def _batch_get_kinematics(self, vehicle_names):
        kinematics_data = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    value = self.client.simGetGroundTruthKinematics(vehicle_name=name)
                pos = value.position.to_numpy_array().astype(np.float32)
                q   = np.array([value.orientation.w_val, value.orientation.x_val,
                                value.orientation.y_val, value.orientation.z_val],
                               dtype=np.float32)
                pitch, roll, yaw = airsim.to_eularian_angles(value.orientation)
                eular_angles = np.array([roll, pitch, yaw], dtype=np.float32)
                velocity = value.linear_velocity.to_numpy_array().astype(np.float32)
                kinematics_data.append((pos, q, eular_angles, velocity))
            except Exception as e:
                print(f"获取{name}运动学数据失败: {e}")
                kinematics_data.append((np.zeros(3), np.array([1, 0, 0, 0]),
                                        np.zeros(3), np.zeros(3)))
        return kinematics_data

    def _batch_get_lidar(self, vehicle_names):
        lidar_data = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
                lidar_data.append(data)
            except Exception:
                lidar_data.append(None)
        return lidar_data

    def _batch_get_collision(self, vehicle_names):
        collisions = []
        for name in vehicle_names:
            try:
                with self.AIRSIM_LOCK:
                    collision = self.client.simGetCollisionInfo(
                        vehicle_name=name).has_collided
                collisions.append(collision)
            except Exception:
                collisions.append(False)
        return collisions

    # ─────────────────────────────────────────────────────────────────────
    #  Data fetch loop
    # ─────────────────────────────────────────────────────────────────────
    def data_fetch(self):
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        while True:
            if self.state != "Running":
                time.sleep(0.01)
                continue
            try:
                timestamp  = time.time()
                kinematics = self._batch_get_kinematics(vehicle_names)
                lidar_data = self._batch_get_lidar(vehicle_names)
                collisions = self._batch_get_collision(vehicle_names)

                for i in range(self.uav_number):
                    if self.uav_resetting[i]:
                        # Skip this UAV entirely while its reset is in flight
                        continue
                    pos, q, eular, vel = kinematics[i]
                    self.Data_Queue[i].append({
                        "timestamp":   timestamp,
                        "position":    pos,
                        "orientation": q,
                        "eular_angle": eular,
                        "velocity":    vel,
                        "Lidar":       lidar_data[i],
                        "collision":   collisions[i],
                    })
                    if self.Data_Queue[i]:
                        self.batch_buffer[i] = self.Data_Queue[i][-1]
                    time.sleep(0.005)

            except Exception as e:
                print(f"批量数据获取异常: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  Velocity control loop
    # ─────────────────────────────────────────────────────────────────────
    def RL_movebyVelocity(self):
        vehicle_names = [f"UAV{i+1:02d}" for i in range(self.uav_number)]
        while True:
            if self.state != "Running":
                time.sleep(0.01)
                continue
            for i, name in enumerate(vehicle_names):
                if self.uav_resetting[i]:
                    # Don't fight the teleport
                    time.sleep(0.005)
                    continue
                vx  = float(self.uav_velocity[i][0])
                vy  = float(self.uav_velocity[i][1])
                vz  = float(self.uav_velocity[i][2])
                yaw = float(self.uav_yaw_speed[i])
                with self.AIRSIM_LOCK:
                    self.client.moveByVelocityAsync(
                        vx, vy, vz, 0.5,
                        vehicle_name=name,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
                    )
                time.sleep(0.005)

    # ─────────────────────────────────────────────────────────────────────
    #  Reset — robust teleport to an arbitrary NED pose
    # ─────────────────────────────────────────────────────────────────────
    def _set_pose_verified(self, vehicle_name, target_pose, target_xyz,
                           tol=_RESET_POS_TOL, max_retries=_RESET_MAX_RETRIES):
        """
        Issue simSetVehiclePose up to max_retries times, reading back the
        actual pose in between and retrying if the delta exceeds tol.

        Returns (success: bool, final_xyz: np.ndarray).
        """
        for attempt in range(max_retries):
            with self.AIRSIM_LOCK:
                self.client.simSetVehiclePose(target_pose, True,
                                              vehicle_name=vehicle_name)
            time.sleep(_RESET_SETTLE_TIME)
            with self.AIRSIM_LOCK:
                actual = self.client.simGetVehiclePose(
                    vehicle_name=vehicle_name).position.to_numpy_array()
            err = float(np.linalg.norm(actual - target_xyz))
            if err < tol:
                return True, actual
            print(f"  [reset] {vehicle_name} attempt {attempt+1}/{max_retries}: "
                  f"err={err:.3f} m, retry")
        return False, actual

    def reset(self, vehicle_name, start_ned):
        """
        Teleport the named UAV to the NED coordinates in start_ned.

        Sequence:
          1. Mark UAV as resetting → data_fetch/velocity-control skip it.
          2. Clear command velocities and issue a hover to zero physical velocity.
          3. Teleport to a high-altitude 'park' pose to break any simulator
             momentum / collision state; verify.
          4. Teleport to target pose; verify with up to N retries.
          5. Hover at zero velocity, flush stale observations, then unblock.
        """
        uav_id = int(vehicle_name[3:]) - 1
        start_ned = np.asarray(start_ned, dtype=np.float64)
        target_xyz = start_ned.copy()

        print(f"[reset] {vehicle_name} → target={np.round(target_xyz, 2)}")

        # 1. Block other loops from touching this UAV
        self.uav_resetting[uav_id] = True

        try:
            # 2. Clear commanded velocity and zero physical velocity
            self.uav_velocity[uav_id]  = np.array([0, 0, 0], dtype=np.float32)
            self.uav_yaw_speed[uav_id] = 0.0
            with self.AIRSIM_LOCK:
                # Zero-velocity command absorbed over a short window
                self.client.moveByVelocityAsync(0, 0, 0, _RESET_STALL_TIME,
                                                vehicle_name=vehicle_name)
            time.sleep(_RESET_STALL_TIME)

            # 3. Park pose (high altitude, origin XY) — breaks collision /
            #    rotor-wash state cleanly.
            park_pose = Pose(
                Vector3r(0.0, 0.0, _RESET_PARK_Z),
                airsim.to_quaternion(0, 0, 0)
            )
            ok_park, _ = self._set_pose_verified(
                vehicle_name, park_pose,
                np.array([0.0, 0.0, _RESET_PARK_Z]),
                tol=1.0, max_retries=2  # park is loose on tolerance
            )
            if not ok_park:
                print(f"  [reset] {vehicle_name} WARNING: park teleport "
                      f"did not verify")

            # Clear physical velocity accumulated during park teleport
            with self.AIRSIM_LOCK:
                self.client.moveByVelocityAsync(0, 0, 0, 0.05,
                                                vehicle_name=vehicle_name)
            time.sleep(0.05)

            # 4. Final pose — USE THE SUPPLIED Z (fix for the earlier bug).
            target_pose = Pose(
                Vector3r(float(target_xyz[0]),
                         float(target_xyz[1]),
                         float(target_xyz[2])),
                airsim.to_quaternion(0, 0, 0)
            )
            ok_final, final_xyz = self._set_pose_verified(
                vehicle_name, target_pose, target_xyz,
                tol=_RESET_POS_TOL, max_retries=_RESET_MAX_RETRIES
            )
            if not ok_final:
                print(f"  [reset] {vehicle_name} FAILED to land within "
                      f"{_RESET_POS_TOL} m of target. Final pos = "
                      f"{np.round(final_xyz, 3)}")
            else:
                print(f"  [reset] {vehicle_name} verified pos = "
                      f"{np.round(final_xyz, 3)}")

            # 5. Hover-and-hold to keep it where we put it
            with self.AIRSIM_LOCK:
                self.client.moveByVelocityAsync(0, 0, 0, 0.5,
                                                vehicle_name=vehicle_name)
            time.sleep(_RESET_POST_SLEEP)

            # 6. Flush stale observations — the first observation the training
            #    side pops MUST reflect the new pose.
            self.Data_Queue[uav_id].clear()
            self.batch_buffer[uav_id] = None
            self.uav_reset_pose[uav_id] = final_xyz.copy()

        except Exception as e:
            print(f"[reset] {vehicle_name} exception: {e}")

        finally:
            self.uav_resetting[uav_id] = False

    # ─────────────────────────────────────────────────────────────────────
    #  Communication
    # ─────────────────────────────────────────────────────────────────────
    def communication_recvcenter(self):
        buffers = {}
        while True:
            if self.state != "Running":
                time.sleep(0.01)
                continue
            try:
                packet, _ = self.udp_recv.recvfrom(MAX_PACKET_SIZE + 16)
                pkt_id, total_pkts = struct.unpack('!II', packet[:8])
                seq   = struct.unpack('!I', packet[8:12])[0]
                chunk = packet[12:]
                if pkt_id not in buffers:
                    buffers[pkt_id] = {}
                buffers[pkt_id][seq] = chunk
                if len(buffers[pkt_id]) == total_pkts:
                    full_data = b''.join([buffers[pkt_id][i]
                                          for i in sorted(buffers[pkt_id])])
                    control, data = pickle.loads(full_data)
                    del buffers[pkt_id]
                    self.task_deque.append([control, data])
            except Exception as e:
                print(f"接收异常: {e}")

    def communication_taskcenter(self):
        while True:
            if self.state == "Running" and self.task_deque:
                while len(self.task_deque) > 0:
                    try:
                        control, data = self.task_deque.popleft()
                        if control == "action":
                            uavname, action = data
                            uav_id = int(uavname[3:]) - 1
                            if self.uav_resetting[uav_id]:
                                # Drop commands during reset
                                continue
                            self.uav_velocity[uav_id]  = action[0:3].astype(np.float32)
                            self.uav_yaw_speed[uav_id] = float(action[3])
                        elif control == "reset":
                            # data is expected to be [uavname, start_ned_list]
                            if isinstance(data, (list, tuple)) and len(data) >= 2:
                                uavname  = data[0]
                                start_ned = np.asarray(data[1], dtype=np.float64)
                            else:
                                # Back-compat: old payload [uavname]
                                uavname = data[0] if isinstance(data, (list, tuple)) else data
                                start_ned = np.array([0.0, 0.0, -20.0], dtype=np.float64)
                                print(f"[reset] {uavname}: no start_ned "
                                      f"supplied, using default")
                            self.reset(uavname, start_ned)
                    except Exception as e:
                        print(f"任务处理异常: {e}")
                    time.sleep(0.003)
            time.sleep(0.01)

    def communication_sendcenter(self):
        packet_counter = 0
        while True:
            if self.state != "Running":
                time.sleep(0.01)
                continue
            send_data = []
            for i in range(self.uav_number):
                if self.uav_resetting[i]:
                    continue
                if self.batch_buffer[i]:
                    data = pickle.dumps(self.batch_buffer[i],
                                        protocol=pickle.HIGHEST_PROTOCOL)
                    send_data.append((i, data))

            for uav_id, data in send_data:
                try:
                    total_size  = len(data)
                    num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                    header      = struct.pack('!II', packet_counter % 1000, num_packets)
                    for seq in range(num_packets):
                        start = seq * MAX_PACKET_SIZE
                        end   = start + MAX_PACKET_SIZE
                        chunk = data[start:end]
                        packet = header + struct.pack('!I', seq) + chunk
                        self.udp_sender.sendto(packet,
                                               (self.trainmanager_ip, 2000 + uav_id))
                except Exception as e:
                    print(f"发送UAV{uav_id:02d}数据失败: {e}")
                time.sleep(0.005)
            packet_counter += 1
            if packet_counter >= 10000:
                packet_counter = 0


if __name__ == "__main__":
    ADM = AirSim_Data_Manager(airsim_ip='172.21.224.1', uav_number=1)