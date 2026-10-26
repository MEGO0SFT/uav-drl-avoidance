import numpy as np
import airsim
import time
from numba import njit, prange

# 最后更新时间 2025年11月19日
# 作者： 石辅天
# 最新功能：激光雷达模块诞生
# 默认参数来源蜂鸟Genius
# 通道: 16/32
# 扫描视角: 360x30
# 频率: 5-20Hz
# 点数: 320000 pts/s
# 测距范围: 200m(反射率20%)
__all__ = ['LidarModule', 'quat_to_rot_matrix', 'Rb2g_matrix']

@njit(fastmath=True)
def quat_to_rot_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*x*x - 2*z*z, 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*x*x - 2*y*y]
    ])
    return R

@njit(fastmath=True)
def Rb2g_matrix(q, v_body):
    """使用旋转矩阵批量计算 v_body 的变换"""
    R = quat_to_rot_matrix(q)
    
    # 确保 v_body 是 (N, 3) 或 (3,)
    if v_body.ndim == 1:
        return R @ v_body  # 返回 (3,)
    else:
        return (R @ v_body.T).T  # 返回 (N, 3)

class LidarModule:
    def __init__(self, NumberOfChannels=16, PointsPerSecond=320000, RotationFrequency=20, UpperFOV=15, LowerFOV=-15, HorizontalFOVStart=0, HorizontalFOVEnd=360, MaxRange=200, GlobalStartPoint=np.array([0.0, 0.0, -0.5], dtype=np.float64)):
        """
        激光雷达模块\n
        适用平台: AirSim\n
        参数: 默认适用蜂鸟Genius的参数
        """
        self.GSP = GlobalStartPoint 
        self.noc = NumberOfChannels
        self.pts = PointsPerSecond
        self.rf = RotationFrequency
        self.uf = UpperFOV
        self.lf = LowerFOV
        self.hfs = HorizontalFOVStart
        self.hfe = HorizontalFOVEnd
        self.mr = MaxRange
        self.point_cloud = np.empty((0, 3), dtype=np.float64)

    def LidarDataResolver(self, lidar_data):
        """解析激光雷达数据"""
        points_clouds = np.array(lidar_data.point_cloud, dtype=np.float64).reshape(-1, 3)
        quaternion = np.array([lidar_data.pose.orientation.w_val, lidar_data.pose.orientation.x_val, lidar_data.pose.orientation.y_val, lidar_data.pose.orientation.z_val], dtype=np.float64)
        position = np.array([lidar_data.pose.position.x_val, lidar_data.pose.position.y_val, lidar_data.pose.position.z_val], dtype=np.float64)
        timestamp = lidar_data.time_stamp/1e9
        original_lidardata = {
            "points_clouds": points_clouds,
            "quaternion": quaternion,
            "position": position,
            "timestamp": timestamp
        }
        return original_lidardata
    
    def LidarDataB2G(self, original_lidardata, voxel_available=False, voxel_size=0.5):
        """将激光雷达数据从机体坐标系转换到全局坐标系(NED)"""
        points_clouds = original_lidardata["points_clouds"]
        quaternion = original_lidardata["quaternion"]
        position = original_lidardata["position"]
        points_global = Rb2g_matrix(quaternion, points_clouds) + position + self.GSP
        if voxel_available:
            points_global = get_unique_voxels_fast(points_global, voxel_size)
        lidar_globaldata = {
            "points_global": points_global,
            "timestamp": original_lidardata["timestamp"]
        }
        return lidar_globaldata

from numba import jit

@jit(nopython=True)
def compute_voxel_key(x, y, z, voxel_size):
    """
    计算点所属的体素键值（使用整数运算提高效率）
    """
    x_idx = int(np.floor(x / voxel_size))
    y_idx = int(np.floor(y / voxel_size))
    z_idx = int(np.floor(z / voxel_size))
    # 将三维坐标转换为一维键值，使用质数减少冲突
    key = x_idx * 73856093 + y_idx * 19349663 + z_idx * 83492791
    return key

@jit(nopython=True)
def voxel_filter_numba_optimized(points, voxel_size=0.5):
    """
    基于体素网格的点云滤波函数（优化版）
    
    参数:
    points: numpy数组 (n, 3)，表示激光雷达点云数据
    voxel_size: float，默认0.5，体素的边长（米）
    
    返回:
    filtered_points: numpy数组，滤波后的点云数据
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    # 预估哈希表大小（避免频繁扩容）
    table_size = max(int(n * 0.3), 10000)
    keys = np.full(table_size, -1, dtype=np.int64)
    indices = np.full(table_size, -1, dtype=np.int64)
    
    # 线性探测解决哈希冲突
    def find_slot(key):
        idx = key % table_size
        while keys[idx] != -1 and keys[idx] != key:
            idx = (idx + 1) % table_size
        return idx
    
    result_indices = []
    
    for i in range(n):
        p = points[i]
        key = compute_voxel_key(p[0], p[1], p[2], voxel_size)
        slot = find_slot(key)
        
        if keys[slot] == -1:
            # 新体素，插入数据
            keys[slot] = key
            indices[slot] = i
            result_indices.append(i)
        # 如果体素已存在，则跳过（保留第一个点）
    
    # 构建结果数组
    result_count = len(result_indices)
    result = np.zeros((result_count, 3), dtype=np.float64)
    for i in range(result_count):
        orig_idx = result_indices[i]
        result[i] = points[orig_idx]
    
    return result

@jit(nopython=True)
def get_unique_voxels_fast(points, voxel_size=0.5):
    """
    快速获取唯一体素点的另一种实现
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    # 计算所有点的体素索引
    voxel_coords = np.empty((n, 3), dtype=np.int32)
    for i in range(n):
        voxel_coords[i, 0] = int(np.floor(points[i, 0] / voxel_size))
        voxel_coords[i, 1] = int(np.floor(points[i, 1] / voxel_size))
        voxel_coords[i, 2] = int(np.floor(points[i, 2] / voxel_size))
    
    # 使用线性化索引进行排序去重
    linear_indices = (
        voxel_coords[:, 0].astype(np.int64) * 1000000000 +
        voxel_coords[:, 1].astype(np.int64) * 1000000 +
        voxel_coords[:, 2].astype(np.int64)
    )
    
    # 排序并去重
    sorted_indices = np.argsort(linear_indices)
    unique_mask = np.ones(len(sorted_indices), dtype=np.bool_)
    
    for i in range(1, len(sorted_indices)):
        if linear_indices[sorted_indices[i]] == linear_indices[sorted_indices[i-1]]:
            unique_mask[i] = False
    
    # 获取唯一点的索引
    unique_point_indices = sorted_indices[unique_mask]
    
    # 构建结果
    result = np.zeros((len(unique_point_indices), 3), dtype=np.float64)
    for i, idx in enumerate(unique_point_indices):
        result[i] = points[idx]
    
    return result

def voxel_filter(points, voxel_size=0.25):
    """
    激光雷达点云体素滤波主函数
    
    参数:
    points: numpy数组 (n, 3)，表示激光雷达点云数据
    voxel_size: float，默认0.25，体素的边长（米）
    
    返回:
    filtered_points: numpy数组，滤波后的点云数据
    """
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    # 使用优化版本的函数
    return get_unique_voxels_fast(points.astype(np.float64), voxel_size)

if __name__ == "__main__":
    testLidarModule = LidarModule(GlobalStartPoint=np.array([0.0, 0.0, -0.5], dtype=np.float64))
    client = airsim.MultirotorClient(ip='172.21.224.1', port=41451)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    uav = "UAV01"
    client.takeoffAsync(vehicle_name=uav).join()
    import OctoMap_o3d as om_o3d
    octomap = om_o3d.OctreeMapManager()
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, -115, -10), airsim.to_quaternion(0, 0, 0)), True, vehicle_name=uav)
    points = np.array([[0,-110,-30],
                       [-110,-110,-30],
                       [-110,110,-30],
                       [110,110,-30],
                       [110,-110,-30],
                       [0,-110,-30]],
                      dtype=np.float64)
    p = 0
    while True:
        stat = client.simGetGroundTruthKinematics(vehicle_name=uav).position
        pos = np.array([stat.x_val, stat.y_val, stat.z_val])
        v = (points[p] - pos)/(np.linalg.norm(points[p] - pos))*5
        client.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), 5.5, vehicle_name=uav)
        timea = time.time()
        lidarData = client.getLidarData(lidar_name="Lidar1", vehicle_name=uav)
        timeb = time.time()
        original_lidardata = testLidarModule.LidarDataResolver(lidarData)
        lidar_globaldata = testLidarModule.LidarDataB2G(original_lidardata)
        timec = time.time()
        octomap.update_with_obstacles(lidar_globaldata['points_global'])
        print(f"Processing Time: {timec - timeb:.6f} seconds, Retrieval Time: {timeb - timea:.6f} seconds")
        print(f"Timestamp: {lidar_globaldata['timestamp']}")
        print(f"Point Cloud Shape: {lidar_globaldata['points_global'].shape}")
        print(f"lidar_globaldata Sample Points:\n {lidar_globaldata['points_global'][:5]}")
        if np.linalg.norm(points[p] - pos) < 5:
            p += 1
            if p >= len(points):
                break
        time.sleep(0.15)
    octomap.visualize()  # 可视化点云与八叉树