import numpy as np
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

class APFOptimized:
    def __init__(self, stat_range=50.0):
        self.stat_range = stat_range
        self.pos = np.array([0.0, 0.0, 0.0])  # NED position
        self.windowstat = (np.empty((0, 3)), np.empty((0,)))  # (points, probabilities)
    
    def APF_SeqCalculate3D(self, position, yaw, max_directions=24, min_angle_deg=30.0):
        """
        优化版本：使用方向网格化降低计算复杂度
        返回: (max_directions, 5) 的矩阵，每行: [normed_distance, cos(alpha), sin(alpha), cos(beta), sin(beta)]
        """
        points_xyz, probs = self.windowstat
        
        # 空点云检查
        if len(points_xyz) == 0:
            return self._get_empty_state(max_directions)
        
        # Step 1: 筛选有效点
        valid_mask = probs > 0.5
        if not np.any(valid_mask):
            return self._get_empty_state(max_directions)
        
        valid_points = points_xyz[valid_mask]
        K = len(valid_points)
        agent_pos = self.pos + position
        
        # Step 2: 快速下采样（如果点太多）
        if K > 2000:
            indices = np.random.choice(K, 1000, replace=False)
            valid_points = valid_points[indices]
            K = 1000
        
        # Step 3: 转换到体坐标系
        rel_ned = valid_points - agent_pos
        
        # NED到体坐标的旋转矩阵 (yaw: 0=北, 顺时针为正)
        c, s = np.cos(yaw), np.sin(yaw)
        R_ned_to_body = np.array([
            [c, s, 0],    # x_body = cos(yaw)*北 + sin(yaw)*东
            [-s, c, 0],   # y_body = -sin(yaw)*北 + cos(yaw)*东
            [0, 0, 1]     # z_body = down (不变)
        ], dtype=np.float32)
        
        rel_body = rel_ned @ R_ned_to_body.T
        
        # Step 4: 计算距离和方向角
        distances = np.linalg.norm(rel_body, axis=1)
        
        # 计算方位角和仰角
        x, y, z = rel_body[:, 0], rel_body[:, 1], rel_body[:, 2]
        horiz_norm = np.sqrt(x**2 + y**2) + 1e-8
        
        # 方位角: 机头方向为0，右为正，左为负
        alpha_rad = np.arctan2(y, x)
        
        # 仰角: 水平为0，向上(+z_body)为正，向下(-z_body)为负
        # 注意: NED中z是down，体坐标z也是down，所以向上是-z
        beta_rad = np.arctan2(-z, horiz_norm)
        
        # Step 5: 方向网格化选择（替代成对比较）
        state_matrix = self._select_by_direction_grid(
            distances, alpha_rad, beta_rad, rel_body,
            max_directions, min_angle_deg
        )
        
        return state_matrix
    
    def _select_by_direction_grid(self, distances, alpha_rad, beta_rad, rel_body,
                                 max_directions, min_angle_deg):
        """
        使用方向网格化选择障碍物，复杂度 O(K)
        """
        # 创建方向网格
        grid_res = int(np.ceil(180 / min_angle_deg))  # 网格分辨率
        azimuth_grid = np.linspace(-np.pi, np.pi, grid_res * 2 + 1)[:-1]
        elevation_grid = np.linspace(-np.pi/2, np.pi/2, grid_res + 1)
        
        # 初始化网格存储
        grid_dist = np.full((len(azimuth_grid), len(elevation_grid)), np.inf)
        grid_indices = np.full((len(azimuth_grid), len(elevation_grid)), -1)
        
        # 将每个点分配到最近的网格
        for i in range(len(distances)):
            # 找到最近的方位角和仰角网格索引
            az_idx = np.argmin(np.abs(alpha_rad[i] - azimuth_grid))
            el_idx = np.argmin(np.abs(beta_rad[i] - elevation_grid))
            
            # 如果这个网格还没有点，或者这个点更近，则更新
            if distances[i] < grid_dist[az_idx, el_idx]:
                grid_dist[az_idx, el_idx] = distances[i]
                grid_indices[az_idx, el_idx] = i
        
        # 收集网格中的点
        selected_indices = []
        for az_idx in range(len(azimuth_grid)):
            for el_idx in range(len(elevation_grid)):
                idx = grid_indices[az_idx, el_idx]
                if idx != -1:
                    selected_indices.append(idx)
        
        # 按距离排序并选择最多max_directions个
        if selected_indices:
            selected_indices = np.array(selected_indices)
            selected_distances = distances[selected_indices]
            sorted_idx = np.argsort(selected_distances)[:max_directions]
            selected_indices = selected_indices[sorted_idx]
        
        # 构建输出
        state_matrix = np.zeros((max_directions, 5), dtype=np.float32)
        
        for i, idx in enumerate(selected_indices):
            if i >= max_directions:
                break
            
            # 非线性距离归一化（近处更敏感）
            d_norm = 1.0 - np.exp(-distances[idx] / self.stat_range)
            
            a = alpha_rad[idx]
            b = beta_rad[idx]
            state_matrix[i] = [
                d_norm,
                np.cos(a), np.sin(a),
                np.cos(b), np.sin(b)
            ]
        
        # 填充剩余位置
        for i in range(len(selected_indices), max_directions):
            state_matrix[i] = [1.0, 1.0, 0.0, 1.0, 0.0]
        
        return state_matrix
    
    def _get_empty_state(self, max_directions):
        """返回空状态"""
        return np.tile(np.array([1.0, 1.0, 0.0, 1.0, 0.0]), 
                      (max_directions, 1)).astype(np.float32)
    
    def set_obstacles(self, points_xyz, probs):
        """设置障碍物点云"""
        self.windowstat = (points_xyz, probs)

def test_APF_optimized():
    """测试优化后的APF函数"""
    print("=== 测试 APF_SeqCalculate3D ===")
    
    # 创建APF实例
    apf = APFOptimized(stat_range=50.0)
    
    # 测试1: 空点云
    print("\n1. 测试空点云:")
    empty_points = np.empty((0, 3))
    empty_probs = np.empty((0,))
    apf.set_obstacles(empty_points, empty_probs)
    result = apf.APF_SeqCalculate3D(position=[0,0,0], yaw=0)
    print(f"结果形状: {result.shape}")
    print(f"第一行(应全为[1,1,0,1,0]): {result[0]}")
    
    # 测试2: 单一障碍物（正前方10米）
    print("\n2. 测试正前方障碍物:")
    points = np.array([[10.0, 0.0, 0.0]])  # 北10米（正前方）
    probs = np.array([0.8])
    apf.set_obstacles(points, probs)
    result = apf.APF_SeqCalculate3D(position=[0,0,0], yaw=0)
    print(f"最近障碍物距离: {result[0,0]:.3f}")
    print(f"方位角cos/sin: [{result[0,1]:.3f}, {result[0,2]:.3f}]")
    print(f"仰角cos/sin: [{result[0,3]:.3f}, {result[0,4]:.3f}]")
    
    # 测试3: 多个障碍物
    print("\n3. 测试多个障碍物:")
    np.random.seed(42)
    n_points = 500
    # 生成随机障碍物（主要在前方扇形区域）
    angles = np.random.uniform(-60, 60, n_points)  # 方位角范围
    distances = np.random.uniform(5, 40, n_points)  # 距离范围
    elevations = np.random.uniform(-30, 30, n_points)  # 仰角范围
    
    # 转换为NED坐标
    alpha_rad = np.radians(angles)
    beta_rad = np.radians(elevations)
    x = distances * np.cos(beta_rad) * np.cos(alpha_rad)  # 北
    y = distances * np.cos(beta_rad) * np.sin(alpha_rad)  # 东
    z = -distances * np.sin(beta_rad)  # down（注意负号）
    
    points = np.column_stack([x, y, z])
    probs = np.random.uniform(0.6, 1.0, n_points)
    
    apf.set_obstacles(points, probs)
    
    # 测试不同航向角
    for yaw_deg in [0, 90, 180]:
        print(f"\n  航向角 {yaw_deg}°:")
        result = apf.APF_SeqCalculate3D(
            position=[0,0,0], 
            yaw=np.radians(yaw_deg)
        )
        
        # 统计有效障碍物数量
        valid_mask = result[:, 0] < 1.0
        n_valid = np.sum(valid_mask)
        print(f"  有效障碍物数量: {n_valid}")
        
        if n_valid > 0:
            avg_dist = np.mean(result[valid_mask, 0])
            print(f"  平均归一化距离: {avg_dist:.3f}")
    
    # 测试4: 性能测试
    print("\n4. 性能测试（10000个点）:")
    n_points = 10000
    points = np.random.uniform(-50, 50, (n_points, 3))
    probs = np.random.uniform(0.6, 1.0, n_points)
    apf.set_obstacles(points, probs)
    
    import time
    start = time.time()
    for _ in range(100):
        result = apf.APF_SeqCalculate3D([0,0,0], 0)
    elapsed = time.time() - start
    print(f"  100次调用平均时间: {elapsed*10:.1f}ms, {result}")
    
    print("\n=== 测试完成 ===")

# 运行测试
if __name__ == "__main__":
    test_APF_optimized()