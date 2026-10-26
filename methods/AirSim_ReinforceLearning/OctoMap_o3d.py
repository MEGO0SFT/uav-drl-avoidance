import open3d as o3d
import numpy as np
from typing import Dict, Tuple, List
import time
from collections import defaultdict
import numba
from numba import types, typed
import threading
from concurrent.futures import ThreadPoolExecutor

from numba import njit, prange, int64, float32, int32, jit, types
from numba.typed import Dict
from numba.types import UniTuple
__all__ = ['OctreeProbabilisticMap', 'OctreeMapManager', 'voxel_downsample_jit', 'batch_compute_voxel_keys']
# 使用Numba加速的体素下采样函数
@njit(parallel=True, fastmath=True)
def voxel_downsample_jit(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    使用Numba加速的体素下采样函数，对大量点云进行高效降维
    """
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    inv_voxel_size = 1.0 / voxel_size

    # Numba typed.Dict: key=(int64,int64,int64), value=(float32[3], int64)
    voxel_dict = Dict.empty(
        key_type=UniTuple(int64, 3),
        value_type=UniTuple(float32[:], int64)
    )

    # 遍历点云
    for i in prange(points.shape[0]):
        vx = int(np.floor(points[i, 0] * inv_voxel_size))
        vy = int(np.floor(points[i, 1] * inv_voxel_size))
        vz = int(np.floor(points[i, 2] * inv_voxel_size))
        voxel_key = (vx, vy, vz)

        if voxel_key in voxel_dict:
            old_sum, old_count = voxel_dict[voxel_key]
            new_sum = old_sum + points[i]
            voxel_dict[voxel_key] = (new_sum, old_count + 1)
        else:
            voxel_dict[voxel_key] = (points[i].copy(), 1)

    # 输出体素中心点
    downsampled = np.zeros((len(voxel_dict), 3), dtype=np.float32)
    j = 0
    for key, (point_sum, count) in voxel_dict.items():
        downsampled[j, :] = point_sum / count
        j += 1

    return downsampled

@njit(parallel=True, fastmath=True)
def batch_compute_voxel_keys(points: np.ndarray, inv_voxel_size: float):
    """
    并行计算多个点的体素键，返回两个数组 (keys, counts)
    """
    n = points.shape[0]
    keys = np.zeros((n, 3), dtype=np.int64)
    for i in prange(n):
        vx = int(np.floor(points[i, 0] * inv_voxel_size))
        vy = int(np.floor(points[i, 1] * inv_voxel_size))
        vz = int(np.floor(points[i, 2] * inv_voxel_size))
        keys[i, 0] = vx
        keys[i, 1] = vy
        keys[i, 2] = vz

    return keys

class OctreeProbabilisticMap:
    def __init__(self, point_cloud: np.ndarray = None, max_depth: int = 8, 
                 map_size: float = 256.0, center: np.ndarray = None):
        """
        基于八叉树的概率占用地图
        
        参数:
            point_cloud: 初始点云(N×3 numpy数组)
            max_depth: 八叉树最大深度
            map_size: 地图尺寸 (默认为 256x256x256)
            center: 地图中心坐标 (默认为原点 [0,0,0])
        """
        # 设置地图尺寸和中心
        self.map_size = map_size
        self.center = np.array([0, 0, 0]) if center is None else np.array(center)
        self.max_depth = max_depth
        
        # 计算地图边界
        half_size = map_size / 2
        self.min_bound = self.center - half_size
        self.max_bound = self.center + half_size
        
        # 初始化八叉树
        self.octree = None
        self.initialize_octree(max_depth, self.map_size, self.center)
        
        # 初始化概率地图
        self.probability_map = {}
        
        # 体素大小 = 地图大小 / (2^最大深度)
        self.voxel_size = map_size / (2 ** max_depth)
        
        # 线程锁，保护共享资源
        self.lock = threading.RLock()
        
        # 最后更新时间戳
        self.last_update_time = time.time()
        
        # 添加初始点云数据
        if point_cloud is not None and len(point_cloud) > 0:
            self.update_map(point_cloud)
    
    def initialize_octree(self, max_depth, map_size=256.0, center=None):
        """初始化八叉树，兼容不同版本的Open3D"""
        # 计算地图边界
        if center is None:
            center = np.array([0, 0, 0])
        else:
            center = np.array(center)
            
        half_size = map_size / 2
        min_bound = center - half_size
        max_bound = center + half_size
        
        try:
            # 尝试使用新版API创建八叉树
            self.octree = o3d.geometry.Octree(max_depth=max_depth)
            
            # 存储边界信息，因为某些版本可能不提供max_bound属性
            self.octree_min_bound = min_bound
            self.octree_max_bound = max_bound
            
            # 尝试获取八叉树边界
            try:
                if hasattr(self.octree, 'max_bound'):
                    self.octree.max_bound = max_bound
                    self.octree.min_bound = min_bound
                # 某些版本可能使用不同的属性名
                elif hasattr(self.octree, 'set_max_bound'):
                    self.octree.set_max_bound(max_bound)
                    self.octree.set_min_bound(min_bound)
            except:
                # 使用已设置的值
                pass
                
        except TypeError:
            # 兼容旧版API
            try:
                self.octree = o3d.geometry.Octree(max_depth)
                # 存储边界信息
                self.octree_min_bound = min_bound
                self.octree_max_bound = max_bound
            except Exception as e:
                print(f"创建八叉树时出错: {e}")
                # 创建一个具有默认边界的八叉树
                self.octree = o3d.geometry.Octree()
                # 保存最大深度
                if hasattr(self.octree, 'max_depth'):
                    self.octree.max_depth = max_depth
                # 存储边界信息
                self.octree_min_bound = min_bound
                self.octree_max_bound = max_bound
        
        # 存储最大深度
        self.max_depth = max_depth

    def update_map(self, points: np.ndarray, point_type: str = 'obstacle'):
        """
        高速批量更新地图概率 (极致向量化/自动集成)
        """
        if len(points) == 0:
            return
            
        # 安全检查：确保有 probability_map 属性
        if not hasattr(self, 'probability_map'):
            self.probability_map = {}
            
        voxel_size = self.voxel_size
        voxel_keys = np.floor(points / voxel_size)
        voxel_keys = voxel_keys.astype(np.int32)
        
        # 直接对数组使用 np.unique (axis=0 保证按行去重)
        unique_keys, counts = np.unique(voxel_keys, axis=0, return_counts=True)
        unique_keys_tuples = [tuple(x) for x in unique_keys]  # 转为元组列表
        
        is_obstacle = (point_type.lower() == 'obstacle')
        # 批量取出旧概率
        old_probs = np.array([self.probability_map.get(k, 0.5) for k in unique_keys_tuples])
        update_factors = np.minimum(0.1 * counts, 0.5)
        if is_obstacle:
            new_probs = old_probs + (1 - old_probs) * update_factors
        else:
            new_probs = old_probs - old_probs * update_factors
        new_probs = np.clip(new_probs, 0.0, 1.0)
        
        # 更新 probability_map
        for i, k in enumerate(unique_keys_tuples):
            self.probability_map[k] = new_probs[i]  # 直接使用元组作为键
    
    def _estimate_voxel_size(self):
        """估计体素大小"""
        return self.voxel_size

    def get_voxel_probability(self, point: np.ndarray) -> float:
        """获取指定点所在体素的占据概率"""
        # 计算点所在的体素坐标
        voxel_size = self._estimate_voxel_size()
        voxel_key = tuple(np.floor(point / voxel_size).astype(np.int32))
        
        # 返回体素概率，如果不存在则返回默认概率0.5
        return self.probability_map.get(voxel_key, 0.5)

class OctreeMapManager:
    def __init__(self, map_size=256.0, max_depth=8, grid_size=None, overlap=0.0):
        """
        八叉树地图管理器 - 管理多个子地图，并处理它们之间的协调
        
        参数:
            map_size: 每个子地图的尺寸 (默认256x256x256)
            max_depth: 八叉树最大深度 (默认8)
            grid_size: 子地图网格大小 (默认等于map_size)
            overlap: 子地图之间的重叠比例 (0-1)
        """
        self.map_size = map_size
        self.max_depth = max_depth
        
        # 如果未指定网格大小，则使用地图大小
        self.grid_size = map_size if grid_size is None else grid_size
        
        # 子地图之间的重叠比例 (0-1)
        self.overlap = np.clip(overlap, 0.0, 0.5)  # 限制最大重叠50%
        
        # 存储所有子地图的字典 {(grid_x, grid_y, grid_z): OctreeProbabilisticMap}
        self.submaps = {}
        
        # 线程池，用于并行操作
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 全局边界 (所有子地图的边界合并)
        self.global_min_bound = None
        self.global_max_bound = None
        
        # 用于管理并发访问的锁
        self.lock = threading.RLock()
        
        # 缓存以加速计算
        self._grid_to_bounds_cache = {}
        self._point_to_grid_cache = {}
        self._cache_size_limit = 10000  # 缓存大小限制
        
        # 地图重置标志
        self.reset_index = False
    
    def _normalize_to_grid(self, point):
        """
        将点坐标标准化到网格坐标系统
        
        参数:
            point: 3D点坐标 [x, y, z]
            
        返回:
            [normalized_x, normalized_y, normalized_z]: 网格化的坐标
        """
        # 将点坐标除以网格大小，得到浮点网格坐标
        return np.array(point) / self.grid_size
    
    def _get_grid_coords(self, point):
        """
        计算点所属的网格坐标，保证网格边界与地图尺寸对齐
        
        参数:
            point: 3D点坐标 [x, y, z]
            
        返回:
            (grid_x, grid_y, grid_z): 网格索引坐标
        """
        # 检查缓存
        point_tuple = tuple(point)
        if point_tuple in self._point_to_grid_cache:
            return self._point_to_grid_cache[point_tuple]
            
        # 标准化坐标
        normalized = self._normalize_to_grid(point)
        
        # 计算网格索引 - 使用floor确保负数正确取整
        # 注意：网格大小等于地图大小 * (1-overlap)
        grid_coords = tuple(np.floor(normalized).astype(int))
        
        # 更新缓存
        if len(self._point_to_grid_cache) >= self._cache_size_limit:
            # 简单的缓存清理策略：清除一半缓存
            keys_to_remove = list(self._point_to_grid_cache.keys())[:self._cache_size_limit//2]
            for key in keys_to_remove:
                del self._point_to_grid_cache[key]
        
        self._point_to_grid_cache[point_tuple] = grid_coords
        return grid_coords
    
    def _get_map_center(self, grid_coords):
        """
        计算指定网格的地图中心点，确保与网格边界对齐
        
        参数:
            grid_coords: (grid_x, grid_y, grid_z) 网格索引
            
        返回:
            [center_x, center_y, center_z]: 地图中心坐标
        """
        # 网格中心 = (网格索引 + 0.5) * 网格大小
        # 这确保了地图中心正好位于网格的中心
        grid_x, grid_y, grid_z = grid_coords
        center = (np.array([grid_x, grid_y, grid_z]) + 0.5) * self.grid_size
        return center
    
    def _get_grid_bounds(self, grid_coords):
        """
        计算网格的边界
        
        参数:
            grid_coords: (grid_x, grid_y, grid_z) 网格索引
            
        返回:
            (min_bound, max_bound): 网格的最小和最大边界
        """
        # 检查缓存
        if grid_coords in self._grid_to_bounds_cache:
            return self._grid_to_bounds_cache[grid_coords]
            
        center = self._get_map_center(grid_coords)
        half_size = self.map_size / 2
        min_bound = center - half_size
        max_bound = center + half_size
        
        # 更新缓存
        bounds = (min_bound, max_bound)
        self._grid_to_bounds_cache[grid_coords] = bounds
        return bounds
    
    def _create_map_for_point(self, point):
        """
        为指定点创建对应的地图（如果不存在）
        
        参数:
            point: 3D点坐标 [x, y, z]
            
        返回:
            octree_map: 新创建或已存在的八叉树地图
        """
        grid = self._get_grid_coords(point)
        
        # 如果该网格已有地图，则直接返回
        if grid in self.submaps:
            return self.submaps[grid]
        
        # 计算地图中心
        center = self._get_map_center(grid)
        
        # 创建新地图
        with self.lock:
            if grid not in self.submaps:  # 再次检查，防止并发创建
                octree_map = OctreeProbabilisticMap(map_size=self.map_size, 
                                                  max_depth=self.max_depth,
                                                  center=center)
                self.submaps[grid] = octree_map
                
                # 更新全局边界
                self._update_global_bounds(octree_map.min_bound, octree_map.max_bound)
                
                return octree_map
            else:
                return self.submaps[grid]

    def _update_global_bounds(self, min_bound, max_bound):
        """更新全局边界"""
        if self.global_min_bound is None:
            self.global_min_bound = min_bound.copy()
            self.global_max_bound = max_bound.copy()
        else:
            self.global_min_bound = np.minimum(self.global_min_bound, min_bound)
            self.global_max_bound = np.maximum(self.global_max_bound, max_bound)

    def update_with_obstacles(self, points: np.ndarray):
        """
        批量添加障碍点并自动集成更新（分配到子地图）
        
        参数:
            points: 障碍点云，shape=(N,3)的NumPy数组
        """
        if len(points) == 0:
            return
            
        # 按网格分组点云
        grid_points = defaultdict(list)
        for p in points:
            g = self._get_grid_coords(p)
            grid_points[g].append(p)
        
        # 更新每个网格的子地图
        for grid, pts in grid_points.items():
            pts_array = np.asarray(pts, dtype=np.float32)
            octree_map = self.submaps.get(grid) or self._create_map_for_point(pts_array[0])
            octree_map.update_map(pts_array, point_type='obstacle')

    def update_with_free_space(self, points: np.ndarray):
        """
        批量添加空闲点并自动集成更新（分配到子地图）
        
        参数:
            points: 空闲点云，shape=(N,3)的NumPy数组
        """
        if len(points) == 0:
            return
            
        # 按网格分组点云
        grid_points = defaultdict(list)
        for p in points:
            g = self._get_grid_coords(p)
            grid_points[g].append(p)
        
        # 更新每个网格的子地图
        for grid, pts in grid_points.items():
            pts_array = np.asarray(pts, dtype=np.float32)
            octree_map = self.submaps.get(grid) or self._create_map_for_point(pts_array[0])
            octree_map.update_map(pts_array, point_type='free')

    def update_with_mixed_points(self, obstacle_points: np.ndarray, free_points: np.ndarray):
        """
        批量添加混合点云（障碍点和空闲点）
        
        参数:
            obstacle_points: 障碍点云，shape=(N,3)的NumPy数组
            free_points: 空闲点云，shape=(M,3)的NumPy数组
        """
        # 先处理障碍点
        self.update_with_obstacles(obstacle_points)
        # 再处理空闲点
        self.update_with_free_space(free_points)

    def get_stats(self):
        """
        获取地图统计信息
        
        返回:
            stats: 包含统计信息的字典
        """
        total_voxels = 0
        occupied_voxels = 0
        
        # 遍历所有子地图
        for grid, octree_map in self.submaps.items():
            # 安全检查：确保子地图有 probability_map 属性
            if hasattr(octree_map, 'probability_map'):
                total_voxels += len(octree_map.probability_map)
                # 计算占据体素数量 (概率 > 0.5)
                occupied_voxels += sum(1 for prob in octree_map.probability_map.values() if prob > 0.5)
        
        return {
            'total_voxels': total_voxels,
            'occupied_voxels': occupied_voxels,
            'global_min_bound': self.global_min_bound,
            'global_max_bound': self.global_max_bound,
            'submap_count': len(self.submaps)
        }

    def query_points(self, points, threshold=None):
        """
        批量查询点的占据状态
        
        参数:
            points: 点云数据 (N, 3) 形状的numpy数组
            threshold: 概率阈值，若指定则返回布尔数组 (True=占用, False=空闲)
            
        返回:
            is_occupied: 每个点是否被占用的布尔数组
        """
        if len(points) == 0:
            return np.array([], dtype=bool)
        
        # 初始化结果数组
        probabilities = np.full(len(points), 0.5, dtype=np.float32)
        
        # 按网格分组点云
        grid_points = defaultdict(list)
        for i, p in enumerate(points):
            g = self._get_grid_coords(p)
            grid_points[g].append((i, p))
        
        # 查询每个网格的子地图
        for grid, pts in grid_points.items():
            # 如果该网格有地图
            if grid in self.submaps:
                octree_map = self.submaps[grid]
                
                # 查询每个点的概率
                for i, p in pts:
                    probabilities[i] = octree_map.get_voxel_probability(p)
        
        # 如果指定了阈值，返回布尔数组
        if threshold is not None:
            return probabilities > threshold
        
        return probabilities

    def get_region_probabilities(self, bbox_min, bbox_max, return_coords=False):
        """
        获取指定立方体区域内的体素概率
        
        参数:
            bbox_min: 区域最小角点 [x,y,z]
            bbox_max: 区域最大角点 [x,y,z]
            return_coords: 是否返回坐标矩阵 (默认为False)
            
        返回:
            probabilities: 概率字典 {voxel_key: probability}
            如果return_coords=True，还会返回 coords: 体素坐标矩阵 (N,3)
            以及 prob_values: 对应的概率值数组 (N,)
        """
        # 确保输入是numpy数组
        if not isinstance(bbox_min, np.ndarray):
            bbox_min = np.array(bbox_min, dtype=np.float32)
        if not isinstance(bbox_max, np.ndarray):
            bbox_max = np.array(bbox_max, dtype=np.float32)
            
        # 获取所有受影响的网格
        affected_grids = []
        
        # 计算包含边界框的所有网格
        min_grid = self._get_grid_coords(bbox_min)
        max_grid = self._get_grid_coords(bbox_max)
        
        # 遍历所有可能的网格
        for x in range(min_grid[0], max_grid[0] + 1):
            for y in range(min_grid[1], max_grid[1] + 1):
                for z in range(min_grid[2], max_grid[2] + 1):
                    affected_grids.append((x, y, z))
        
        # 收集结果
        combined_probs = {}
        all_coords = [] if return_coords else None
        all_prob_values = [] if return_coords else None
        
        # 处理每个网格
        for grid in affected_grids:
            if grid in self.submaps:
                octree_map = self.submaps[grid]
                # 安全检查：确保子地图有 probability_map 属性
                if not hasattr(octree_map, 'probability_map'):
                    print(f"警告: 网格 {grid} 的子地图没有 probability_map 属性，跳过")
                    continue
                    
                # 获取体素大小
                voxel_size = octree_map._estimate_voxel_size()
                
                # 筛选在边界框内的体素
                for voxel_key, prob in octree_map.probability_map.items():
                    # 计算体素中心坐标
                    voxel_center = (np.array(voxel_key, dtype=np.float64) + 0.5) * voxel_size
                    
                    # 检查是否在边界框内
                    if (voxel_center >= bbox_min).all() and (voxel_center <= bbox_max).all():
                        combined_probs[voxel_key] = prob
                        if return_coords:
                            all_coords.append(voxel_center)
                            all_prob_values.append(prob)
        
        # 返回结果
        if return_coords:
            if all_coords:
                return combined_probs, np.array(all_coords), np.array(all_prob_values)
            else:
                return combined_probs, np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        else:
            return combined_probs

    def reset(self):
        """
        重置地图，清除所有子地图
        """
        self.reset_index = True
        self.submaps.clear()
        # 重置边界
        self.global_min_bound = None
        self.global_max_bound = None
        # 清除所有缓存
        self._grid_to_bounds_cache.clear()
        self._point_to_grid_cache.clear()
        self.reset_index = False

    def visualize(self, show_probability=True, only_occupied=False, prob_threshold=0.5):
        """
        可视化所有子地图的体素
        
        参数:
            show_probability: 是否显示概率颜色(True=概率颜色, False=统一颜色)
            only_occupied: 是否只显示占据体素
            prob_threshold: 占据概率阈值(仅在only_occupied=True时使用)
        """
        # 创建Open3D可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="OctreeMap Visualization", width=1280, height=720)
        
        # 计数器
        voxel_count = 0
        filtered_count = 0
        
        # 收集所有点
        all_points = []
        all_colors = []
        
        # 调试信息
        print(f"子地图数量: {len(self.submaps)}")
        
        # 遍历所有子地图
        for grid, octree_map in self.submaps.items():
            # 获取地图中心
            center = self._get_map_center(grid)
            
            # 安全检查：确保子地图有 probability_map 属性
            if not hasattr(octree_map, 'probability_map'):
                print(f"警告: 网格 {grid} 的子地图没有 probability_map 属性，跳过")
                continue
                
            # 调试信息
            print(f"网格 {grid} 体素数量: {len(octree_map.probability_map)}")
            
            # 获取所有体素
            try:
                voxel_size = octree_map._estimate_voxel_size()
                for voxel_key, prob in octree_map.probability_map.items():
                    voxel_count += 1
                    
                    # 如果只显示占据体素，则跳过概率低于阈值的体素
                    if only_occupied and prob < prob_threshold:
                        filtered_count += 1
                        continue
                    
                    # 体素中心
                    voxel_center = (np.array(voxel_key, dtype=np.float64) + 0.5) * voxel_size
                    all_points.append(voxel_center)
                    
                    # 基于概率设置颜色
                    if show_probability:
                        # 红色表示高概率，蓝色表示低概率
                        r = prob if prob > 0.5 else 0
                        b = 1 - prob if prob < 0.5 else 0
                        g = 1 - abs(2 * prob - 1)  # 绿色在0.5概率处最大
                        all_colors.append([r, g, b])
                    else:
                        # 统一颜色
                        all_colors.append([0.5, 0.5, 0.5])
            except Exception as e:
                print(f"处理网格 {grid} 时出错: {str(e)}")
        
        # 如果没有点，则提前返回
        if not all_points:
            print("没有体素可显示")
            return
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        # 添加点云到可视化器
        vis.add_geometry(pcd)
        
        # 优化视图
        vis.get_render_option().point_size = 5
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        vis.get_view_control().set_zoom(0.8)
        
        # 输出统计信息
        print(f"显示 {voxel_count - filtered_count} 个体素 (总共 {voxel_count} 个, 过滤了 {filtered_count} 个)")
        
        # 运行可视化器
        vis.run()
        vis.destroy_window()
    
    def get_all_maps_pointcloud(self, threshold=0.5, only_occupied=True):
        """
        获取所有子地图的点云表示
        
        参数:
            threshold: 概率阈值，超过该值判定为占用
            only_occupied: 是否只返回占据体素
            
        返回:
            pcd: Open3D点云对象
        """
        # 收集所有点
        all_points = []
        all_colors = []
        
        # 遍历所有子地图
        for grid, octree_map in self.submaps.items():
            # 安全检查：确保子地图有 probability_map 属性
            if not hasattr(octree_map, 'probability_map'):
                print(f"警告: 网格 {grid} 的子地图没有 probability_map 属性，跳过")
                continue
                
            # 获取所有体素
            voxel_size = octree_map._estimate_voxel_size()
            for voxel_key, prob in octree_map.probability_map.items():
                # 如果只显示占据体素，则跳过概率低于阈值的体素
                if only_occupied and prob < threshold:
                    continue
                
                # 体素中心
                voxel_center = (np.array(voxel_key, dtype=np.float64) + 0.5) * voxel_size
                all_points.append(voxel_center)
                
                # 基于概率设置颜色
                r = prob if prob > 0.5 else 0
                b = 1 - prob if prob < 0.5 else 0
                g = 1 - abs(2 * prob - 1)  # 绿色在0.5概率处最大
                all_colors.append([r, g, b])
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        if all_points:
            pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        return pcd

# 示例使用
if __name__ == "__main__":
    print("开始测试八叉树概率占用地图管理器...")
    
    # 基本功能测试
    def basic_test():
        try:
            # 创建地图管理器（使用较小的子地图大小，便于测试）
            print("\n创建八叉树地图管理器...")
            start_time = time.time()
            # 使用较小的地图尺寸，以便创建多个子地图
            map_manager = OctreeMapManager(map_size=64, max_depth=7, overlap=0.1)
            print(f"地图管理器创建完成，耗时: {time.time() - start_time:.2f}秒")
            
            # 生成测试点云 - 在不同位置创建多个点云
            print("\n生成测试点云...")
            
            # 在多个不同位置创建障碍物，确保会创建多个子地图
            locations = [
                [0, 0, 0],        # 原点
                [100, 100, 100],  # 第一象限
                [-100, -100, -100], # 第三象限
                [200, -100, 50],  # 混合象限
                [32, 32, 32],    # 靠近原点但不同网格
                [300, 200, -200]   # 对角线正方向
            ]
            
            for loc in locations:
                # 创建一个球体障碍物
                sphere_center = np.array(loc)
                obstacles = []
                
                for _ in range(2000):  # 每个位置500个点
                    r = 20 * np.random.rand()  # 半径20以内
                    theta = np.random.rand() * 2 * np.pi
                    phi = np.random.rand() * np.pi
                    
                    x = r * np.sin(phi) * np.cos(theta)
                    y = r * np.sin(phi) * np.sin(theta)
                    z = r * np.cos(phi)
                    
                    obstacles.append(sphere_center + np.array([x, y, z]))
                
                obstacles = np.array(obstacles, dtype=np.float32)
                
                # 更新地图
                for i in range(10):
                    print(f"更新位置 {loc} 的地图...")
                    timea = time.time()
                    map_manager.update_with_obstacles(obstacles)  # 使用正确的方法
                    map_manager.update_with_free_space(obstacles+3)  # 添加一些空闲点
                    timeb = time.time()
                    print(f'插入点所用时间{timeb-timea}')
            
            # 打印地图统计信息
            stats = map_manager.get_stats()
            print("\n地图统计信息:")
            print(f"  子地图数量: {len(map_manager.submaps)}")
            print(f"  总体素数量: {stats['total_voxels']}")
            print(f"  全局边界: {stats['global_min_bound']} -> {stats['global_max_bound']}")
            
            # 查询测试
            print("\n进行区域查询测试...")
            # 在原点附近查询一个区域
            bbox_min = np.array([1, 1, 1])
            bbox_max = np.array([41, 41, 41])
            for i in range(5):
                timea = time.time()
                probs, coords, prob_values = map_manager.get_region_probabilities(bbox_min, bbox_max, return_coords=True)
                timeb = time.time()
                print(f"  查询区域体素数量: {len(coords)}")
                print(f"  体素坐标: {coords[:5] if len(coords) > 5 else coords}")  # 打印前5个坐标作为示例
                print(f"  对应概率: {prob_values[:5] if len(prob_values) > 5 else prob_values}")  # 打印前5个概率值
                print(f"  查询耗时: {timeb - timea:.5f}秒")

            # 可视化
            print("\n准备可视化...")
            try:
                # 可视化全部地图
                print("\n显示所有体素:")
                map_manager.visualize(show_probability=True)
                
                # 显示只有占据体素的地图
                print("\n只显示占据体素 (概率 >= 0.5):")
                map_manager.visualize(show_probability=True, only_occupied=True, prob_threshold=0.5)
                
                # 显示高概率占据体素的地图
                print("\n只显示高概率占据体素 (概率 >= 0.7):")
                map_manager.visualize(show_probability=True, only_occupied=True, prob_threshold=0.7)
                
                # 获取点云并显示
                print("\n获取并显示点云 (只包含占据体素):")
                pcd = map_manager.get_all_maps_pointcloud(threshold=0.5, only_occupied=True)
                
                # 如果用户希望，可以保存点云
                # pcd.write("occupied_pointcloud.ply")
                
                import open3d as o3d
                o3d.visualization.draw_geometries([pcd])
                
            except Exception as e:
                print(f"可视化时出错: {e}")
                import traceback
                traceback.print_exc()
            
            timea = time.time()
            map_manager.reset()
            timeb = time.time()
            print(f'重置地图所用时间{timeb-timea}')
                
        except Exception as e:
            print(f"程序运行中出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行基本功能测试
    basic_test()
