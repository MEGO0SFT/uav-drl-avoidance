import numpy as np
import pandas as pd
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree
import numba
from numba import jit, njit, prange, float32, int32, int64
from numba import types, typed

# 最后更新时间 2025年5月22日
# 作者： 石辅天
# 版本： stable v1.4.1 (优化多线程性能)
# 待更新：暂无
global OPG_dict, OctoMap_dict
global MAP_SIZE, ORIGIN
OPG_dict = dict()       # key: tuple(x,y,z), value: timestamp
OctoMap_dict = dict()   # key: tuple(x,y,z), value: OctoMap对象
OctoMap_lock = threading.RLock() # 互斥锁,防止插入时的数据竞争
opg_locks = defaultdict(threading.RLock)  # 每个OPG的专属锁

# 使用JIT加速的辅助函数
@njit(parallel=True)
def process_opg_points_jit(points, indices, probabilities, query_output, query_range):
    """JIT加速的处理OPG点函数"""
    # 筛选有效概率 (非0.5)
    for i in prange(len(probabilities)):
        prob = probabilities[i]
        if prob != 0.5:  # 只更新非默认值
            idx = indices[i].astype(np.int32)
            # 确保索引在有效范围内
            if 0 <= idx[0] < query_output.shape[0] and 0 <= idx[1] < query_output.shape[1] and 0 <= idx[2] < query_output.shape[2]:
                query_output[idx[0], idx[1], idx[2]] = prob
    
    return query_output

MAP_SIZE = 256
ORIGIN = np.zeros(3)

class OctoMaps:
    def __init__(self, origin=None, map_size=None, min_resolution=1, max_depth=16):
        global OPG_dict, OctoMap_dict, OctoMap_lock
        self.OPG_dict= OPG_dict
        self.OctoMap_dict = OctoMap_dict
        self.origin = ORIGIN if origin == None else origin                 # 全局八叉树地图中心
        self.map_size = MAP_SIZE if map_size == None else map_size         # 八叉树地图尺寸
        self.min_resolution = min_resolution  # 最小体素分辨率
        self.max_depth = max_depth            # 最大树深度
        self.thread_pool = ThreadPoolExecutor(max_workers=8)  # 线程上限
        self.lock_dict = defaultdict(threading.Lock)  # 新增锁字典初始化
        self._executor = ThreadPoolExecutor(max_workers=4)  # 共享线程池
        self.query_fig = None
        self.reset_index = False
        
        # 预热Numba函数，确保首次使用时不会有编译延迟
        warmup_numba_functions()
        
        Octomap_creator(self.origin, self.map_size, self.min_resolution, self.max_depth)
    
    def Custom_OctoMapQuery(self, position: np.array, query_range):
        """极致优化版：针对小区域(8x8x8)的快速查询"""
        if self.reset_index:
            return
        
        query_size = 2 * query_range + 1
        query_output = np.full((query_size, query_size, query_size), 0.5, dtype=np.float32)
    
        # 1. 直接生成查询点坐标网格 - 避免使用meshgrid
        x_min, x_max = position[0] - query_range, position[0] + query_range
        y_min, y_max = position[1] - query_range, position[1] + query_range
        z_min, z_max = position[2] - query_range, position[2] + query_range
    
        # 预分配内存
        query_points = np.empty((query_size, query_size, query_size, 3), dtype=np.float32)
    
        # 使用向量化填充坐标
        for i in range(query_size):
            for j in range(query_size):
                for k in range(query_size):
                    query_points[i, j, k] = [
                    x_min + i,
                    y_min + j,
                    z_min + k
                    ]
    
        # 2. 计算所有点的OPG坐标 - 向量化操作
        signs = np.sign(query_points - self.origin)
        opg_coords = ((query_points - self.origin - signs * self.map_size/2) / 
                  self.map_size + signs).astype(np.int32)
    
        # 3. 获取唯一的OPG坐标
        unique_opgs = np.unique(opg_coords.reshape(-1, 3), axis=0)
    
        # 4. 筛选存在的OPG - 使用集合操作加速
        with OctoMap_lock:
            existing_opg_keys = set(self.OPG_dict.keys())
    
        existing_mask = np.array([tuple(opg) in existing_opg_keys for opg in unique_opgs])
        existing_opgs = unique_opgs[existing_mask]
    
         # 5. 单线程处理每个OPG - 避免线程开销
        for opg in existing_opgs:
            # 获取属于此OPG的点
            mask = np.all(opg_coords == opg, axis=3)
            points_in_opg = query_points[mask]
        
            if points_in_opg.size == 0:
                continue
            
            # 获取八叉树地图
            with opg_locks[tuple(opg)]:
                octomap = OPG_Positioning(opg)
            
            if octomap is None:
                continue
            
            # 批量查询概率
            probabilities = octomap.batch_query_points(points_in_opg.reshape(-1, 3))
        
            # 更新查询输出
            indices = np.argwhere(mask)
            query_output[indices[:, 0], indices[:, 1], indices[:, 2]] = probabilities
    
        return query_output
    
    def circle_query_points(self, origin, azimuth_step, elevation_step, radius):
        """批量查询多个点的占据概率"""
        step = 1
        total_length = 0
        azimuths = np.arange(0, 360, azimuth_step)
        elevations = np.arange(-90, 91, elevation_step)
        directions = np.zeros((len(azimuths), len(elevations), 3), dtype=np.float32)
        for i, az in enumerate(azimuths):
            for j, el in enumerate(elevations):
                directions[i, j] = _direction_vector(az, el)
        opgs = np.unique(_calc_opgs(origin + radius*directions.reshape(-1,3)), axis=0)
        total_points_p = np.zeros((1,4), dtype=np.float32)
        while total_length < radius:
            if total_length >= 5:
                step += 0.25
            total_length += step
            all_points = directions * total_length + origin
            # 体素滤波
            points = voxel_downsample_jit(all_points.reshape(-1,3), step)
            points_opg = _calc_opgs(points)
            for opg in opgs:
                mask = np.all(points_opg == opg, axis=1)
                if np.any(mask):
                    # 处理匹配的OPG
                    octomap = OPG_Positioning(opg)
                    if octomap is None:
                        continue
                    # 批量查询概率
                    probabilities = octomap.batch_query_points(points[mask])
                    # 更新查询输出
                    points_p = np.zeros((len(probabilities), 4), dtype=np.float32)
                    points_p[:, :3] = points[mask]
                    points_p[:, 3] = probabilities
                    total_points_p = np.vstack((total_points_p, points_p))
        return total_points_p[1:]  # 返回去除初始化的空行


    def _process_opg_batch_chunk(self, all_opg_coords, all_query_points, all_idx_offset, opg_batch):
        """处理OPG批次的优化方法 - 每个线程处理多个OPG"""
        all_update_indices = []
        all_update_probs = []
    
        for opg in opg_batch:
            # 使用OPG级别的细粒度锁
            with opg_locks[tuple(opg)]:
                octomap = OPG_Positioning(opg)
            
            if octomap is None:
                continue
            
            # 获取属于此OPG的点
            mask = np.all(all_opg_coords == opg, axis=1)
            points_in_opg = all_query_points[mask]
            indices_in_opg = all_idx_offset[mask]
        
            if len(points_in_opg) == 0:
                continue
            
            # 使用批处理查询
            probabilities = np.array(octomap.query_points(points_in_opg))
            update_mask = probabilities != 0.5
        
            if np.any(update_mask):
                all_update_indices.append(indices_in_opg[update_mask])
                all_update_probs.append(probabilities[update_mask])
    
        # 合并结果 - 添加空数组检查
        if len(all_update_indices) > 0:
            return np.vstack(all_update_indices), np.concatenate(all_update_probs)
        else:
            # 返回空数组而不是空列表
            return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32)

    def visualize_octomap_query(self, position: np.array, query_range):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # 获取占据体素坐标
        query_output = self.Custom_OctoMapQuery(position, query_range)
        occupied_mask = query_output > 0.5
        free_mask = query_output < 0.5
        center_mask = query_output == 2
    
        # 创建坐标网格
        x, y, z = np.indices(query_output.shape)
    
        # 绘制占据体素(红色)
        ax.scatter(x[occupied_mask], y[occupied_mask], z[occupied_mask], 
                c='r', marker='o', s=50, label='Occupied')
        # 绘制中心点(蓝色)
        ax.scatter(x[center_mask], y[center_mask], z[center_mask], 
                c='b', marker='*', s=200, label='Center')
    
        ax.set_xlim(0, 2*query_range)
        ax.set_ylim(2*query_range,0)
        ax.set_zlim(0, 2*query_range)
    
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('OctoMap Query Visualization')
        ax.legend()
        plt.tight_layout()
        self.query_fig = fig
        plt.show()
        plt.pause(2)
        plt.close(fig)

    def visualize_octomap_query_nopltshow(self, position: np.array, query_range, start_time):
    
        # 获取占据体素坐标
        query_output = self.Custom_OctoMapQuery(position, query_range)
    
        return query_output
        
    def Fast_insert_point(self, position: np.array, points_list, free_points_list, chunk_size=1024):
        """"
        根据无人机当前位置信息，快速插入点到最近的八叉树地图中\n
        :param position: 无人机当前位置信息
        :param points_list: 深度信息点集，增加体素概率
        :param free_points_list: 空闲信息点集，降低体素概率
        """
        # 计算无人机所在的OPG坐标
        if self.reset_index:
            return
        if points_list.size == 0:
            return
        voxel_size = 0.25  # 25cm体素滤波
        points_list = voxel_downsample_jit(points_list, voxel_size)  # 降低点云数据量
        # free_points_list = remove_nearby_points(free_points_list, points_list, radius=2.0)
        points = np.vstack((points_list, free_points_list))
        node_true, node_false= np.ones((points_list.shape[0],1)), np.zeros((free_points_list.shape[0], 1))
        nodes = np.vstack((node_true, node_false))
        located_OPG = _calc_opg(position)
        # 查询该OPG坐标下的八叉树地图是否已经存在，如果不存在则创建
        if not _is_OctoMap_Center_created(located_OPG*self.map_size):
            Octomap_creator(located_OPG*self.map_size, self.map_size, self.min_resolution, self.max_depth)
        octomap = OPG_Positioning(located_OPG)
        points_xyz_d = points - octomap.root.center
        # 对数据进行分块
        if len(points)//chunk_size > 0:
            points_chunks = np.array_split(points, len(points)//chunk_size)
            xyz_dchunks = np.array_split(points_xyz_d, len(points_xyz_d)//chunk_size)
            node_chunks = np.array_split(nodes, len(nodes)//chunk_size)
            futures = []
            for p_chunk, x_chunk, n_chunk in zip(points_chunks, xyz_dchunks, node_chunks):
                    future = self.thread_pool.submit(
                        self._batch_insert_wrapper, 
                        octomap, p_chunk, x_chunk, n_chunk
                    )
            futures.append(future)

            for f in futures:
                f.result()
        else:
            points_chunks = points
            xyz_dchunks = points_xyz_d
            n_chunks = nodes
            futures = []
            future = self.thread_pool.submit(
                    self._batch_insert_wrapper, 
                    octomap, points_chunks, xyz_dchunks, n_chunks
                )
            futures.append(future)

            for f in futures:
                f.result()
    
    def _batch_insert_wrapper(self, octomap, points, xyz_d, node_judgment):
        """优化后的批量插入方法（矩阵运算+分块锁）"""
        if self.reset_index:
            return
        block_size = 10.0  # 空间分块尺寸
        # 步骤1：计算所有点的分块坐标（向量化）
        block_coords = np.floor(points / block_size).astype(int)
    
        # 步骤2：生成唯一分块索引（快速哈希）
        block_hash = block_coords[:,0] * 1000000 + block_coords[:,1] * 1000 + block_coords[:,2]
    
         # 步骤3：按分块分组点云（避免循环）
        unique_blocks, inverse_indices = np.unique(block_hash, return_inverse=True)
        grouped_points = [points[inverse_indices == i] for i in range(len(unique_blocks))]
        grouped_xyz_d = [xyz_d[inverse_indices == i] for i in range(len(unique_blocks))]
        grouped_node = [node_judgment[inverse_indices == i] for i in range(len(unique_blocks))]
    
        # 步骤4：并行处理每个分块（线程池+动态批处理）
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for block_points, block_xyz, block_node in zip(grouped_points, grouped_xyz_d, grouped_node):
                if len(block_points) == 0:
                    continue
                
                # 将分块坐标转换为元组作为锁键
                sample_coord = block_points[0] // block_size
                block_key = tuple(sample_coord.astype(int))
            
                # 提交并行任务
                futures.append(
                    executor.submit(
                        self._insert_block,  # 新增的块插入方法
                        octomap, 
                        block_key,
                        block_points,
                        block_xyz,
                        block_node
                    )
                )
            
            # 等待所有任务完成
            _ = [f.result() for f in futures]
        
    def _insert_block(self, octomap, block_key, points, xyz_d, node_judgment):
        """单个分块的原子插入操作"""
        if self.reset_index:
            return 
        with self.lock_dict[block_key]:  # 分块级锁
            octomap.fast_insert_point(points, xyz_d, node_judgment)  # 假设已支持矩阵输入

    def fast_insert_point_thread(self, octomap, points_list, xyz_d_list):
        """批量插入优化（弃用）"""
        # 根据空间位置哈希分块加锁
        block_size = 1.0  # 1m分块
        lock_dict = defaultdict(threading.Lock)
    
        for p in points_list:
            block_key = tuple((p // block_size).astype(int))
            with lock_dict[block_key]:  # 仅锁住空间块
                octomap.fast_insert_point(p, xyz_d_list)
    

    def global_reset(self):
        """重置所有八叉树地图"""
        global OPG_dict, OctoMap_dict
        print("OctoMap reset task begin")
        self.reset_index = True
        time.sleep(2)
        self.thread_pool.shutdown(wait=True)  # 等待现有任务完成    
        OctoMap_lock.acquire()
        for octomap in list(OctoMap_dict.values()):
            try:
                octomap.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")
            del octomap
        OPG_dict.clear()
        OctoMap_dict.clear()
        self.OctoMap_dict = OctoMap_dict
        self.OPG_dict = OPG_dict
        self.OctoMap_dict = OctoMap_dict
        self.OPG_dict = OPG_dict
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        OctoMap_lock.release()
        Octomap_creator(self.origin, self.map_size, self.min_resolution, self.max_depth)
        self.reset_index = False
        

class OctoNode:
    def __init__(self, center :np.array, size, prob=0.5):
        """
        八叉树节点类\n
        :param center: 节点中心坐标 (x,y,z)
        :param size: 立方体边长
        :param prob: 占据概率初始值（默认0.5表示未知）
        """
        self.center = center                # 立方体中心坐标
        self.size = size                    # 立方体边长
        self.prob = prob                    # 占据概率（0-1）
        self.children = [None]*8            # 子节点（最多8个）
        self.is_leaf = True                 # 是否为叶节点
        self.last_update = time.time()      # 最后更新时间
        self.ref_count = 0

    def subdivide(self):
        """将当前节点细分为8个子节点"""
        if not self.is_leaf:
            return
        
        child_size = self.size / 2
        offset = child_size / 2
        
        # 计算八个子节点的中心坐标
        x, y, z = self.center
        child_centers = [
            (x - offset, y - offset, z - offset),
            (x + offset, y - offset, z - offset),
            (x - offset, y + offset, z - offset),
            (x + offset, y + offset, z - offset),
            (x - offset, y - offset, z + offset),
            (x + offset, y - offset, z + offset),
            (x - offset, y + offset, z + offset),
            (x + offset, y + offset, z + offset)
        ]
        
        # 创建子节点
        for i in range(8):
            self.children[i] = OctoNode(child_centers[i], child_size)
        
        self.is_leaf = False

class OctoMap:
    def __init__(self, origin :np.array, Octree_Positioning_Grid: np.array, map_size=32, resolution=1, max_depth=16):
        """
        概率八叉树地图类\n
        :param origin: 地图原点坐标 np.array([x,y,z])
        :param Octree_Positioning_Grid: 八叉树地图位于总地图网格中的位置[x,y,z]
        :param map_size: 地图尺寸（立方体边长）
        :param resolution: 最小体素分辨率
        :param max_depth: 最大树深度
        """
        self.root = OctoNode(origin, map_size)  # 根节点
        self.resolution = resolution          # 最小体素尺寸
        self.max_depth = max_depth            # 最大树深度
        self.clamp_min = 0.03                 # 概率下限（防止数值下溢）
        self.clamp_max = 0.97                 # 概率上限（防止数值上溢）
        self.prob_hit = 0.7                  # 观测到占据时的概率更新值
        self.prob_miss = 0.40                 # 观测到空闲时的概率更新值
        self.OPG = Octree_Positioning_Grid    # 八叉树地图位于总地图网格中的位置

    def cleanup(self):
        """安全释放八叉树资源"""
        # 1. 广度优先遍历释放所有节点
        if not hasattr(self, 'root'):
            return
            
        stack = [self.root]
        while stack:
            node = stack.pop()
            
            # 递归释放子节点
            if not node.is_leaf:
                for i, child in enumerate(node.children):
                    if child:
                        stack.append(child)
                        node.children[i] = None  # 解除引用
                
                # 清空子节点数组
                node.children = []
            
            # 2. 显式删除节点属性（避免循环引用）
            del node.center, node.prob, node.last_update
        
        # 3. 删除根节点并清空属性
        del self.root
        self.root = None


    def log_odds(self, prob):
        """将概率转换为对数几率表示"""
        return np.log(prob / (1 - prob))

    def probability(self, log_odds):
        """将对数几率转换回概率"""
        return 1 - (1 / (1 + np.exp(log_odds)))

    def update_node(self, node, occupancy):
        """
        更新节点占据概率\n
        :param node: 要更新的节点
        :param occupancy: 观测结果（True=占据，False=空闲）
        """
        """向量化概率更新"""
        log_odds = np.log(node.prob / (1 - node.prob))
        delta = np.log(self.prob_hit/(1-self.prob_hit)) if occupancy else np.log(self.prob_miss/(1-self.prob_miss))
        new_log_odds = log_odds + delta
        node.prob = 1 / (1 + np.exp(-new_log_odds))
        node.prob = np.clip(node.prob, self.clamp_min, self.clamp_max)

    def insert_point(self, point):
        """
        插入一个占据点（自动细分节点）\n
        :param point: 三维坐标点 (x,y,z)
        """
        # 如果点在节点内部则插入
        global OctoMap_dict, OPG_dict
        xyz_d = point - self.root.center
        if np.max(np.abs(xyz_d)) <= self.root.size/2:
            stack = [(self.root, 0)]  # (节点, 当前深度)
            while stack:
                node, depth = stack.pop()
            
                # 如果达到最大深度或满足分辨率要求
                if depth >= self.max_depth or node.size <= self.resolution:
                    self.update_node(node, True)
                    node.last_update = time.time()
                    continue
                
                # 如果当前节点是叶子节点且需要细分
                if node.is_leaf:
                    node.subdivide()
                
                # 找到点所在的子节点索引（0-7）
                idx = 0
                rel_pos = point - node.center
                idx = compute_child_index(rel_pos)
                child = node.children[idx]
                if child:
                    stack.append((child, depth+1))
        # 如果节点不在内部，则创建新的八叉树地图，并插入点
        else:
            Direction, halfsize= np.array([0,0,0]), self.root.size*0.5
            # 判断点应该在哪个方向上的八叉树地图中
            mask = np.where(np.abs(xyz_d) > halfsize)
            idx = mask[0]
            Direction[idx] = ((xyz_d[idx] - np.sign(xyz_d[idx])*halfsize)/self.root.size + np.sign(xyz_d[idx])).astype(int)
            # 根据方向确定新地图的中心坐标
            NewOctomapCenter = self.root.center + Direction * self.root.size
            NewOctomapOPG = self.OPG + Direction
            # 判断新地图是否已经存在，如果存在则插入点，否则创建新地图并插入点
            if _is_OctoMap_Center_created(NewOctomapOPG):
                octomap = OPG_Positioning(NewOctomapOPG)
                if octomap != None:
                    octomap.insert_point(point)
            else:
                Octomap_creator(NewOctomapCenter, self.root.size, self.resolution, self.max_depth)
                octomap = OPG_Positioning(NewOctomapOPG)
                if octomap != None:
                    octomap.insert_point(point)
    
    def fast_insert_point(self, point, xyz_d, node_judgment):
        """SIMD加速插入"""
        # 筛选xyz_d处于当前八叉树地图范围内的点
        mask = np.max(np.abs(xyz_d), axis=1) <=self.root.size
        _batch_insert_points, _batch_node_judgment= point[mask], node_judgment[mask]
        _other_insert_points, _other_node_judgment= point[~mask], node_judgment[~mask]
        _other_points_opg = np.zeros((len(point[~mask]),3))
        for i in range(len(point[~mask])):
            _other_points_opg[i] = _calc_opg(point[~mask][i])
        if _batch_insert_points.shape[0] !=0:
            self._batch_insert(_batch_insert_points, _batch_node_judgment)
        if _other_insert_points.shape[0] !=0:
            self._other_insert(_other_insert_points, _other_points_opg, _other_node_judgment)
    
    def _other_insert(self, points, opgs, node_judgment):
        unique_opgs = np.unique(opgs, axis=0)
        valid_mask = np.apply_along_axis(_is_OctoMap_Center_created, 1, unique_opgs*self.root.size)
        # 找到还没有创建的opg坐标下的八叉树地图，然后创建
        noexist_opgs = unique_opgs[~valid_mask]
        for opg in noexist_opgs:
            Octomap_creator(opg*self.root.size, self.root.size, self.resolution, self.max_depth)
        for opg in unique_opgs:
            mask = np.all(opg == opgs, axis=1)
            octomap = OPG_Positioning(opg)
            octomap._batch_insert(points[mask], node_judgment[mask])

    def _batch_insert(self, points, judgment):
        """JIT加速批量插入"""
        for i in prange(len(points)):
            point = points[i]
            update = judgment[i]
            # 保留原始插入逻辑的JIT实现
            stack = [(self.root, 0)]  # (节点, 当前深度)
            while stack:
                node, depth = stack.pop()
            
                # 如果达到最大深度或满足分辨率要求
                if depth >= self.max_depth or node.size <= self.resolution:
                    self.update_node(node, update)
                    node.last_update = time.time()
                    continue
                
                # 如果当前节点是叶子节点且需要细分
                if node.is_leaf:
                    node.subdivide()
                
                # 找到点所在的子节点索引（0-7）
                # idx = 0
                rel_pos = point - node.center
                idx = compute_child_index(rel_pos)
                child = node.children[idx]
                if child:
                    stack.append((child, depth+1))

    def query_point(self, point):
        """
        查询点的占据概率\n
        :param point: 要查询的三维坐标
        :return: 占据概率（0-1）
        """
        node = self.root
        point = np.array(point)
        # 如果节点不存在于当前八叉树地图中，则返回None
        if np.max(np.abs(point - node.center)) <= node.size / 2:
            while not node.is_leaf:
            # 计算属于哪个子节点
                idx = 0
                rel_pos = point - node.center
                for i in range(3):
                    if rel_pos[i] > 0:
                        idx |= 1 << i
                next_node = node.children[idx]
                if not next_node:
                    break  # 如果子节点不存在则返回当前节点概率
                node = next_node
            return node.prob
        else:
            return 0.5
            
    def batch_query_points(self, points):
        """批量查询多个点的占据概率"""
        probabilities = np.empty(len(points), dtype=np.float32)
        for i, point in enumerate(points):
            probabilities[i] = self.query_point(point)
        return probabilities
    
    def query_points(self, points):
        """
        查询多个点的占据概率\n
        :param points: 要查询的多个三维坐标点
        :return: 占据概率列表
        """
        return [self.query_point(point) for point in points]
        

    def visualize(self, threshold=0.7):
        """可视化占据节点（简单文本输出）"""
        q = deque([self.root])
        
        while q:
            node = q.popleft()
            if node.prob >= threshold:
                print(f"Node at {node.center}, size {node.size}, prob {node.prob:.2f}")
            if not node.is_leaf:
                q.extend([child for child in node.children if child])
    
    def traverse(self):
        """ 返回所有节点的概率和它们的位置 """
        probabilities = []
        positions = []
        
        stack = [self.root]
        while stack:
            current_node = stack.pop()
            
            if current_node.size == self.resolution:
                probabilities.append(current_node.prob)
                positions.append(current_node.center)
                
                # 检查是否需要应用衰减规则
                elapsed_time = time.time() - current_node.last_update
                if (elapsed_time > 5.0) & (current_node.prob > 0.5) & (current_node.prob <=0.85):  # 如果超过5秒
                    current_node.prob = 0.5
                    current_node.last_update = time.time()
                elif (elapsed_time > 5.0) & (current_node.prob > 0.85):
                    current_node.prob = current_node.prob - self.prob_miss
                    current_node.last_update = time.time()
            else:
                for child in reversed(current_node.children):
                    if child is not None:
                        stack.append(child)
        
        return np.array(positions, dtype=np.float64), np.array(probabilities, dtype=np.float64)
    
    def visualize_3d(self, time_perf_counter):
        """
        可视化八叉树占用情况（三维）
        """
        # 创建绘图对象
        time_a = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = self.root.center

        # 颜色映射：概率 -> 蓝色深度
        def get_color(prob):
            if node.prob <= 0.5:
                color_code = (1, 0, 0, 1)  # 不显示颜色
                return color_code
            else:
                # 将概率映射到蓝色强度（0-255）
                blue_intensity = int(255 * (prob - 0.5)/0.5)
                color_code = f'#{blue_intensity:02x}00ff'
                return color_code

        # 广度优先遍历所有节点
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            
            # 如果不是叶子节点，将子节点加入队列
            if not node.is_leaf:
                for child in node.children:
                    if child is not None:
                        queue.append(child)
            else:
                # 如果是叶子结点，且prob> 0.5，则绘制当前节点的立方体
                if node.prob > 0.5:
                    self.add_cube_to_plot(ax, node.center, node.size, get_color(node.prob))
        
        # 设置坐标轴范围
        ax.set_xlim([self.root.center[0]-self.root.size/2, self.root.center[0]+self.root.size/2])
        ax.set_ylim([self.root.center[1]+self.root.size/2, self.root.center[1]-self.root.size/2])
        ax.set_zlim([self.root.center[2]+self.root.size/2, self.root.center[2]-self.root.size/2])

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Origin:{origin}, Octomap at time: {time_perf_counter}')
        print(time.time()-time_a)
        plt.show()

    def add_cube_to_plot(self, ax, center, size, color):
        """
        在三维坐标系中绘制一个立方体\n
        :param ax: 绘图对象
        :param center: 立方体中心坐标 (x, y, z)
        :param size: 立方体边长
        :param color: 颜色 (r, g, b)
        """
        # 计算立方体顶点坐标
        half_size = size / 2
        # 将Z轴翻转，使得坐标系与地图一致
        vertices = np.array([
            [center[0] - half_size, center[1] - half_size, center[2] - half_size],
            [center[0] + half_size, center[1] - half_size, center[2] - half_size],
            [center[0] + half_size, center[1] + half_size, center[2] - half_size],
            [center[0] - half_size, center[1] + half_size, center[2] - half_size],
            [center[0] - half_size, center[1] - half_size, center[2] + half_size],
            [center[0] + half_size, center[1] - half_size, center[2] + half_size],
            [center[0] + half_size, center[1] + half_size, center[2] + half_size],
            [center[0] - half_size, center[1] + half_size, center[2] + half_size]
        ])

        # 定义正方体的六个面，每个面由四个顶点组成，按顺序排列
        faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]]   # 后面
                ]
        # 创建 Poly3DCollection 对象
        poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.15, edgecolors=color, alpha=0.5)

        # 将 Poly3DCollection 对象添加到坐标轴中
        ax.add_collection3d(poly3d)
    
def Octomap_creator(CenterPoint: np.array, map_size: int, resolution, max_depth):
    """创建八叉树地图，并写入列表\n
    :param CenterPoint: 地图中心坐标 [x,y,z]，单位为m
    :param map_size: 地图尺寸（立方体边长），单位为m
    :param resolution: 最小体素分辨率，单位为m
    :param max_depth: 最大树深度，无量纲，最大为16
    """
    global OctoMap_dict, OPG_dict, OctoMap_lock
    OPG, _ = Octree_Positioning_Gird_Calculator(CenterPoint, map_size)
    if _:
        octomap = OctoMap(origin=CenterPoint, 
                         Octree_Positioning_Grid=OPG,
                         map_size=map_size,
                         resolution=resolution,
                         max_depth=max_depth)
        opg_key = tuple((int(OPG[0]), int(OPG[1]), int(OPG[2])))
        with OctoMap_lock:
            OctoMap_dict[opg_key] = octomap
            OPG_dict[opg_key] = time.time()
        print(f"Created New Octomap, Center at {CenterPoint}, size {map_size}, State located at {OPG}")


def Octree_Positioning_Gird_Calculator(CenterPoint: np.array, map_size: int):
    """计算八叉树地图位于总地图网格中的位置\n
    当返回的判定值不为1时，说明该中心为假，不生成新的八叉树地图\n
    :param CenterPoint: 地图中心坐标 [x,y,z]，单位为m
    :param map_size: 地图尺寸（立方体边长），单位为m
    :param resolution: 最小体素分辨率，单位为m
    """
    OPG, limit= np.array([0,0,0]), np.array([0,0,0])
    Octree_root = np.array([0,0,0])
    _true_Center = True
    for i in range(3):
        OPG[i] = (CenterPoint[i] - Octree_root[i]) / map_size
        limit[i] = (CenterPoint[i] - Octree_root[i]) % map_size
        if limit[i] > 1e-2:
            _true_Center = False
            break
    return OPG, _true_Center

def _is_OctoMap_Center_created(CenterPoint: np.array):
    """查找现有的总地图网格中是否存在有该中心的八叉树地图\n
    :param CenterPoint: 地图中心坐标 [x,y,z]，单位为m
    """
    global OctoMap_dict
    opg = _calc_opg(CenterPoint)
    opg = tuple((int(opg[0]),int(opg[1]),int(opg[2])))
    target_map = OctoMap_dict.get(opg, None)
    if target_map != None:
        return True
    else:
        return False
    

def Center_to_OPG(CenterPoint: np.array, map_size: int):
    """将地图中心坐标转换为OPG坐标\n
    :param CenterPoint: 地图中心坐标 [x,y,z]，单位为m
    :param map_size: 地图尺寸（立方体边长），单位为m
    """
    global ORIGIN
    OPG = np.array([0,0,0])
    Octree_root = ORIGIN
    for i in range(3):
        OPG[i] = (CenterPoint[i] - Octree_root[i]) / map_size
    return OPG

def OPG_Positioning(OPG: np.array):
    """根据OPG坐标调用对应的八叉树地图\n
    :param OPG: 八叉树地图位于总地图网格中的位置[x,y,z]
    """
    global OctoMap_dict
    # 根据OPG坐标调用对应地图，存在则返回地图，不存在则返回None
    key = tuple(OPG)
    target_map = OctoMap_dict.get(key, None)
    if target_map:
        return target_map
    else:
        return None
    


def compute_child_index(rel_pos):
    return (rel_pos[0]>0) | ((rel_pos[1]>0)<<1) | ((rel_pos[2]>0)<<2)


key_type = types.UniTuple(int32, 3)  # 键是3个int32的元组
value_type = types.Tuple((float32, float32, float32, int64))  # 值是(float32, float32, float32, int32)
@jit(nopython=True, boundscheck=True,locals={
        'x': float32,
        'y': float32,
        'z': float32,
        'key': key_type,
        'i': numba.int64,
        'v': value_type
    })
def voxel_downsample_jit(points: types.float32[:,:], voxel_size: types.float32) -> types.float32[:,:]:
    """
    体素滤波，降低密集点云带来的庞大数据量影响运行速率\n
    """
    if len(points) == 0:
        return points
    voxel_dict = typed.Dict.empty(key_type, value_type)
    for i in prange(points.shape[0]):
        x, y, z = points[i]
        key = (int(x//voxel_size), int(y//voxel_size), int(z//voxel_size))
        if key not in voxel_dict:
            voxel_dict[key] = (np.float32(x), np.float32(y), np.float32(z), 1)
        else:
            voxel_dict[key] = (
                np.float32(voxel_dict[key][0] + x),
                np.float32(voxel_dict[key][1] + y),
                np.float32(voxel_dict[key][2] + z),
                voxel_dict[key][3] + 1
            )
    
    downsampled = np.empty((len(voxel_dict), 3), dtype=np.float32)
    for i, (k, v) in enumerate(voxel_dict.items()):
        downsampled[i] = (v[0]/v[3], v[1]/v[3], v[2]/v[3])
    return downsampled

def _global_val_set(map_size=None, origin=None):
    global MAP_SIZE, ORIGIN
    MAP_SIZE = MAP_SIZE if map_size is None else map_size
    ORIGIN = ORIGIN if origin is None else origin
    print(f'Global map size set to {MAP_SIZE}, origin set to {ORIGIN}')

@jit(nopython=True)
def _calc_opg(position):
    """计算输入点所在OPG网格坐标"""
    global MAP_SIZE, ORIGIN
    opg = np.zeros(3, dtype=np.int32)
    for i in range(3):
        delta = position[i] - ORIGIN[i]
        if abs(delta) > MAP_SIZE/2:
            opg[i] = int((delta - np.sign(delta)*MAP_SIZE/2) / MAP_SIZE) + np.sign(delta)
    return opg

@jit(nopython=True)
def _calc_opgs(position: np.array):
    global MAP_SIZE, ORIGIN
    opg = np.zeros((len(position), 3), dtype=np.int32)
    for j in range(len(position)):
        for i in range(3):
            delta = position[j, i] - ORIGIN[i]
            if abs(delta) > MAP_SIZE/2:
                opg[j, i] = int((delta - np.sign(delta)*MAP_SIZE/2) / MAP_SIZE) + np.sign(delta)
    return opg

def remove_nearby_points(A, B, radius=1.0):
    """
    删除A中与B任意点距离<radius的点

    Args:
        A: (m,3) 待处理点集
        B: (n,3) 参考点集
        radius: 邻近阈值

    Returns:
        A_filtered: 过滤后的点集

    """
    if len(B) == 0 or len(A) == 0:
        return A
    
    # 构建B的KDTree加速距离查询
    tree = cKDTree(B)
    
    # 查询A中每个点到B的最短距离
    dists, _ = tree.query(A, k=1, workers=-1)  # 获取最近距离
    
    # 创建过滤掩码
    mask = dists >= radius
    print(A, B,A.shape, B.shape, np.sum(mask), len(A), np.max(A[mask], axis=0), np.min(A[mask], axis=0))
    return A[mask]


@njit(parallel=True)
def process_opg_points_jit(points, indices, probabilities, query_output, query_range):
    """JIT加速处理OPG点云并更新query_output"""
    valid_mask = probabilities != 0.5
    for i in prange(len(points)):
        if valid_mask[i]:
            idx = indices[i].astype(np.int32)
            query_output[idx[0], idx[1], idx[2]] = probabilities[i]
    return query_output

@njit
def check_valid_opgs(opgs, map_size, opg_dict_keys):
    """JIT加速检查有效OPG"""
    result = np.zeros(len(opgs), dtype=np.bool_)
    for i in range(len(opgs)):
        opg = opgs[i]
        key = (opg[0], opg[1], opg[2])
        if key in opg_dict_keys:
            result[i] = True
    return result

def warmup_numba_functions():
    """
    预热Numba函数，通过预先编译减少第一次调用时的延迟
    这个函数应该在程序启动时调用一次
    """
    print("正在预热Numba函数...")
    start_time = time.time()
    
    # 预热voxel_downsample_jit函数
    test_points = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [5.0, 6.0, 7.0]], dtype=np.float32)
    result = voxel_downsample_jit(test_points, np.float32(0.5))
    
    # 预热process_opg_points_jit函数
    test_output = np.full((3, 3, 3), 0.5, dtype=np.float32)
    test_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    test_probs = np.array([0.2, 0.5, 0.8])
    process_opg_points_jit(test_points[:3], test_indices, test_probs, test_output, 1)
    
    # 预热使用prange的其他函数
    # 如果有其他使用@jit或@njit的函数，也可以在这里预热
    
    end_time = time.time()
    print(f"Numba函数预热完成，耗时: {end_time - start_time:.4f}秒")

@jit(nopython=True)
def _direction_vector(az, el):
    """计算方向向量"""
    x = np.cos(el) * np.sin(az)
    y = np.sin(el)
    z = np.cos(el) * np.cos(az)
    return np.array([x, y, z], dtype=np.float32)

# def _raycast_batch_jit(origin, directions, max_range):
#         """
#         JIT加速的批量射线探测
#         """
#         n_az, n_el, _ = directions.shape
#         results = np.zeros((n_az, n_el), dtype=np.float32)
#         step_size = 0.5
#         total_distance = 0
#         while total_distance < max_range:
#             total_distance += step_size
#             if total_distance == 0:
#                 current_pos = origin.copy()
#                 opg = _calc_opg(current_pos)
#                 octomap = OPG_Positioning(opg)
#             else:
#                 current_pos = origin.copy() + total_distance*directions
#                     opg = _calc_opg(current_pos)
#                     octomap = OPG_Positioning(opg)
#                     if octomap is None:
#                         results[i, j] = max_range
#                         break
#                     else:
#                         prob = octomap.query_point(current_pos)
#                         if prob > 0.5:
#                             results[i, j] = total_distance
#                             break
#                         # 移动到下一个位置
#                         else:
#                             current_pos += direction * step_size
#                             total_distance += step_size
#                 else:
#                     results[i, j] = max_range
        
#         return results

# 使用示例
if __name__ == "__main__":
    # 预热Numba函数，减少第一次调用时的延迟
    warmup_numba_functions()
    
    # 创建八叉树地图
    OCL = OctoMaps()
    print(len(OCL.OPG_dict))
    # 插入一些占据点
    points = np.array([
        [0, 0, -8],
        [0, 0, 17],
        [0, 0, -17],
        [0, 17, 0],
        [-17, 0, -8],
        [17, 0, -8],
        [17,17, -8],
        [1.2, 0.5, 0.8],
        [-5,-2,7.03],
        [-5,-2.1,7.1],
        [-5,-2.024,7.02],
        [-5,-2.024,7.02],
        [16,16,16],
        [-16,-16,-16],
        [50,10,30],
        [32,32,32],
        [5,3,2]
    ])
    points_list = np.array([
        [0, 0, -8],
        [0, 0, 17],
        [0, 0, -17],
        [0, 17, 0],
        [-17, 0, -8],
        [17, 0, -8],
        [17,17, -8],
        [1.2, 0.5, 0.8],
        [-5,-2,7.03],
        [-5,-2.1,7.1],
        [-5,-2.024,7.02],
        [-5,-2.024,7.02],
        [16,16,16],
        [-16,-16,-16],
        [50,10,30],
        [32,32,32],
        [5,3,2]
    ], dtype=np.float32)
    for i in range(10):
        points_list = np.vstack((points_list,points_list))
    # octomap.insert_point(np.array([20,20,20]))
    # print(len(points_list[:,0]))
    # for p in points:
    #     print(p)
    #     time_a = time.time()
    #     octomap.insert_point(p)
    #     time_b = time.time()
    #     print(time_b- time_a)
    for i in range(2):
        time_a = time.time()
        OCL.Fast_insert_point(np.array([4,5,2]), points_list.copy(), points_list+1)
        time_b = time.time()
        print(time_b-time_a)
    for i in range(2):
        time_a = time.time()
        OCL.Fast_insert_point(np.array([-5,0,-47.98]), 50*np.random.random((1024,3)).astype(np.float32), points_list+1)
        time_b = time.time()
        print(time_b-time_a)
    print(OCL.OctoMap_dict)
    octomap = OCL.OctoMap_dict[(0, 0, 0)]
    for j in range(10):
        total_time = 0
        for i in range(4913):
            timea = time.time()
            p = octomap.query_point(np.array([i,i*2,2*i-1]))
            timeb = time.time()
            total_time += timeb - timea
        print(f"Total query time: {total_time:.7f} seconds, average_time: {total_time/4913:.7f} seconds")
    for j in range(100):
        timea = time.time()
        a = OCL.circle_query_points(np.array([j,j*0.2,np.sin(j/5)], dtype=np.float32), 30, 15, 10)
        b = octomap.query_point(np.array([j,j*0.2,np.sin(j/5)], dtype=np.float32))
        timeb = time.time()
        print(f"{j},query points:{a.shape}, time: {timeb-timea:.7f} seconds, {np.max(a[:,3]) if a.size>0 else a.shape}, {np.min(a[:,3]) if a.size>0 else a.shape}")
        print(b)