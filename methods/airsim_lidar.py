import numpy as np
from numba import jit
import time

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

# 测试函数
def test_voxel_filter():
    """
    测试体素滤波函数
    """
    # 创建示例点云数据
    np.random.seed(42)
    n_points = 50000  # 增加点数测试性能
    points = np.random.uniform(-10, 10, size=(n_points, 3))
    
    print(f"原始点云数量: {len(points)}")
    print(f"体素大小: 0.5m")
    
    # 应用体素滤波
    original_points = points.copy()
    for i in range(5):
        print(f"正在进行第{i+1}次滤波...")
        timea = time.time()
        filtered_points = voxel_filter(points, voxel_size=0.5)
        timeb = time.time()
        print(f"滤波耗时: {timeb - timea:.4f}秒")
        print(f"滤波后点数: {len(filtered_points)}")
    
    final_filtered = filtered_points
    
    print(f"\n最终滤波后点云数量: {len(final_filtered)}")
    print(f"压缩比: {len(final_filtered)/len(original_points):.3f}")
    print(f"保留率: {len(final_filtered)/len(original_points)*100:.2f}%")
    
    return original_points, final_filtered

# 性能对比测试
def performance_test():
    """
    性能对比测试
    """
    print("=== 性能对比测试 ===")
    
    # 生成大规模测试数据
    np.random.seed(42)
    sizes = [10000, 50000, 100000]
    
    for size in sizes:
        print(f"\n测试点云大小: {size}")
        points = np.random.uniform(-20, 20, size=(size, 3))
        
        # 测试优化版本
        start_time = time.time()
        result_opt = get_unique_voxels_fast(points.astype(np.float64), 0.5)
        opt_time = time.time() - start_time
        
        print(f"优化版本耗时: {opt_time:.4f}s")
        print(f"优化版本结果点数: {len(result_opt)}")

if __name__ == "__main__":
    test_voxel_filter()
    print("\n" + "="*50)
    performance_test()
