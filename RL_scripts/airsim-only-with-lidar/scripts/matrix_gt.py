import numpy as np

def matrix_global_to_target(vector_target, vector_global, epsilon=1e-10):
    """
    将全局坐标系下的向量转换到目标坐标系下
    
    Args:
        vector_target: 目标方向向量 (无需单位化)
        vector_global: 全局坐标系下的向量
        epsilon: 数值稳定系数
    
    Returns:
        目标坐标系下的向量
    
    数学原理：
        构造从全局到目标坐标系的旋转矩阵R，满足：v_target = R @ v_global
        R的列向量是全局坐标系基向量在目标坐标系下的表示
    """
    # 输入校验
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_global = np.asarray(vector_global, dtype=np.float64)
    
    # 归一化目标方向向量（x轴）
    target_norm = np.linalg.norm(vector_target)
    if target_norm < epsilon:
        raise ValueError("目标方向向量模长过小，无法归一化")
    x_axis = vector_target / target_norm
    
    # 计算y轴（与全局z轴和x轴正交）
    z_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis = np.cross(z_global, x_axis)
    y_norm = np.linalg.norm(y_axis)
    
    # 处理x轴与全局z轴平行的情况
    if y_norm < epsilon:
        # 选择与x轴不共线的任意向量
        if abs(x_axis[0]) > epsilon or abs(x_axis[1]) > epsilon:
            y_axis = np.array([-x_axis[1], x_axis[0], 0], dtype=np.float64)
        else:
            y_axis = np.array([0, 1, 0], dtype=np.float64)
        # 施密特正交化
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < epsilon:
            raise ValueError("无法构造正交的y轴")
    
    y_axis /= y_norm
    
    # 计算z轴（完成右手系）
    z_axis = np.cross(x_axis, y_axis)  # 注意叉积顺序
    z_norm = np.linalg.norm(z_axis)
    if abs(z_norm - 1) > epsilon:
        raise ValueError("z轴模长异常")
    
    # 构造旋转矩阵：全局→目标
    # 矩阵的列是目标坐标系的基向量在全局坐标系下的表示
    R_global_to_target = np.column_stack([x_axis, y_axis, z_axis]).T
    
    return R_global_to_target @ vector_global

def matrix_target_to_global(vector_target, vector_target_frame, epsilon=1e-10):
    """
    将目标坐标系下的向量转换到全局坐标系下
    
    Args:
        vector_target: 目标方向向量 (无需单位化)
        vector_target_frame: 目标坐标系下的向量
        epsilon: 数值稳定系数
    
    Returns:
        全局坐标系下的向量
    """
    # 使用相同的基向量构造方法
    vector_target = np.asarray(vector_target, dtype=np.float64)
    vector_target_frame = np.asarray(vector_target_frame, dtype=np.float64)
    
    target_norm = np.linalg.norm(vector_target)
    if target_norm < epsilon:
        raise ValueError("目标方向向量模长过小，无法归一化")
    x_axis = vector_target / target_norm
    
    z_global = np.array([0, 0, 1], dtype=np.float64)
    y_axis = np.cross(z_global, x_axis)
    y_norm = np.linalg.norm(y_axis)
    
    if y_norm < epsilon:
        if abs(x_axis[0]) > epsilon or abs(x_axis[1]) > epsilon:
            y_axis = np.array([-x_axis[1], x_axis[0], 0], dtype=np.float64)
        else:
            y_axis = np.array([0, 1, 0], dtype=np.float64)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < epsilon:
            raise ValueError("无法构造正交的y轴")
    
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    
    # 构造旋转矩阵：目标→全局（这是原矩阵的逆，即转置）
    R_target_to_global = np.column_stack([x_axis, y_axis, z_axis])
    
    return R_target_to_global @ vector_target_frame


def test_original_functions():
    """测试原代码功能"""
    print("=== 测试原代码功能 ===")
    
    # 测试用例1：简单情况
    print("\n测试用例1：x轴在xy平面")
    target_dir = np.array([1, 0, 0])  # 全局x轴
    test_vector = np.array([1, 2, 3])
    
    # 构造坐标系
    x_axis = target_dir / np.linalg.norm(target_dir)
    z_global = np.array([0, 0, 1])
    y_axis = np.cross(z_global, x_axis)  # [0, 1, 0] × [1, 0, 0] = [0, 0, 1]? 等等，错了！
    # 应该是：z_global × x_axis = [0,0,1] × [1,0,0] = [0,1,0] ✓
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)  # [1,0,0] × [0,1,0] = [0,0,1]
    
    print(f"构造的坐标系基向量：")
    print(f"x_axis: {x_axis}")
    print(f"y_axis: {y_axis}")
    print(f"z_axis: {z_axis}")
    
    # 验证正交性
    print(f"\n验证正交性：")
    print(f"x·y = {np.dot(x_axis, y_axis):.2e}")
    print(f"x·z = {np.dot(x_axis, z_axis):.2e}")
    print(f"y·z = {np.dot(y_axis, z_axis):.2e}")
    
    # 验证y轴与全局z轴正交
    print(f"\n验证y轴与全局z轴正交：")
    print(f"y·z_global = {np.dot(y_axis, z_global):.2e}")
    
    # 测试用例2：特殊情况
    print("\n测试用例2：x轴与全局z轴平行")
    target_dir = np.array([0, 0, 1])
    
    # 使用原代码逻辑
    x_axis = target_dir / np.linalg.norm(target_dir)
    z_global = np.array([0, 0, 1])
    y_axis = np.cross(z_global, x_axis)
    print(f"初始y轴: {y_axis} (应为零向量)")
    
    # 原代码的处理逻辑
    if np.linalg.norm(y_axis) < 1e-10:
        if abs(x_axis[0]) > 1e-10 or abs(x_axis[1]) > 1e-10:
            y_axis = np.array([-x_axis[1], x_axis[0], 0])
        else:
            y_axis = np.array([0, 1, 0])
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis)
    
    z_axis = np.cross(x_axis, y_axis)
    
    print(f"处理后坐标系：")
    print(f"x_axis: {x_axis}")
    print(f"y_axis: {y_axis}")
    print(f"z_axis: {z_axis}")
    
    # 验证这个坐标系是否满足要求
    print(f"\n验证：")
    print(f"y轴与全局z轴正交？ y·z_global = {np.dot(y_axis, z_global):.2e}")
    print(f"右手系？ det([x,y,z]) = {np.linalg.det(np.column_stack([x_axis, y_axis, z_axis])):.2f}")

def test_transforms():
    """测试坐标变换"""
    print("\n=== 测试坐标变换 ===")
    
    # 使用原函数（假设已定义）
    target_dir = np.array([1, 2, -2])  # 在xy平面
    test_vector = np.array([1, 0, 0])
    
    print(f"目标方向: {target_dir}")
    print(f"测试向量: {test_vector}")
    
    # 手动计算
    x_axis = target_dir / np.linalg.norm(target_dir)
    z_global = np.array([0, 0, 1])
    y_axis = np.cross(z_global, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 全局→目标
    v_target_manual = R.T @ test_vector
    print(f"\n手动计算全局→目标: {v_target_manual}")
    
    # 目标→全局
    v_global_manual = R @ v_target_manual
    print(f"手动计算目标→全局: {v_global_manual}")
    print(f"恢复误差: {np.linalg.norm(v_global_manual - test_vector):.2e}")
    
    # 解释这个变换的意义
    print(f"\n解释：")
    print(f"在目标坐标系中，x分量 = 测试向量在目标方向上的投影")
    print(f"x分量计算: {np.dot(test_vector, x_axis):.3f}")
    print(f"手动计算x分量: {v_target_manual[0]:.3f}")

if __name__ == "__main__":
    velocity_vector = np.array([1, 0, 1])
    target_direction = np.array([100, 1, -1])
    transformed_vector = matrix_global_to_target(target_direction, velocity_vector)
    print("Transformed Vector (Global to Target):", transformed_vector)
    recovered_vector = matrix_target_to_global(target_direction, transformed_vector)
    print("Recovered Vector (Target to Global):", recovered_vector)