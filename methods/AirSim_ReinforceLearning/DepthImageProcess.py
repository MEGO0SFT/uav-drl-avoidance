import numpy as np
import time
from . import AirSimProcessor as AP
from numba import njit, prange


# 最后更新时间 2025年11月19日
# 作者： 石辅天
# 最新功能：空闲点计算采集
__version__ = '1.4.0-stable'
__all__ = ['DepthImageProcessor', 'compute_aabb', 'aabb_points', 'is_point_in_frustum', 'perspective_projection', 'Frustum_Filtering']
# 深度相机分辨率与视场角
# 参数来源 Gemini 335
# CAMERA_RESOLUTION = (1280, 800)
# FOV_DEGREES = 90
MAX_DISTANCE = 20.0  # 最大探测距离
MIN_DISTANCE = 0.2   # 最小探测距离

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

def Frustum_Filtering(points_list, P_matrix):
    """
    计算输入的点集中属于期望视锥体内的点
    Args:
        points_list: 待滤波点集 nx3
        P_matrix: 视锥体矩阵 (OpenGL格式)
    Returns:
        mx3 滤波后点集
    """
    points = points_list.copy()
    points[:,2] = -points[:,2]
    ndc = is_point_in_frustum(points, P_matrix)
    mask = np.max(abs(ndc.T), axis=1) < 1.0
    return points_list[mask]


def perspective_projection(fov, aspect, n, f):
    """
    计算透视投影矩阵
    Args:
        fov_deg: 水平视场角(度)
        aspect: 宽高比(width / height)
        n: 近平面距离
        f: 远平面距离
    Returns:
        4x4 投影矩阵
    """
    fov_rad = fov
    t = np.tan(fov_rad / 2) * n  # 上边界
    b = -t                        # 下边界
    r = t * aspect                # 右边界
    l = -r                        # 左边界

    # 投影矩阵
    P = np.zeros((4, 4))
    P[0, 0] = 2 * n / (r - l)
    P[1, 1] = 2 * n / (t - b)
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -2 * f * n / (f - n)
    P[3, 2] = -1

    return P

def is_point_in_frustum(points_list, projection_matrix):
    """
    判断点集中的点是否在视锥体内
    Args:
        points_list: 3D 点 [x, y, z] Nx3
        projection_matrix: 4x4 投影矩阵
    Returns:
        points[mask]: 返还在视锥体内的点 
    """
    # 转换为齐次坐标
    one = np.ones((len(points_list),1), dtype=np.float64)
    points_hom = np.hstack((points_list, one))  # Nx4
    
    # 变换到裁剪空间
    clip_space = projection_matrix @ points_hom.T
    
    # 透视除法
    ndc = clip_space[:3] / (clip_space[3]+1e-6)
    
    # 检查 NDC 范围
    return ndc

class DepthImageProcessor:
    def __init__(self, CAMERA_RESOLUTION=(256,160), FOV_DEGREES=90.0):
        self.resolution = CAMERA_RESOLUTION
        self.fov = FOV_DEGREES
        self.vfov = 2*np.arctan(np.tan(np.deg2rad(self.fov/2))*self.resolution[1]/self.resolution[0])
        # 计算相机的内参矩阵
        self.fx = (self.resolution[0] / 2) / np.tan(np.deg2rad(self.fov/2))
        self.fy = (self.resolution[1] / 2) / np.tan(self.vfov/2)
        self.K = np.array([[self.fx, 0, self.resolution[0]/2],
                           [0, self.fy, self.resolution[1]/2],
                           [0, 0, 1]])
        self.Rf2g = np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]], dtype=np.float64)  # 前视相机3D坐标系->NED相对坐标系
        self.Rbo2g = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]], dtype=np.float64)  # 底视相机3D坐标系->NED相对坐标系
        self.Rba2g = np.array([[0, 0, -1],
                               [-1, 0, 0],
                               [0, 1, 0]], dtype=np.float64) # 后中相机3D坐标系->NED相对坐标系
        self.P = perspective_projection(self.vfov, self.resolution[0]/self.resolution[1],MIN_DISTANCE,MAX_DISTANCE)
        self.matrix = aabb_points(np.array([[-MAX_DISTANCE,-MAX_DISTANCE,0.5*MAX_DISTANCE],[MAX_DISTANCE, MAX_DISTANCE,0.5*MAX_DISTANCE]]), 1)
        self.free_points = Frustum_Filtering(self.matrix, self.P)
    
    def _pixel_to_point(self, pixel_x, pixel_y, depth, which_camara):
        global MAX_DISTANCE, MIN_DISTANCE
        valid_mask1, valid_mask2= depth <= MAX_DISTANCE, depth >=MIN_DISTANCE
        valid_mask = valid_mask1*valid_mask2
        
        x = (pixel_x - self.resolution[0]/2) / self.fx
        y = (pixel_y - self.resolution[1]/2) / self.fy
        
        denominator = np.sqrt(x**2 + y**2) + np.finfo(float).eps
        theta = np.arctan(denominator)
        xy_d = depth * np.sin(theta)

        # 计算相机坐标系下每一点对应的三维深度信息，并归纳进3xn的矩阵中已用于后续处理
        z = depth * np.cos(theta)
        x_cam = xy_d * x / denominator
        y_cam = xy_d * y / denominator
        xyz_depth = np.vstack((x_cam, y_cam, z))
        
        # 构建相机坐标系点，过滤无效点
        points = np.full((3, len(pixel_x)), np.nan)
        # 选择对应的相机坐标系转换矩阵，如果没有对应的矩阵，则返回空值
        if which_camara == "front_center":
            points[:, valid_mask] = self.Rf2g @ xyz_depth[:, valid_mask]
        elif which_camara == "bottom_center":
            points[:, valid_mask] = self.Rbo2g @ xyz_depth[:, valid_mask]
        elif which_camara == "back_center":
            points[:, valid_mask] = self.Rba2g @ xyz_depth[:, valid_mask]
        else:
            print(f"Invalid camara type: {which_camara}, returning Zero Matrix")
            return np.zeros((1,3), dtype=np.float64)
        return points.T

    def _point_to_global(self, depth_image, current_pos, eular_angles, which_camara):
        """
        将深度图像中的每个点转换到无人机体坐标系\n
        depth_image: 256x144,np.array
        current_pos : np.array
        eular_angles: np.array(roll, pitch, yaw)
        """
        pixel_x, pixel_y = np.meshgrid(np.arange(self.resolution[0]), 
                                      np.arange(self.resolution[1]), indexing='xy')
        pixel_x = pixel_x.ravel()
        pixel_y = pixel_y.ravel()
        depths = depth_image.ravel()
        # 批量转换到相机坐标系,根据which_camara选择不同的转换矩阵
        points_cam = self._pixel_to_point(pixel_x, pixel_y, depths, which_camara)
        
        # 过滤无效点
        valid_mask = ~np.isnan(points_cam).any(axis=1)
        if not np.any(valid_mask):
            return np.empty((0, 3)), np.empty((0, 3))
        points_cam_valid = free_points_downsample(points_cam[valid_mask], voxel=1)
        # 计算光线矢量方向，用于后续空闲空间计算
        Light_vector, norm = Light_Vectors_Calculation(points_cam_valid, self.free_points)
        free_points = _free_points_calculation(Light_vector, norm)
        free_points = free_points_downsample(free_points, voxel=1)
        # 将点从相机器坐标系转换到全局坐标系
        # 这里假设无人机的姿态和位置已经更新，且相机坐标系与无人机的坐标系一致
        if eular_angles.shape[0] == 4:
            points_cam_global = Rb2g_matrix(eular_angles, points_cam_valid) + current_pos
            free_points_global = Rb2g_matrix(eular_angles, free_points) + current_pos
        else:
            points_cam_global = AP.rotation_point(eular_angles, points_cam_valid) + current_pos
            free_points_global = AP.rotation_point(eular_angles, free_points) + current_pos
        return points_cam_global.astype(np.float64), free_points_global.astype(np.float64)

def compute_aabb(points):
    """计算轴对齐最小包围盒"""
    if len(points) != 0:
        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)
        return np.array([min_corner, max_corner], dtype=np.float64)
    else:
        return np.array([points, points])

def aabb_points(corners_array, pixel=1):
    """
    生成aabb内部点点集
    Args:
        corners_array: [最小点,最大点] 2x3
        pixel: 点集间隔,默认为1
    Returns:
        points_list: Nx3
    """
    if len(corners_array[0]) != 0:
        x = np.arange(corners_array[0,0], corners_array[1,0]+np.sign(corners_array[1,0])*pixel, pixel)
        y = np.arange(corners_array[0,1], corners_array[1,1]+np.sign(corners_array[1,1])*pixel, pixel)
        z = np.arange(corners_array[0,2], corners_array[1,2]+np.sign(corners_array[1,2])*pixel, pixel)
        xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return points.astype(np.float64)
    else:
        return np.empty((0,3))


def Light_Vectors_Calculation(xyz_depth, FREE_POINTS):
    if len(xyz_depth) == 0:
        # 如果没有检测到障碍点，说明前方探测矩阵范围内没有障碍物
        # 额外计算光线方向，提高无人机认知
        norm = np.linalg.norm(FREE_POINTS[::20], axis=1, keepdims=True)
        norm[norm == 0] = 1e-6
        Light_vector = FREE_POINTS[::10] / norm
        return Light_vector, norm
    norm = np.linalg.norm(xyz_depth[::20], axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    Light_vector = xyz_depth[::20] / norm
    return Light_vector, norm

def _free_points_calculation(vectors, norms, voxel_size=1.0):
    length = max(1, np.max(norms))
    pointlist = []
    current_length = voxel_size
    while length > current_length + voxel_size:
        mask = norms >current_length
        pointlist.append(vectors[mask[:,0], :] * current_length)
        if current_length < 5:
            current_length += voxel_size
        else:
            voxel_size += 1
            current_length += voxel_size
        if len(pointlist) > 0 and pointlist[-1].shape[0] == 0:
            break
    if len(pointlist) > 0:
        return np.vstack(pointlist)
    else:
        return np.empty((0,3))
        

def free_points_downsample(free_points, voxel=0.5):
    """
    降低空闲点云密度
    """
    if len(free_points) == 0:
        return free_points
    floored = np.floor_divide(free_points, voxel)                # 将空闲点集根据分辨率进行归类
    sample_mask = np.unique(floored, axis=0, return_index=True)  # 去重
    downsampled = free_points[sample_mask[1]]                    # 选择对应的点
    return downsampled


if __name__ == "__main__":
    T = DepthImageProcessor()
    points = np.array([
    [0, 0, -1],    
    [0, 0, -10],   
    [5, 0, -1],   
    [0, 3, -3],   
    [0, 0, -0.05], 
    [5, 1, -19]])
    output = is_point_in_frustum(points, T.P)
    print(output.T)
