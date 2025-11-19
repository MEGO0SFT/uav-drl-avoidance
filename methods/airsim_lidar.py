import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
uav = "UAV01"

client.takeoffAsync(vehicle_name=uav).join()
client.moveByVelocityAsync(1, 0, -0.5, 5, vehicle_name=uav)
for i in range(10):
    lidarData = client.getLidarData(lidar_name="Lidar1", vehicle_name=uav)
    print(len(lidarData.point_cloud), len(lidarData.point_cloud) / 3, len(lidarData.segmentation))
    print(lidarData.point_cloud[2::3])
    time.sleep(1)

# 结论：
# 1. point_cloud 是一个一维数组，包含所有点的 x, y, z 坐标，按顺序排列。
# 2. len(point_cloud) / 3 给出点的数量，因为每个点有三个坐标值。
# 3. segmentation 数组的长度与点的数量相同，表示每个点所属的类别或对象，可以用于语义地图构建。