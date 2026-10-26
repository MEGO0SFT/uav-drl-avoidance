import open3d as o3d
import numpy as np

print("正在测试 Open3D 窗口创建...")
vis = o3d.visualization.Visualizer()
success = vis.create_window(window_name="Test", width=640, height=480)

if not success:
    print("❌ 失败：create_window 返回 False。环境问题确认为图形界面缺失。")
    print("   请检查: 1. WSLg 是否启用 2. DISPLAY 变量 3. 显卡驱动")
else:
    print("✅ 窗口创建成功。")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    if opt is None:
        print("❌ 失败：get_render_option 返回 None。OpenGL 上下文缺失。")
    else:
        print("✅ 渲染选项获取成功。")
        opt.background_color = [0.5, 0.5, 0.5]
        print("准备显示窗口... (如果此时没弹窗，说明窗口创建了但不可见)")
        vis.run()
    
    vis.destroy_window()