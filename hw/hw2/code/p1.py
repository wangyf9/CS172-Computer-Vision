# MAC/Windows：open3d：pip install open3d==0.15.1
import open3d as o3d

# Load point cloud
pcd1 = o3d.io.read_point_cloud("data/bun000.ply")
pcd2 = o3d.io.read_point_cloud("data/bun090.ply")
pcd3 = o3d.io.read_point_cloud("data/bun180.ply")
pcd4 = o3d.io.read_point_cloud("data/bun270.ply")

# Visualize point cloud

# Together
o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4])

# Solo
o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd2])
o3d.visualization.draw_geometries([pcd3])
o3d.visualization.draw_geometries([pcd4])