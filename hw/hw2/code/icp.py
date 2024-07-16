import open3d as o3d
import numpy as np
import copy


source = o3d.io.read_point_cloud("data/bun000.ply")
target = o3d.io.read_point_cloud("data/bun045.ply")

def draw_registration_result(source, target, transformation):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    # Set new color
    source_copy.paint_uniform_color([0, 0.75, 0.75])
    target_copy.paint_uniform_color([1, 0.75, 0])
    source_copy.transform(transformation)
    o3d.visualization.draw_geometries([source_copy, target_copy])

threshold = 0.5
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], 
                            [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(source, target, trans_init)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))
print(f"Transformation is:{reg_p2p.transformation}")
draw_registration_result(source, target, reg_p2p.transformation)