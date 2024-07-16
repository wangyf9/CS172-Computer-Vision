import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm
from scipy.spatial import KDTree

def nearest_neighbor(source, target):
    # Trans to numpy
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Build kd-tree to accelerate
    tree = KDTree(target_points)

    # Get index
    distances, indices = tree.query(source_points)
    return indices, distances

def compute_transformation(source, target, indices):
    # Trans to numpy
    P = np.asarray(source.points)
    Q = np.asarray(target.points)

    # Get the matched target points 
    matched_target_points = Q[indices]

    # Get the mean
    p = np.mean(P, axis=0)
    q = np.mean(matched_target_points, axis=0)

    # Change the axis position
    P_new = P - p
    Q_new = matched_target_points - q

    # Get H
    H = np.dot(P_new.T, Q_new)
    # print(H.shape)

    # Decomposition H
    U, Sigma, V_T = np.linalg.svd(H)

    # Calculate the initial R
    R = np.dot(V_T.T, U.T)

    # Amend the value of R because sometimes R represents a reflection
    det_R = np.linalg.det(R)
    amend_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, det_R]
    ])

    # Get R
    R = np.dot(np.dot(V_T.T, amend_matrix), U.T)

    # Get t
    t = q.T - np.dot(R, p.T)

    # Construct T
    transformation = np.identity(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation

def ransac_transformation(source, target, indices, num_samples=50, threshold=0.06, max_trials=200):
    best_inliers = 0
    best_transformation = None

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    for _ in range(max_trials):
        # Every time random choose num_samples points
        sample_indices = np.random.choice(len(indices), size=num_samples, replace=False)
        sampled_source = source_points[sample_indices]
        sampled_target = target_points[indices[sample_indices]]

        # Transform data type, satisfying it is a point cloud
        transformation = compute_transformation(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_source)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_target)),
            np.arange(num_samples)
        )

        transformed_source = (np.dot(source_points, transformation[:3, :3].T) + transformation[:3, 3]).reshape(-1, 3)
        distances = np.linalg.norm(transformed_source - target_points[indices], axis=1)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_transformation = transformation

    return best_transformation

def draw_registration_result(source, target, transformation):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    # Set new color
    source_copy.paint_uniform_color([0, 0.75, 0.75])
    target_copy.paint_uniform_color([1, 0.75, 0])
    source_copy.transform(transformation)
    o3d.visualization.draw_geometries([source_copy, target_copy])

def icp_ransac(source, target, init_transformation, threshold, max_iterations, subset_size, start, end):
    source_copy = copy.deepcopy(source)
    transformation = init_transformation
    prev_error = float('inf')

    pbar = tqdm(range(max_iterations), desc="ICP Iteration")
    for i in pbar:

        # Randomly subset because two point clouds doesnot have the same number of points
        # Also, in this way, it can accelerate the speed of iteration
        num = len(source_copy.points)
        subset_indices = np.random.choice(num, size=subset_size, replace=False)
        source_subset = source_copy.select_by_index(subset_indices)
        
        # Find nearest points
        indices, distances = nearest_neighbor(source_subset, target)

        # Calculate error
        error = np.sum(distances ** 2)
        pbar.set_postfix(loss=error)

        # break condition
        if error < threshold:
            break

        # Find T
        best_transformation = ransac_transformation(source_subset, target, indices)
        source_copy.transform(best_transformation)

        R = best_transformation[:3, :3]
        t = best_transformation[:3, 3]

        delta_R = np.linalg.norm(R - transformation[:3, :3])
        delta_t = np.linalg.norm(t - transformation[:3, 3])
        
        # break condition
        if delta_R < threshold and delta_t < threshold:
            break

        # Get the accumulated transformation
        transformation = np.dot(best_transformation, transformation)
        prev_error = error
    print(f"Transformation for {start} and {end} is:{transformation}")
    print("error", error)
    draw_registration_result(source, target, transformation)

source1 = o3d.io.read_point_cloud("data/bun000.ply")
target1 = o3d.io.read_point_cloud("data/bun315.ply")

source2 = o3d.io.read_point_cloud("data/bun000.ply")
target2 = o3d.io.read_point_cloud("data/bun045.ply")

source3 = o3d.io.read_point_cloud("data/bun270.ply")
target3 = o3d.io.read_point_cloud("data/bun315.ply")

# Initial transformation matrix
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

threshold = 1e-4
max_iterations = 1000
subset_size = 2000

start_angle = [0, 0, 270]
end_angle = [315, 45, 315]
icp_ransac(source1, target1, trans_init, threshold, max_iterations, subset_size, start_angle[0], end_angle[0])
icp_ransac(source2, target2, trans_init, threshold, max_iterations, subset_size, start_angle[1], end_angle[1])
icp_ransac(source3, target3, trans_init, threshold, max_iterations, subset_size, start_angle[2], end_angle[2])
