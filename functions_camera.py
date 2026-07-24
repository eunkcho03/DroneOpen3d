import numpy as np

DEPTH_MIN_VALID = 0.0
DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01

def make_valid_depth_mask(depth, far):
    depth = np.asarray(depth)
    return (
        np.isfinite(depth)
        & (depth > DEPTH_MIN_VALID)
        & (depth != DEPTH_INVALID_VALUE)
        & (depth < far - FAR_PLANE_MARGIN)
    )

def camera_intrinsics(width, height, fov_degrees):
    fy = height / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3)

    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])

def depth_to_camera_points(depth, fov_degrees, far):
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
    valid_mask = make_valid_depth_mask(depth, far)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy

    points_camera = np.column_stack((x, y, z))
    return points_camera

def camera_to_world(points_camera, position, quaternion):
    points_camera = np.asarray(points_camera, dtype=float)
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return points_camera @ R.T + position


def world_to_camera(points_world, position, quaternion):
    points_world = np.asarray(points_world, dtype=float)
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return (points_world - position) @ R

def filter_by_height(depth, fov_degrees, far, position, quaternion, floor_height, height_threshold):
    points_camera = depth_to_camera_points(depth, fov_degrees, far)
    points_world = camera_to_world(points_camera, position, quaternion)
    min_height = floor_height + height_threshold 
    return points_world[points_world[:, 1] > min_height ]
