import numpy as np


def normalize_vector(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < eps:
        return np.zeros_like(v)

    return v / norm


def rotation_matrix_to_quaternion(R):
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)

    return q


def look_at_quaternion(
    camera_position,
    target_position,
    world_up=np.array([0.0, 1.0, 0.0]),
):
    """
    Compute quaternion [x, y, z, w] so that the camera faces the target.

    Assumption:
        - Unity-style coordinate system
        - Camera/object forward direction is local +Z
        - Y is world up
    """

    camera_position = np.asarray(camera_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = normalize_vector(target_position - camera_position)

    if np.linalg.norm(forward) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])

    right = np.cross(world_up, forward)

    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)

    right = normalize_vector(right)
    up = normalize_vector(np.cross(forward, right))

    R = np.column_stack((right, up, forward))

    return rotation_matrix_to_quaternion(R)


def as_points_array(points):
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return points.reshape(0, 3)

    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(f"Expected a 3D point, got shape {points.shape}")
        return points.reshape(1, 3)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) array, got shape {points.shape}")

    return points


# Backward-compatible private alias, in case older code imports it.
_as_points_array = as_points_array
