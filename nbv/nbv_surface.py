import numpy as np

from nbv_utils import as_points_array


def estimate_voxel_size_from_points(points):
    """
    Estimate voxel size from voxel center coordinates.

    This assumes voxel centers lie on a regular grid.
    """

    points = as_points_array(points)

    if len(points) < 2:
        return 0.01

    diffs = []

    for axis in range(3):
        vals = np.unique(np.round(points[:, axis], 8))
        vals = np.sort(vals)

        if len(vals) > 1:
            d = np.diff(vals)
            d = d[d > 1e-9]

            if len(d) > 0:
                diffs.append(np.min(d))

    if len(diffs) == 0:
        return 0.01

    return float(np.min(diffs))


def voxel_centers_to_grid_keys(points, voxel_size, origin=None):
    """
    Convert voxel center positions to integer grid keys.
    """

    points = as_points_array(points)

    if origin is None:
        origin = points.min(axis=0) if len(points) > 0 else np.zeros(3)

    keys = np.rint((points - origin) / voxel_size).astype(np.int64)

    return keys, origin


def extract_surface_voxels(points, voxel_size=None, origin=None):
    """
    Extract surface voxels from a voxel-center point cloud.

    A voxel is considered a surface voxel if at least one of its
    6-connected neighbors is missing.

    This works for:
        - occupied voxels
        - unknown voxels

    For unknown voxels, this gives the exposed boundary of the unknown region.
    """

    points = as_points_array(points)

    if len(points) == 0:
        return points

    if voxel_size is None:
        voxel_size = estimate_voxel_size_from_points(points)

    keys, origin = voxel_centers_to_grid_keys(
        points=points,
        voxel_size=voxel_size,
        origin=origin,
    )

    key_set = set(map(tuple, keys))

    neighbor_offsets = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.int64)

    surface_mask = np.zeros(len(points), dtype=bool)

    for i, key in enumerate(keys):
        for offset in neighbor_offsets:
            neighbor_key = tuple(key + offset)

            if neighbor_key not in key_set:
                surface_mask[i] = True
                break

    return points[surface_mask]
