import numpy as np

from nbv.nbv_utils import as_points_array, normalize_vector


def compute_camera_basis(
    camera_position,
    look_at,
    world_up=np.array([0.0, 1.0, 0.0]),
):
    """
    Create camera basis vectors from camera position and look-at point.

    Camera convention:
        x_cam = right
        y_cam = up
        z_cam = forward

    Forward is positive toward the object.
    """

    camera_position = np.asarray(camera_position, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = normalize_vector(look_at - camera_position)

    if np.linalg.norm(forward) < 1e-9:
        forward = np.array([0.0, 0.0, 1.0])

    right = np.cross(world_up, forward)

    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)

    right = normalize_vector(right)
    up = normalize_vector(np.cross(forward, right))

    return right, up, forward


def world_points_to_camera(points, camera_position, look_at):
    """
    Transform world points into candidate camera coordinates.
    """

    points = as_points_array(points)
    camera_position = np.asarray(camera_position, dtype=float)

    right, up, forward = compute_camera_basis(
        camera_position=camera_position,
        look_at=look_at,
    )

    rel = points - camera_position

    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward

    return np.column_stack((x_cam, y_cam, z_cam))


def project_points_to_image(
    points,
    camera_position,
    look_at,
    fov_degrees=58.0,
    image_width=160,
    image_height=160,
    aspect_ratio=16.0 / 9.0,
    near=1e-6,
):
    """
    Project world points into a simple pinhole candidate camera.

    Returns:
        pixel_u, pixel_v, depth_z, valid_mask
    """

    points = as_points_array(points)

    if len(points) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            np.empty(0, dtype=bool),
        )

    cam = world_points_to_camera(
        points=points,
        camera_position=camera_position,
        look_at=look_at,
    )

    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    vertical_half_fov = 0.5 * np.deg2rad(fov_degrees)
    horizontal_half_fov = np.arctan(np.tan(vertical_half_fov) * aspect_ratio)

    valid = (
        (z > near)
        & (np.abs(x / z) <= np.tan(horizontal_half_fov))
        & (np.abs(y / z) <= np.tan(vertical_half_fov))
    )

    u = np.empty(len(points), dtype=np.int64)
    v = np.empty(len(points), dtype=np.int64)

    u[:] = -1
    v[:] = -1

    if np.any(valid):
        x_norm = (x[valid] / z[valid]) / np.tan(horizontal_half_fov)
        y_norm = (y[valid] / z[valid]) / np.tan(vertical_half_fov)

        u_valid = ((x_norm + 1.0) * 0.5 * (image_width - 1)).astype(np.int64)
        v_valid = ((1.0 - (y_norm + 1.0) * 0.5) * (image_height - 1)).astype(np.int64)

        u_valid = np.clip(u_valid, 0, image_width - 1)
        v_valid = np.clip(v_valid, 0, image_height - 1)

        u[valid] = u_valid
        v[valid] = v_valid

    return u, v, z, valid


def compute_surface_projection_gain(
    occupied_surface_voxels,
    unknown_surface_voxels,
    camera_position,
    look_at,
    fov_degrees=58.0,
    image_width=160,
    image_height=160,
    aspect_ratio=16.0 / 9.0,
):
    """
    Compute NBV gain using visible surface voxels only.

    Unknown surface voxels and occupied surface voxels are projected into
    the candidate camera image. A z-buffer keeps only the nearest surface
    voxel per projected pixel.

    Score:
        visible_unknown_area * gain_ratio

    where:
        gain_ratio = visible_unknown_area / visible_total_surface_area
    """

    occupied_surface_voxels = as_points_array(occupied_surface_voxels)
    unknown_surface_voxels = as_points_array(unknown_surface_voxels)

    if len(occupied_surface_voxels) == 0 and len(unknown_surface_voxels) == 0:
        return {
            "gain_ratio": 0.0,
            "gain_score": 0.0,
            "visible_unknown_surface_pixels": 0,
            "visible_known_surface_pixels": 0,
            "visible_total_surface_pixels": 0,
            "unknown_surface_in_fov": 0,
            "known_surface_in_fov": 0,
        }

    all_points = []
    labels = []

    if len(occupied_surface_voxels) > 0:
        all_points.append(occupied_surface_voxels)
        labels.append(np.zeros(len(occupied_surface_voxels), dtype=np.int8))

    if len(unknown_surface_voxels) > 0:
        all_points.append(unknown_surface_voxels)
        labels.append(np.ones(len(unknown_surface_voxels), dtype=np.int8))

    all_points = np.vstack(all_points)
    labels = np.concatenate(labels)

    u, v, z, valid = project_points_to_image(
        points=all_points,
        camera_position=camera_position,
        look_at=look_at,
        fov_degrees=fov_degrees,
        image_width=image_width,
        image_height=image_height,
        aspect_ratio=aspect_ratio,
    )

    if not np.any(valid):
        return {
            "gain_ratio": 0.0,
            "gain_score": 0.0,
            "visible_unknown_surface_pixels": 0,
            "visible_known_surface_pixels": 0,
            "visible_total_surface_pixels": 0,
            "unknown_surface_in_fov": 0,
            "known_surface_in_fov": 0,
        }

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]
    labels_valid = labels[valid]

    unknown_surface_in_fov = int(np.count_nonzero(labels_valid == 1))
    known_surface_in_fov = int(np.count_nonzero(labels_valid == 0))

    pixel_index = v_valid * image_width + u_valid

    # Sort by pixel first, then depth.
    # For each pixel, the first entry is the closest surface voxel.
    order = np.lexsort((z_valid, pixel_index))

    pixel_sorted = pixel_index[order]
    labels_sorted = labels_valid[order]

    first_mask = np.ones(len(pixel_sorted), dtype=bool)
    first_mask[1:] = pixel_sorted[1:] != pixel_sorted[:-1]

    visible_labels = labels_sorted[first_mask]

    visible_unknown = int(np.count_nonzero(visible_labels == 1))
    visible_known = int(np.count_nonzero(visible_labels == 0))
    visible_total = visible_unknown + visible_known

    if visible_total == 0:
        gain_ratio = 0.0
    else:
        gain_ratio = visible_unknown / visible_total

    gain_score = visible_unknown * gain_ratio

    return {
        "gain_ratio": float(gain_ratio),
        "gain_score": float(gain_score),
        "visible_unknown_surface_pixels": visible_unknown,
        "visible_known_surface_pixels": visible_known,
        "visible_total_surface_pixels": visible_total,
        "unknown_surface_in_fov": unknown_surface_in_fov,
        "known_surface_in_fov": known_surface_in_fov,
    }
