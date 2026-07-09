import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# Quaternion utilities
# ============================================================

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


# ============================================================
# Input helper
# ============================================================

def _as_points_array(points):
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


# ============================================================
# Candidate view generation
# ============================================================

def generate_candidate_views_from_bbox(
    bbox_min,
    bbox_max,
    fov_degrees,
    num_azimuth_views=8,
    height_fractions=(0.35, 0.60, 0.85),
    margin_factor=1.2,
    min_radius_scale=0.25,
):
    """
    Generate diverse candidate viewpoints around the object bounding box.
    """

    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    center = 0.5 * (bbox_min + bbox_max)
    size = bbox_max - bbox_min

    object_radius = 0.5 * np.linalg.norm(size)
    half_fov_rad = 0.5 * np.deg2rad(fov_degrees)

    min_distance_to_center = (
        object_radius / np.sin(half_fov_rad)
    ) * margin_factor

    footprint_radius = 0.5 * np.linalg.norm(size[[0, 2]])
    min_horizontal_radius = min_radius_scale * footprint_radius

    candidate_views = []
    view_id = 0

    for height_fraction in height_fractions:
        cam_y = bbox_min[1] + height_fraction * size[1]
        vertical_offset = cam_y - center[1]

        horizontal_radius_sq = min_distance_to_center ** 2 - vertical_offset ** 2
        radius = np.sqrt(max(horizontal_radius_sq, 0.0))
        radius = max(radius, min_horizontal_radius)

        for i in range(num_azimuth_views):
            angle = 2.0 * np.pi * i / num_azimuth_views

            cam_pos = np.array([
                center[0] + radius * np.cos(angle),
                cam_y,
                center[2] + radius * np.sin(angle),
            ], dtype=float)

            look_at = center.copy()

            rotation_quat = look_at_quaternion(
                camera_position=cam_pos,
                target_position=look_at,
            )

            horizontal_dist = np.linalg.norm([
                cam_pos[0] - center[0],
                cam_pos[2] - center[2],
            ])

            elevation_deg = np.degrees(
                np.arctan2(cam_pos[1] - center[1], horizontal_dist + 1e-9)
            )

            candidate_views.append({
                "view_id": view_id,
                "angle_deg": float(np.degrees(angle)),
                "height_fraction": float(height_fraction),
                "elevation_deg": float(elevation_deg),
                "position": cam_pos,
                "look_at": look_at,
                "rotation_quat": rotation_quat,
            })

            view_id += 1

    return candidate_views


# ============================================================
# Surface voxel extraction
# ============================================================

def estimate_voxel_size_from_points(points):
    """
    Estimate voxel size from voxel center coordinates.

    This assumes voxel centers lie on a regular grid.
    """

    points = _as_points_array(points)

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

    points = _as_points_array(points)

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

    points = _as_points_array(points)

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


# ============================================================
# Camera projection / z-buffer scoring
# ============================================================

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

    points = _as_points_array(points)
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

    points = _as_points_array(points)

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

    occupied_surface_voxels = _as_points_array(occupied_surface_voxels)
    unknown_surface_voxels = _as_points_array(unknown_surface_voxels)

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


# ============================================================
# Matplotlib debug rendering
# ============================================================

def compute_matplotlib_view_angles(camera_position, look_at):
    camera_position = np.asarray(camera_position, dtype=float)
    look_at = np.asarray(look_at, dtype=float)

    direction = camera_position - look_at

    dx = direction[0]
    dy = direction[1]
    dz = direction[2]

    horizontal_dist = np.sqrt(dx ** 2 + dz ** 2)

    azim_deg = np.degrees(np.arctan2(dz, dx))
    elev_deg = np.degrees(np.arctan2(dy, horizontal_dist + 1e-9))

    return elev_deg, azim_deg


def compute_equal_axis_limits(points, zoom_margin=0.08):
    points = _as_points_array(points)

    if len(points) == 0:
        return (-1, 1), (-1, 1), (-1, 1)

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)

    center = 0.5 * (xyz_min + xyz_max)
    size = xyz_max - xyz_min

    max_size = float(np.max(size))

    if max_size <= 1e-9:
        max_size = 1.0

    half = 0.5 * max_size * (1.0 + zoom_margin)

    x_lim = (center[0] - half, center[0] + half)
    y_lim = (center[1] - half, center[1] + half)
    z_lim = (center[2] - half, center[2] + half)

    return x_lim, y_lim, z_lim


def render_candidate_view_figure(
    occupied_voxels,
    unknown_voxels,
    camera_position,
    look_at,
    zoom_margin=0.08,
    figsize=(6, 6),
    marker_size=2,
    save_path=None,
    dpi=150,
    close_after_save=False,
    show_camera_triangle=True,
):
    """
    Debug renderer only.

    This should not be used for NBV scoring anymore.
    """

    occupied_voxels = _as_points_array(occupied_voxels)
    unknown_voxels = _as_points_array(unknown_voxels)
    camera_position = np.asarray(camera_position, dtype=float)
    look_at = np.asarray(look_at, dtype=float)

    if len(occupied_voxels) > 0 and len(unknown_voxels) > 0:
        all_voxels = np.vstack([occupied_voxels, unknown_voxels])
    elif len(occupied_voxels) > 0:
        all_voxels = occupied_voxels
    elif len(unknown_voxels) > 0:
        all_voxels = unknown_voxels
    else:
        all_voxels = np.empty((0, 3))

    x_lim, y_lim, z_lim = compute_equal_axis_limits(
        all_voxels,
        zoom_margin=zoom_margin,
    )

    elev_deg, azim_deg = compute_matplotlib_view_angles(
        camera_position=camera_position,
        look_at=look_at,
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if len(occupied_voxels) > 0:
        ax.scatter(
            occupied_voxels[:, 0],
            occupied_voxels[:, 2],
            occupied_voxels[:, 1],
            s=marker_size,
            c="blue",
            alpha=1.0,
            depthshade=False,
        )

    if len(unknown_voxels) > 0:
        ax.scatter(
            unknown_voxels[:, 0],
            unknown_voxels[:, 2],
            unknown_voxels[:, 1],
            s=marker_size,
            c="orange",
            alpha=1.0,
            depthshade=False,
        )

    if show_camera_triangle:
        cam_x = camera_position[0]
        cam_y_plot = camera_position[2]
        cam_z_plot = camera_position[1]

        look_x = look_at[0]
        look_y_plot = look_at[2]
        look_z_plot = look_at[1]

        ax.scatter(
            cam_x,
            cam_y_plot,
            cam_z_plot,
            s=80,
            c="black",
            marker="^",
            depthshade=False,
        )

        ax.plot(
            [cam_x, look_x],
            [cam_y_plot, look_y_plot],
            [cam_z_plot, look_z_plot],
            c="black",
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(z_lim)
    ax.set_zlim(y_lim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev_deg, azim=azim_deg)

    ax.set_axis_off()
    ax.grid(False)

    plt.tight_layout(pad=0)

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )

        if close_after_save:
            plt.close(fig)

    return fig, ax


# ============================================================
# NBV using surface voxels
# ============================================================

def compute_next_best_view(
    occupied_voxels,
    unknown_voxels,
    object_center,
    candidate_views,
    current_camera_position=None,
    visited_view_ids=None,
    lambda_distance=1.0,
    save_debug_images=False,
    output_folder="potential_views",
    fov_degrees=58.0,
    image_width=160,
    image_height=160,
    aspect_ratio=16.0 / 9.0,
    voxel_size=None,
    use_distance_penalty=False,
    print_scores=True,
):
    """
    Compute next best view using visible surface voxels.

    This version does NOT render Matplotlib figures to compute the score.
    It uses direct projection + z-buffering.

    Score:
        visible_unknown_surface_pixels * gain_ratio

    where:
        gain_ratio =
            visible_unknown_surface_pixels /
            visible_total_surface_pixels

    Parameters:
        occupied_voxels:
            Known/occupied voxel centers.

        unknown_voxels:
            Unknown voxel centers.

        object_center:
            Object center.

        candidate_views:
            Views from generate_candidate_views_from_bbox().

        fov_degrees:
            Candidate camera vertical FOV.

        image_width, image_height:
            Low-resolution virtual image used for projected surface area.
            160x160 is usually enough and much faster than Matplotlib.

        voxel_size:
            Optional. If None, estimated from voxel centers.

        use_distance_penalty:
            If True, score is divided by:
                1 + lambda_distance * travel_distance
    """

    if occupied_voxels is None:
        occupied_voxels = np.empty((0, 3))

    if unknown_voxels is None:
        unknown_voxels = np.empty((0, 3))

    occupied_voxels = _as_points_array(occupied_voxels)
    unknown_voxels = _as_points_array(unknown_voxels)
    object_center = np.asarray(object_center, dtype=float)

    if current_camera_position is not None:
        current_camera_position = np.asarray(current_camera_position, dtype=float)

    if visited_view_ids is None:
        visited_view_ids = set()
    else:
        visited_view_ids = set(visited_view_ids)

    if len(unknown_voxels) == 0:
        print("No unknown voxels left. NBV not needed.")
        return None, 0.0

    if candidate_views is None or len(candidate_views) == 0:
        print("No candidate views available.")
        return None, 0.0

    # Estimate voxel size once.
    if voxel_size is None:
        if len(occupied_voxels) > 0 and len(unknown_voxels) > 0:
            voxel_size = estimate_voxel_size_from_points(
                np.vstack([occupied_voxels, unknown_voxels])
            )
        elif len(unknown_voxels) > 0:
            voxel_size = estimate_voxel_size_from_points(unknown_voxels)
        else:
            voxel_size = estimate_voxel_size_from_points(occupied_voxels)

    # Use shared origin so occupied and unknown voxel grids align.
    if len(occupied_voxels) > 0 and len(unknown_voxels) > 0:
        grid_origin = np.vstack([occupied_voxels, unknown_voxels]).min(axis=0)
    elif len(unknown_voxels) > 0:
        grid_origin = unknown_voxels.min(axis=0)
    elif len(occupied_voxels) > 0:
        grid_origin = occupied_voxels.min(axis=0)
    else:
        grid_origin = np.zeros(3)

    # Extract only surface voxels.
    occupied_surface_voxels = extract_surface_voxels(
        occupied_voxels,
        voxel_size=voxel_size,
        origin=grid_origin,
    )

    unknown_surface_voxels = extract_surface_voxels(
        unknown_voxels,
        voxel_size=voxel_size,
        origin=grid_origin,
    )

    if len(unknown_surface_voxels) == 0:
        print("No unknown surface voxels left. NBV not needed.")
        return None, 0.0

    if save_debug_images:
        os.makedirs(output_folder, exist_ok=True)

    print("")
    print("====================================================")
    print("Surface-based NBV")
    print(f"Voxel size used: {voxel_size}")
    print(f"Occupied voxels: {len(occupied_voxels)}")
    print(f"Unknown voxels: {len(unknown_voxels)}")
    print(f"Occupied surface voxels: {len(occupied_surface_voxels)}")
    print(f"Unknown surface voxels: {len(unknown_surface_voxels)}")
    print("====================================================")

    best_view = None
    best_score = -np.inf

    if print_scores:
        print("")
        print("--- Candidate Views: Surface-Based Gain ---")

    for view in candidate_views:
        view_id = view["view_id"]

        if view_id in visited_view_ids:
            view.update({
                "gain_ratio": 0.0,
                "gain_score_raw": 0.0,
                "visible_unknown_surface_pixels": 0,
                "visible_known_surface_pixels": 0,
                "visible_total_surface_pixels": 0,
                "unknown_surface_in_fov": 0,
                "known_surface_in_fov": 0,
                "distance": np.inf,
                "score": -np.inf,
            })

            if print_scores:
                print(f"View {view_id:02d} | visited | score = -inf")

            continue

        camera_position = np.asarray(view["position"], dtype=float)
        look_at = np.asarray(view["look_at"], dtype=float) if "look_at" in view else object_center.copy()

        view["position"] = camera_position
        view["look_at"] = look_at
        view["rotation_quat"] = look_at_quaternion(
            camera_position=camera_position,
            target_position=look_at,
        )

        gain_result = compute_surface_projection_gain(
            occupied_surface_voxels=occupied_surface_voxels,
            unknown_surface_voxels=unknown_surface_voxels,
            camera_position=camera_position,
            look_at=look_at,
            fov_degrees=fov_degrees,
            image_width=image_width,
            image_height=image_height,
            aspect_ratio=aspect_ratio,
        )

        raw_gain_score = float(gain_result["gain_score"])
        gain_ratio = float(gain_result["gain_ratio"])

        visible_unknown = int(gain_result["visible_unknown_surface_pixels"])
        visible_known = int(gain_result["visible_known_surface_pixels"])
        visible_total = int(gain_result["visible_total_surface_pixels"])

        unknown_in_fov = int(gain_result["unknown_surface_in_fov"])
        known_in_fov = int(gain_result["known_surface_in_fov"])

        if current_camera_position is None:
            travel_distance = 0.0
        else:
            travel_distance = float(np.linalg.norm(camera_position - current_camera_position))

        if use_distance_penalty:
            score = raw_gain_score / (1.0 + lambda_distance * travel_distance)
        else:
            score = raw_gain_score

        view.update({
            "gain_ratio": gain_ratio,
            "gain_score_raw": raw_gain_score,
            "visible_unknown_surface_pixels": visible_unknown,
            "visible_known_surface_pixels": visible_known,
            "visible_total_surface_pixels": visible_total,
            "unknown_surface_in_fov": unknown_in_fov,
            "known_surface_in_fov": known_in_fov,
            "distance": travel_distance,
            "score": score,
        })

        if print_scores:
            print(
                f"View {view_id:02d} | "
                f"angle = {view.get('angle_deg', 0.0):.1f} deg | "
                f"height_frac = {view.get('height_fraction', -1):.2f} | "
                f"visible unknown surface = {visible_unknown} | "
                f"visible known surface = {visible_known} | "
                f"ratio = {gain_ratio:.3f} | "
                f"score = {score:.4f}"
            )

        if save_debug_images:
            filepath = os.path.join(
                output_folder,
                f"candidate_{view_id:02d}_surface_unknown_{visible_unknown}_ratio_{gain_ratio:.3f}_score_{score:.4f}.png"
            )

            fig, ax = render_candidate_view_figure(
                occupied_voxels=occupied_surface_voxels,
                unknown_voxels=unknown_surface_voxels,
                camera_position=camera_position,
                look_at=look_at,
                zoom_margin=0.08,
                figsize=(6, 6),
                marker_size=2,
                save_path=filepath,
                dpi=150,
                close_after_save=True,
                show_camera_triangle=True,
            )

        if score > best_score:
            best_score = score
            best_view = view

    if best_view is None:
        return None, 0.0

    print("")
    print("====================================================")
    print("Selected NBV")
    print(f"View ID: {best_view['view_id']}")
    print(f"Angle: {best_view.get('angle_deg', 0.0):.1f} deg")
    print(f"Height fraction: {best_view.get('height_fraction', -1):.2f}")
    print(f"Visible unknown surface pixels: {best_view['visible_unknown_surface_pixels']}")
    print(f"Visible known surface pixels: {best_view['visible_known_surface_pixels']}")
    print(f"Gain ratio: {best_view['gain_ratio']:.3f}")
    print(f"Best score: {best_score:.4f}")
    print("====================================================")
    print("")

    return best_view, best_score