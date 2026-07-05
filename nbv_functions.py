import os
import numpy as np
import cv2
#import matplotlib 
#matplotlib.use('Agg') # Force headless rendering to prevent memory leaks
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
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].

    This format matches Unity's Quaternion convention:
        x, y, z, w
    """

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

    # If camera is almost exactly above/below object, world_up becomes unstable.
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)

    right = normalize_vector(right)
    up = normalize_vector(np.cross(forward, right))

    # Columns are local x=right, y=up, z=forward
    R = np.column_stack((right, up, forward))

    return rotation_matrix_to_quaternion(R)


# ============================================================
# Input helper
# ============================================================

def _as_points_array(points):
    """
    Convert input points into an Nx3 float NumPy array.
    """

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
    num_views=4,
    distance_factor=2,
    height_fraction=0.6,
):
    """
    Generate fixed candidate viewpoints around the object bounding box.

    Each candidate view includes:
        - view_id
        - angle_deg
        - position: Unity/world camera position [x, y, z]
        - look_at: Unity/world target position [x, y, z]
        - rotation_quat: quaternion [x, y, z, w] that faces look_at

    Important:
        position + look_at are the single source of truth.
        They are used for both:
            1. Matplotlib candidate-view gain image
            2. Unity drone target pose
    """

    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    if np.any(bbox_max < bbox_min):
        raise ValueError("bbox_max must be >= bbox_min in every axis")

    center = 0.5 * (bbox_min + bbox_max)
    size = bbox_max - bbox_min

    # Horizontal radius needed to be outside bbox footprint
    half_diag_xz = 0.5 * np.sqrt(size[0] ** 2 + size[2] ** 2)
    radius = half_diag_xz * distance_factor

    # Candidate camera height
    cam_y = bbox_min[1] + height_fraction * size[1]

    candidate_views = []

    for i in range(num_views):
        angle = 2.0 * np.pi * i / num_views

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

        candidate_views.append({
            "view_id": i,
            "angle_deg": float(np.degrees(angle)),
            "position": cam_pos,
            "look_at": look_at,
            "rotation_quat": rotation_quat,
        })

    return candidate_views


# ============================================================
# Image-based unknown area score
# ============================================================

def compute_area_gain_from_matplotlib_figure(
    fig,
    orange_lower=(5, 80, 80),
    orange_upper=(35, 255, 255),
    blue_lower=(90, 50, 50),
    blue_upper=(140, 255, 255),
):
    """
    Compute image-based unknown-area score from a rendered Matplotlib figure.

    Orange = unknown voxels
    Blue = occupied / known voxels

    Returns:
        dict with gain ratio, gain score, and pixel counts
    """

    # Draw the current Matplotlib canvas
    fig.canvas.draw()

    # Convert canvas to RGB image array
    width, height = fig.canvas.get_width_height()
    img_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_rgb = img_rgb.reshape((height, width, 3))

    # Convert RGB to HSV for color thresholding
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    orange_lower = np.array(orange_lower, dtype=np.uint8)
    orange_upper = np.array(orange_upper, dtype=np.uint8)

    blue_lower = np.array(blue_lower, dtype=np.uint8)
    blue_upper = np.array(blue_upper, dtype=np.uint8)

    orange_mask = cv2.inRange(img_hsv, orange_lower, orange_upper)
    blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)

    orange_area = int(np.count_nonzero(orange_mask))
    blue_area = int(np.count_nonzero(blue_mask))

    total_area = orange_area + blue_area

    if total_area == 0:
        gain_ratio = 0.0
    else:
        gain_ratio = orange_area / total_area

    # NBV-like image score:
    # high if the image contains a lot of unknown area,
    # and the unknown area is large compared with known area.
    gain_score = orange_area * gain_ratio

    return {
        "gain_ratio": gain_ratio,
        "gain_score": gain_score,
        "unknown_area_pixels": orange_area,
        "known_area_pixels": blue_area,
        "total_area_pixels": total_area,
    }


# ============================================================
# Helper for Matplotlib camera angle
# ============================================================

def compute_matplotlib_view_angles(camera_position, look_at):
    """
    Convert a Unity/world camera pose into Matplotlib view angles.

    World convention:
        x = world X
        y = world Y / height
        z = world Z

    Plot convention used here:
        plot x-axis = world X
        plot y-axis = world Z
        plot z-axis = world Y / height

    Important:
        This does not create a separate camera pose.
        It converts the same position + look_at used for Unity into
        Matplotlib elev/azim angles for the candidate-view image.
    """

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
    """
    Compute equal-length axis limits around the voxel block.

    Input points are in world coordinates:
        [x, y, z]

    Returned limits are also in world-coordinate order:
        x_lim, y_lim, z_lim

    The plotting function will apply them according to:
        plot x = world x
        plot y = world z
        plot z = world y
    """

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
    Render occupied and unknown voxels from one candidate camera view.

    Blue = occupied / known voxels
    Orange = unknown voxels

    Also shows the candidate camera position as a black triangle
    and a dashed line from the camera to the look_at point.

    Uses the same candidate pose that should be sent to Unity:
        - camera_position
        - look_at

    The figure uses equal scale for x, y, and z axes.

    If save_path is provided, the rendered image is saved.

    Returns:
        fig, ax
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

    # Plot convention:
    #   plot x = world X
    #   plot y = world Z
    #   plot z = world Y / height

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

    # ---------------------------------------------------------
    # Put back candidate camera triangle + viewing line
    # ---------------------------------------------------------
    if show_camera_triangle:
        cam_x = camera_position[0]
        cam_y_plot = camera_position[2]   # world Z -> plot Y
        cam_z_plot = camera_position[1]   # world Y -> plot Z

        look_x = look_at[0]
        look_y_plot = look_at[2]          # world Z -> plot Y
        look_z_plot = look_at[1]          # world Y -> plot Z

        # Camera candidate position as triangle
        ax.scatter(
            cam_x,
            cam_y_plot,
            cam_z_plot,
            s=80,
            c="black",
            marker="^",
            depthshade=False,
        )

        # Dashed viewing direction line
        ax.plot(
            [cam_x, look_x],
            [cam_y_plot, look_y_plot],
            [cam_z_plot, look_z_plot],
            c="black",
            linestyle="--",
            linewidth=1.5,
        )

    # Apply equal-length limits.
    # Remember:
    #   Matplotlib x-axis = world x
    #   Matplotlib y-axis = world z
    #   Matplotlib z-axis = world y
    ax.set_xlim(x_lim)
    ax.set_ylim(z_lim)
    ax.set_zlim(y_lim)

    # Force equal visual scale on all 3 axes.
    ax.set_box_aspect((1, 1, 1))

    # Use the view angle derived from the same position + look_at
    # that will be sent to Unity.
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
# NBV using image-based unknown score only
# ============================================================
def compute_next_best_view(
    occupied_voxels, unknown_voxels, object_center, candidate_views,
    current_camera_position=None, visited_view_ids=None,
    lambda_distance=1.0, save_debug_images=False, output_folder="potential_views"
):
    if occupied_voxels is None: occupied_voxels = np.empty((0, 3))
    if unknown_voxels is None: unknown_voxels = np.empty((0, 3))

    occupied_voxels = _as_points_array(occupied_voxels)
    unknown_voxels = _as_points_array(unknown_voxels)
    object_center = np.asarray(object_center, dtype=float)

    if current_camera_position is not None:
        current_camera_position = np.asarray(current_camera_position, dtype=float)

    if visited_view_ids is None: visited_view_ids = set()
    else: visited_view_ids = set(visited_view_ids)

    if len(unknown_voxels) == 0:
        print("No unknown voxels left. NBV not needed.")
        return None, 0.0

    if candidate_views is None or len(candidate_views) == 0:
        print("No candidate views available.")
        return None, 0.0

    if save_debug_images:
        os.makedirs(output_folder, exist_ok=True)

    best_view = None
    best_score = -np.inf

    for view in candidate_views:
        view_id = view["view_id"]

        if view_id in visited_view_ids:
            view.update({"gain_ratio": 0.0, "gain_score_raw": 0.0, "unknown_area_pixels": 0, "known_area_pixels": 0, "total_area_pixels": 0, "distance": np.inf, "score": -np.inf})
            continue

        camera_position = np.asarray(view["position"], dtype=float)
        look_at = np.asarray(view["look_at"], dtype=float) if "look_at" in view else object_center.copy()

        view["position"] = camera_position
        view["look_at"] = look_at
        view["rotation_quat"] = look_at_quaternion(camera_position=camera_position, target_position=look_at)

        fig, ax = render_candidate_view_figure(
            occupied_voxels=occupied_voxels, unknown_voxels=unknown_voxels,
            camera_position=camera_position, look_at=look_at,
            zoom_margin=0.08, figsize=(6, 6), marker_size=2,
        )

        area_result = compute_area_gain_from_matplotlib_figure(fig)
        raw_gain_score = float(area_result["gain_score"])
        gain_ratio = float(area_result["gain_ratio"])
        unknown_area = int(area_result["unknown_area_pixels"])
        known_area = int(area_result["known_area_pixels"])
        total_area = int(area_result["total_area_pixels"])

        travel_distance = 0.0 if current_camera_position is None else float(np.linalg.norm(camera_position - current_camera_position))
        score = gain_ratio / (1.0 + lambda_distance * travel_distance)

        view.update({"gain_ratio": gain_ratio, "gain_score_raw": raw_gain_score, "unknown_area_pixels": unknown_area, "known_area_pixels": known_area, "total_area_pixels": total_area, "distance": travel_distance, "score": score})

        if save_debug_images:
            filepath = os.path.join(output_folder, f"candidate_{view_id:02d}_unknown_{unknown_area}_ratio_{gain_ratio:.3f}_score_{score:.4f}.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0)

        # Clear figure explicitly to free memory
        fig.clf()
        plt.close(fig)

        if score > best_score:
            best_score = score
            best_view = view

    if best_view is None:
        return None, 0.0

    return best_view, best_score