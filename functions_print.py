import numpy as np
import matplotlib.pyplot as plt
from functions_compute import make_valid_depth_mask

DEPTH_MIN_VALID = 0.0
DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01
MAX_POINTS_TO_PLOT = 30_000


def remove_overlapping_points(points_to_filter, priority_points, decimals=4):
    """
    Removes points from points_to_filter that exist in priority_points.
    Rounds to a specified number of decimals to handle floating-point inaccuracies.
    """
    if len(points_to_filter) == 0:
        return points_to_filter
    if len(priority_points) == 0:
        return points_to_filter

    priority_set = set(tuple(np.round(p, decimals)) for p in priority_points)
    mask = [tuple(np.round(p, decimals)) not in priority_set for p in points_to_filter]
    
    return points_to_filter[mask]


def downsample_points(points, max_points=MAX_POINTS_TO_PLOT):
    if points is None or len(points) == 0:
        return np.empty((0, 3))
    if len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), size=max_points, replace=False)
    return points[idx]


def print_depth_info(depth, width, height, fov, near, far, position, quaternion):
    basic_valid = (
        np.isfinite(depth)
        & (depth > DEPTH_MIN_VALID)
        & (depth != DEPTH_INVALID_VALUE)
    )
    final_valid = make_valid_depth_mask(depth, far)
    invalid_minus_one = np.count_nonzero(depth == DEPTH_INVALID_VALUE)
    near_far_pixels = np.count_nonzero(
        np.isfinite(depth) & (depth >= far - FAR_PLANE_MARGIN)
    )

    print("\n--- Received depth frame ---")
    print("Depth shape:", depth.shape)
    print("Width, height:", width, height)
    print("FOV:", fov)
    print("Near/Far:", near, far)
    print("Camera position:", position)
    print("Camera quaternion:", quaternion)
    print("Basic valid pixels:", np.count_nonzero(basic_valid), "/", width * height)
    print("Final valid pixels:", np.count_nonzero(final_valid), "/", width * height)
    print("Invalid -1 pixels:", invalid_minus_one)
    print("Near-far pixels:", near_far_pixels)
    print("Center depth:", depth[height // 2, width // 2])

    if np.any(final_valid):
        print("Min final valid depth:", np.min(depth[final_valid]))
        print("Max final valid depth:", np.max(depth[final_valid]))
        print("Mean final valid depth:", np.mean(depth[final_valid]))
    else:
        print("No final valid depth pixels.")


def set_axes_equal(ax, points):
    """Equal scaling for 3D plots, regardless of axis convention."""
    if points is None or len(points) == 0:
        return

    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]
    x_mid = 0.5 * (x_data.min() + x_data.max())
    y_mid = 0.5 * (y_data.min() + y_data.max())
    z_mid = 0.5 * (z_data.min() + z_data.max())

    max_range = max(np.ptp(x_data), np.ptp(y_data), np.ptp(z_data), 1e-6)
    half_range = max_range / 2.0

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)
    ax.set_box_aspect([1, 1, 1])


def plot_bbox_edges(ax, bbox_corners):
    if bbox_corners is None:
        return

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        p1 = bbox_corners[start]
        p2 = bbox_corners[end]
        ax.plot(
            [p1[0], p2[0]], 
            [p1[2], p2[2]], 
            [p1[1], p2[1]], 
            linewidth=2.0,
        )


def plot_voxel_centers_3d(
    occupied_centers=None,
    unknown_centers=None,
    bbox_corners=None,
    max_voxels_to_plot=300_000,
    title="Voxel Reconstruction",
):
    if occupied_centers is None:
        occupied_centers = np.empty((0, 3))
    if unknown_centers is None:
        unknown_centers = np.empty((0, 3))

    occupied_centers = np.asarray(occupied_centers)
    unknown_centers = np.asarray(unknown_centers)

    unknown_centers = remove_overlapping_points(
        points_to_filter=unknown_centers,
        priority_points=occupied_centers,
    )

    if len(occupied_centers) == 0 and len(unknown_centers) == 0:
        print("No voxel centers to visualize.")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_points_list = []

    if len(unknown_centers) > 0:
        unknown_plot = downsample_points(unknown_centers, max_voxels_to_plot)
        ax.scatter(
            unknown_plot[:, 0], unknown_plot[:, 2], unknown_plot[:, 1],
            s=3, alpha=0.18, color="tab:orange", label="Unknown voxels",
        )
        all_points_list.append(unknown_plot)

    if len(occupied_centers) > 0:
        occupied_plot = downsample_points(occupied_centers, max_voxels_to_plot)
        ax.scatter(
            occupied_plot[:, 0], occupied_plot[:, 2], occupied_plot[:, 1],
            s=4, alpha=0.85, color="tab:blue", label="Occupied voxels",
        )
        all_points_list.append(occupied_plot)

    if bbox_corners is not None:
        plot_bbox_edges(ax, bbox_corners)
        all_points_list.append(np.asarray(bbox_corners))

    all_points = np.vstack(all_points_list)

    ax.set_title(title)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Z [m]")
    ax.set_zlabel("World Y / Height [m]")
    ax.legend()
    set_axes_equal(ax, all_points)
    ax.format_coord = lambda x, y: ""

    plt.tight_layout()
    plt.show()


def plot_unknown_surface_voxel_history(view_numbers, unknown_surface_counts):
    if len(view_numbers) == 0 or len(unknown_surface_counts) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(view_numbers, unknown_surface_counts, marker="o", linewidth=2)
    plt.title("Unknown Surface Voxels After Each View")
    plt.xlabel("Object view number")
    plt.ylabel("Number of unknown surface voxels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_total_unknown_occupied_vs_true_volume(view_numbers, total_unknown_occupied_volumes, true_volumes):
    if len(view_numbers) == 0 or len(total_unknown_occupied_volumes) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(
        view_numbers, total_unknown_occupied_volumes,
        marker="o", linewidth=2, label="Occupied + unknown volume",
    )
    if true_volumes is not None and len(true_volumes) == len(view_numbers):
        plt.plot(
            view_numbers, true_volumes,
            linestyle="--", marker="s", linewidth=2, label="True Unity volume",
        )
    plt.title("Occupied + Unknown Volume vs True Volume")
    plt.xlabel("Object view number")
    plt.ylabel("Volume [m³]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()