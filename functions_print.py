import numpy as np
import pandas as pd
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
    plt.figure(figsize=(7, 4))
    plt.plot(view_numbers, unknown_surface_counts, marker="o", linewidth=2)
    plt.title("Unknown Surface Voxels After Each View")
    plt.xlabel("Object view number")
    plt.ylabel("Number of unknown surface voxels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_total_unknown_occupied_vs_true_volume(view_numbers, total_unknown_occupied_volumes, true_volumes):

    plt.figure(figsize=(7, 4))
    plt.plot(
        view_numbers, total_unknown_occupied_volumes,
        marker="o", linewidth=2, label="Occupied + unknown volume",
    )
    plt.plot(
        view_numbers, true_volumes,
        linestyle="--", marker="s", linewidth=2, label="True Unity volume"
    )
    plt.title("Occupied + Unknown Volume vs True Volume")
    plt.xlabel("Object view number")
    plt.ylabel("Volume [m³]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_cum_nbv_gain_distance_history_sbs(
    view_numbers,
    nbv_gains,
    nbv_distances,
):
    cumulative_gains = np.cumsum(nbv_gains)
    cumulative_distances = np.cumsum(nbv_distances)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        sharex=True,
    )

    # Left: cumulative NBV gain
    axes[0].plot(
        view_numbers,
        cumulative_gains,
        marker="o",
        linewidth=2,
        color="tab:green",
    )
    axes[0].set_title("Cumulative NBV Gain")
    axes[0].set_xlabel("Object View Number")
    axes[0].set_ylabel("Cumulative Gain")
    axes[0].grid(True)

    # Right: cumulative travel distance
    axes[1].plot(
        view_numbers,
        cumulative_distances,
        marker="s",
        linewidth=2,
        color="tab:purple",
    )
    axes[1].set_title("Cumulative Distance to NBV")
    axes[1].set_xlabel("Object View Number")
    axes[1].set_ylabel("Cumulative Distance [m]")
    axes[1].grid(True)

    fig.suptitle(
        "Cumulative Gain and Travel Distance After Each View",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()
    
def plot_cum_nbv_gain_distance_history_one(
    view_numbers,
    nbv_gains,
    nbv_distances,
):
    cumulative_gains = np.cumsum(nbv_gains)
    cumulative_distances = np.cumsum(nbv_distances)

    fig, ax_gain = plt.subplots(figsize=(8, 5))

    # Left y-axis: cumulative gain
    gain_line = ax_gain.plot(
        view_numbers,
        cumulative_gains,
        marker="o",
        linewidth=2,
        color="tab:green",
        label="Cumulative NBV gain",
    )

    ax_gain.set_xlabel("Object view number")
    ax_gain.set_ylabel("Cumulative gain", color="tab:green")
    ax_gain.tick_params(axis="y", labelcolor="tab:green")
    ax_gain.grid(True, alpha=0.3)

    # Right y-axis: cumulative distance
    ax_distance = ax_gain.twinx()

    distance_line = ax_distance.plot(
        view_numbers,
        cumulative_distances,
        marker="s",
        linewidth=2,
        color="tab:purple",
        label="Cumulative distance",
    )

    ax_distance.set_ylabel(
        "Cumulative distance [m]",
        color="tab:purple",
    )
    ax_distance.tick_params(axis="y", labelcolor="tab:purple")

    # Combined legend
    lines = gain_line + distance_line
    labels = [line.get_label() for line in lines]
    ax_gain.legend(lines, labels, loc="upper left")

    plt.title(
        "Cumulative NBV Gain and Travel Distance After Each View"
    )

    fig.tight_layout()
    plt.show()
    
    
def save_nbv_history_to_excel(
    view_numbers,
    nbv_gains,
    nbv_distances,
    output_file_path,
):
    df = pd.DataFrame({
        "View Number": view_numbers,
        "NBV Gain": nbv_gains,
        "NBV Distance": nbv_distances,
        "Cumulative Gain": np.cumsum(nbv_gains),
        "Cumulative Distance": np.cumsum(nbv_distances),
    })
    df.to_excel(output_file_path, index=False)


def save_volume_history_to_excel(
    view_numbers,
    total_unknown_occupied_volumes,
    true_volumes,
    output_file_path,
):
    df = pd.DataFrame({
        "View Number": view_numbers,
        "Total Unknown + Occupied Volume": total_unknown_occupied_volumes,
        "True Unity Volume": true_volumes,
    })
    df.to_excel(output_file_path, index=False)
    

def plot_nbv_evaluation_history(
    view_numbers,
    nbv_gains,
    nbv_distances,
    estimated_volumes,
    true_volumes,
    output_file_path = None,
):

    #view_numbers = np.asarray(view_numbers, dtype=int)
    #nbv_gains = np.asarray(nbv_gains, dtype=float)
    #nbv_distances = np.asarray(nbv_distances, dtype=float)
    estimated_volumes = np.asarray(estimated_volumes, dtype=float)
    true_volumes = np.asarray(true_volumes, dtype=float)


    nbv_gains = np.insert(nbv_gains, 0, 0.0)
    nbv_distances = np.insert(nbv_distances, 0, 0.0)
    gd_views = np.insert(view_numbers, 0, 0)
    
    cumulative_gains = np.cumsum(nbv_gains)
    cumulative_distances = np.cumsum(nbv_distances)

    volume_error_percent = (
        np.abs(estimated_volumes - true_volumes)
        / true_volumes
        * 100.0
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
    )

    ax_gain = axes[0]

    ax_gain.bar(
        gd_views,
        nbv_gains,
        alpha=0.45,
        label="Gain from current view",
    )

    ax_gain.plot(
        gd_views,
        cumulative_gains,
        marker="o",
        linewidth=2,
        label="Cumulative gain",
    )

    ax_gain.set_title("Information Gain")
    ax_gain.set_ylabel("Gain")
    ax_gain.grid(True, alpha=0.3)
    ax_gain.legend()

    # ---------------------------------------------------------
    # Distance
    # ---------------------------------------------------------
    ax_distance = axes[1]

    ax_distance.bar(
        gd_views,
        nbv_distances,
        alpha=0.45,
        label="Distance to current view",
    )

    ax_distance.plot(
        gd_views,
        cumulative_distances,
        marker="s",
        linewidth=2,
        label="Cumulative distance",
    )

    ax_distance.set_title("Travel Distance")
    ax_distance.set_ylabel("Distance [m]")
    ax_distance.grid(True, alpha=0.3)
    ax_distance.legend()

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------
    ax_volume = axes[2]

    ax_volume.plot(
        view_numbers,
        estimated_volumes,
        marker="o",
        linewidth=2,
        label="Estimated occupied + unknown volume",
    )

    ax_volume.plot(
        view_numbers,
        true_volumes,
        linestyle="--",
        linewidth=2,
        label="True Unity volume",
    )

    final_error = volume_error_percent[-1]

    ax_volume.annotate(
        f"Final error = {final_error:.2f}%",
        xy=(view_numbers[-1], estimated_volumes[-1]),
        xytext=(8, 10),
        textcoords="offset points",
    )

    ax_volume.set_title("Estimated Volume Convergence")
    ax_volume.set_xlabel("Object view number")
    ax_volume.set_ylabel("Volume [m³]")
    ax_volume.grid(True, alpha=0.3)
    ax_volume.legend()

    for ax in axes:
        ax.set_xticks(view_numbers)

    #fig.suptitle(
    #    f"Path-Planning Evaluation: {planner_name}",
    #    fontsize=15,
    #)

    plt.tight_layout()
    if output_file_path is not None:
        plt.savefig(
            output_file_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()