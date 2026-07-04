import numpy as np
import matplotlib.pyplot as plt

from functions_compute import make_valid_depth_mask


DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01
MAX_POINTS_TO_PLOT = 30_000


def downsample_points(points, max_points=MAX_POINTS_TO_PLOT):
    points = np.asarray(points)
    if len(points) <= max_points:
        return points
    ids = np.random.choice(len(points), max_points, replace=False)
    return points[ids]


def print_depth_info(depth, width, height, fov, near, far, position, quaternion):
    valid = make_valid_depth_mask(depth, far)
    print("\n--- Received depth frame ---")
    print("Depth shape:", depth.shape)
    print("Width, height:", width, height)
    print("FOV:", fov)
    print("Near/Far:", near, far)
    print("Camera position:", position)
    print("Camera quaternion:", quaternion)
    print("Valid depth pixels:", np.count_nonzero(valid), "/", width * height)
    print("Invalid -1 pixels:", np.count_nonzero(depth == DEPTH_INVALID_VALUE))
    print("Near-far pixels:", np.count_nonzero(np.isfinite(depth) & (depth >= far - FAR_PLANE_MARGIN)))
    print("Center depth:", depth[height // 2, width // 2])
    if np.any(valid):
        print("Min valid depth:", np.min(depth[valid]))
        print("Max valid depth:", np.max(depth[valid]))
        print("Mean valid depth:", np.mean(depth[valid]))


def set_axes_equal_xzy_display(ax, points_world):
    points = np.asarray(points_world)
    if len(points) == 0:
        return

    x_data = points[:, 0]
    y_data = points[:, 2]
    z_data = points[:, 1]

    x_mid = 0.5 * (x_data.min() + x_data.max())
    y_mid = 0.5 * (y_data.min() + y_data.max())
    z_mid = 0.5 * (z_data.min() + z_data.max())
    half_range = 0.5 * max(np.ptp(x_data), np.ptp(y_data), np.ptp(z_data), 1e-6)

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
        p1, p2 = bbox_corners[start], bbox_corners[end]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], linewidth=2.0)


def plot_points_3d(points, title="Point Cloud", max_points=MAX_POINTS_TO_PLOT):
    points = downsample_points(points, max_points)
    if len(points) == 0:
        print("No points to plot.")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=1)
    ax.set_title(title)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Z [m]")
    ax.set_zlabel("World Y / Height [m]")
    set_axes_equal_xzy_display(ax, points)
    plt.tight_layout()
    plt.show()


def plot_depth_scatter_camera(points_camera):
    points = downsample_points(points_camera)
    if len(points) == 0:
        print("No camera points to plot.")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=points[:, 2], s=1, cmap="viridis")
    ax.set_title("Depth Image as 3D Scatter Plot - Camera Frame")
    ax.set_xlabel("Camera X [m]")
    ax.set_ylabel("Camera Z / Depth [m]")
    ax.set_zlabel("Camera Y [m]")
    fig.colorbar(scatter, ax=ax, label="Depth Z [m]")
    set_axes_equal_xzy_display(ax, points)
    plt.tight_layout()
    plt.show()


def plot_depth_scatter_world(points_world):
    plot_points_3d(points_world, title="Depth Image as 3D Scatter Plot - World Frame")


def plot_top_view_occupancy(points_world):
    points = np.asarray(points_world)
    if len(points) == 0:
        print("No world points to plot.")
        return

    plt.figure(figsize=(7, 7))
    plt.scatter(points[:, 0], points[:, 2], s=1)
    plt.title("Top View Occupancy")
    plt.xlabel("World X [m]")
    plt.ylabel("World Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_voxel_centers_3d(
    voxel_centers=None,
    bbox_corners=None,
    max_voxels_to_plot=300_000,
    title="Voxel Reconstruction",
    occupied_centers=None,
    unknown_centers=None,
):
    if occupied_centers is None and voxel_centers is not None:
        occupied_centers = voxel_centers

    occupied = np.empty((0, 3)) if occupied_centers is None else np.asarray(occupied_centers)
    unknown = np.empty((0, 3)) if unknown_centers is None else np.asarray(unknown_centers)

    if len(occupied) == 0 and len(unknown) == 0:
        print("No voxel centers to visualize.")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []

    if len(unknown) > 0:
        unknown_plot = downsample_points(unknown, max_voxels_to_plot)
        ax.scatter(unknown_plot[:, 0], unknown_plot[:, 2], unknown_plot[:, 1], s=3, alpha=0.18, color="tab:orange", label="Unknown voxels")
        all_points.append(unknown_plot)

    if len(occupied) > 0:
        occupied_plot = downsample_points(occupied, max_voxels_to_plot)
        ax.scatter(occupied_plot[:, 0], occupied_plot[:, 2], occupied_plot[:, 1], s=4, alpha=0.85, color="tab:blue", label="Occupied voxels")
        all_points.append(occupied_plot)

    if bbox_corners is not None:
        bbox_corners = np.asarray(bbox_corners)
        plot_bbox_edges(ax, bbox_corners)
        all_points.append(bbox_corners)

    ax.set_title(title)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Z [m]")
    ax.set_zlabel("World Y / Height [m]")
    ax.legend()
    set_axes_equal_xzy_display(ax, np.vstack(all_points))
    ax.format_coord = lambda x, y: ""
    plt.tight_layout()
    plt.show()


def plot_reconstruction_layers(
    observed_centers=None,
    extruded_centers=None,
    fused_centers=None,
    bbox_corners=None,
    max_voxels_to_plot=300_000,
    occupied_centers=None,
    unknown_centers=None,
):
    occupied = occupied_centers if occupied_centers is not None else fused_centers
    plot_voxel_centers_3d(
        occupied_centers=occupied,
        unknown_centers=unknown_centers,
        bbox_corners=bbox_corners,
        max_voxels_to_plot=max_voxels_to_plot,
        title="Open3D Voxel Reconstruction Layers",
    )


def plot_unknown_surface_voxel_history(view_numbers, unknown_surface_counts, title="Unknown Surface Voxels After Each View"):
    if len(view_numbers) == 0:
        print("No unknown surface voxel history to plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(view_numbers, unknown_surface_counts, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Object view number")
    plt.ylabel("Number of unknown surface voxels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_total_unknown_occupied_vs_true_volume(
    view_numbers,
    total_unknown_occupied_volumes,
    true_volumes,
    title="Occupied + Unknown Volume vs True Volume",
):
    if len(view_numbers) == 0:
        print("No volume history to plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(view_numbers, total_unknown_occupied_volumes, marker="o", linewidth=2, label="Occupied + unknown volume")

    if true_volumes is not None and len(true_volumes) == len(view_numbers):
        plt.plot(view_numbers, true_volumes, linestyle="--", marker="s", linewidth=2, label="True Unity volume")

    plt.title(title)
    plt.xlabel("Object view number")
    plt.ylabel("Volume [m³]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_volume_history(view_numbers, open3d_volumes, true_volumes=None):
    plot_total_unknown_occupied_vs_true_volume(view_numbers, open3d_volumes, true_volumes, title="Open3D Voxel Volume")
