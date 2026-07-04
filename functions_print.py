import numpy as np
import matplotlib.pyplot as plt

from functions_compute import make_valid_depth_mask


DEPTH_MIN_VALID = 0.0
DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01
MAX_POINTS_TO_PLOT = 30_000


def downsample_points(points, max_points=MAX_POINTS_TO_PLOT):
    if points is None or len(points) == 0:
        return np.empty((0, 3))

    if len(points) <= max_points:
        return points

    idx = np.random.choice(len(points), size=max_points, replace=False)
    return points[idx]


def remove_overlapping_points(points_to_filter, priority_points, decimals=6):
    """
    Remove points from points_to_filter if they overlap with priority_points.

    Used for plotting:
    - occupied/blue voxels have priority
    - unknown/orange voxels are removed if they are at the same center
      as occupied/blue voxels
    """

    if points_to_filter is None or len(points_to_filter) == 0:
        return np.empty((0, 3))

    if priority_points is None or len(priority_points) == 0:
        return np.asarray(points_to_filter)

    points_to_filter = np.asarray(points_to_filter)
    priority_points = np.asarray(priority_points)

    filter_keys = np.round(points_to_filter, decimals=decimals)
    priority_keys = np.round(priority_points, decimals=decimals)

    priority_set = set(map(tuple, priority_keys))

    keep_mask = np.array([
        tuple(p) not in priority_set
        for p in filter_keys
    ])

    return points_to_filter[keep_mask]


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


def plot_depth_scatter_camera(points_camera):
    if points_camera is None or len(points_camera) == 0:
        print("No camera points to plot.")
        return

    points_plot = downsample_points(points_camera)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points_plot[:, 0],
        points_plot[:, 2],
        points_plot[:, 1],
        c=points_plot[:, 2],
        s=1,
        cmap="viridis",
    )

    ax.set_title("Depth Image as 3D Scatter Plot - Camera Frame")
    ax.set_xlabel("Camera X [m]")
    ax.set_ylabel("Camera Z / Depth [m]")
    ax.set_zlabel("Camera Y [m]")

    fig.colorbar(scatter, ax=ax, label="Depth Z [m]")

    set_axes_equal_xyz_display(ax, points_plot)

    plt.tight_layout()
    plt.show()


def plot_depth_scatter_world(points_world):
    if points_world is None or len(points_world) == 0:
        print("No world points to plot.")
        return

    points_plot = downsample_points(points_world)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points_plot[:, 0],
        points_plot[:, 2],
        points_plot[:, 1],
        c=points_plot[:, 1],
        s=1,
        cmap="viridis",
    )

    ax.set_title("Depth Image as 3D Scatter Plot - World Frame")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Z [m]")
    ax.set_zlabel("World Y / Height [m]")

    fig.colorbar(scatter, ax=ax, label="World Y / Height [m]")

    set_axes_equal_xzy_display(ax, points_plot)

    plt.tight_layout()
    plt.show()


def plot_top_view_occupancy(points_world):
    if points_world is None or len(points_world) == 0:
        print("No world points to plot.")
        return

    x = points_world[:, 0]
    z = points_world[:, 2]

    plt.figure(figsize=(7, 7))
    plt.scatter(x, z, s=1)
    plt.title("Top View Occupancy")
    plt.xlabel("World X [m]")
    plt.ylabel("World Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bbox_edges(ax, bbox_corners):
    """
    Plot bounding-box edges using display convention:

    Display X-axis = world X
    Display Y-axis = world Z
    Display Z-axis = world Y / height
    """

    if bbox_corners is None:
        return

    edges = [
        # bottom rectangle
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        # top rectangle
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        # vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for start, end in edges:
        p1 = bbox_corners[start]
        p2 = bbox_corners[end]

        ax.plot(
            [p1[0], p2[0]],  # world X
            [p1[2], p2[2]],  # world Z
            [p1[1], p2[1]],  # world Y / height
            linewidth=2.0,
        )


def plot_voxel_centers_3d(
    voxel_centers=None,
    bbox_corners=None,
    max_voxels_to_plot=300_000,
    title="Voxel Reconstruction",
    occupied_centers=None,
    unknown_centers=None,
):
    """
    Plot voxel centers with separate colors for occupied and unknown voxels.

    Backward compatible usage:
    - plot_voxel_centers_3d(recon.fused_centers)

    New probabilistic/occupancy usage:
    - plot_voxel_centers_3d(
          occupied_centers=recon.occupied_centers,
          unknown_centers=recon.unknown_centers,
          bbox_corners=recon.bbox_corners,
      )

    Display convention:
    - plot x-axis = world X
    - plot y-axis = world Z
    - plot z-axis = world Y / height
    """

    # Backward compatibility: if the old positional argument is used,
    # treat it as occupied voxels.
    if occupied_centers is None and voxel_centers is not None:
        occupied_centers = voxel_centers

    if occupied_centers is None:
        occupied_centers = np.empty((0, 3))
    if unknown_centers is None:
        unknown_centers = np.empty((0, 3))

    occupied_centers = np.asarray(occupied_centers)
    unknown_centers = np.asarray(unknown_centers)

    # Important:
    # Occupied/blue voxels have priority.
    # If an unknown/orange voxel has the same center as an occupied/blue voxel,
    # remove the orange voxel before plotting.
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

    # Plot unknown first so occupied voxels remain visible on top.
    if len(unknown_centers) > 0:
        unknown_plot = downsample_points(unknown_centers, max_voxels_to_plot)
        ax.scatter(
            unknown_plot[:, 0],
            unknown_plot[:, 2],
            unknown_plot[:, 1],
            s=3,
            alpha=0.18,
            color="tab:orange",
            label="Unknown voxels",
        )
        all_points_list.append(unknown_plot)

    if len(occupied_centers) > 0:
        occupied_plot = downsample_points(occupied_centers, max_voxels_to_plot)
        ax.scatter(
            occupied_plot[:, 0],
            occupied_plot[:, 2],
            occupied_plot[:, 1],
            s=4,
            alpha=0.85,
            color="tab:blue",
            label="Occupied voxels",
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

    set_axes_equal_xzy_display(ax, all_points)

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
    """
    Plot reconstruction layers with separate colors.

    Old usage is still supported:
    - observed_centers: directly measured surface voxels
    - extruded_centers: first-frame filled prior
    - fused_centers: final volume estimate

    New probabilistic/occupancy usage is also supported:
    - occupied_centers: voxels classified as occupied
    - unknown_centers: voxels that remain uncertain / unobserved
    """

    if observed_centers is None:
        observed_centers = np.empty((0, 3))
    if extruded_centers is None:
        extruded_centers = np.empty((0, 3))
    if fused_centers is None:
        fused_centers = np.empty((0, 3))
    if occupied_centers is None:
        occupied_centers = np.empty((0, 3))
    if unknown_centers is None:
        unknown_centers = np.empty((0, 3))

    observed_centers = np.asarray(observed_centers)
    extruded_centers = np.asarray(extruded_centers)
    fused_centers = np.asarray(fused_centers)
    occupied_centers = np.asarray(occupied_centers)
    unknown_centers = np.asarray(unknown_centers)

    # Important:
    # Occupied/blue voxels have priority over unknown/orange voxels.
    unknown_centers = remove_overlapping_points(
        points_to_filter=unknown_centers,
        priority_points=occupied_centers,
    )

    if (
        len(observed_centers) == 0
        and len(extruded_centers) == 0
        and len(fused_centers) == 0
        and len(occupied_centers) == 0
        and len(unknown_centers) == 0
    ):
        print("No reconstruction layers to plot.")
        return

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points_list = []

    # Unknown first: transparent background context.
    if len(unknown_centers) > 0:
        unk_plot = downsample_points(unknown_centers, max_voxels_to_plot)
        ax.scatter(
            unk_plot[:, 0],
            unk_plot[:, 2],
            unk_plot[:, 1],
            s=3,
            alpha=0.16,
            color="tab:orange",
            label="Unknown voxels",
        )
        all_points_list.append(unk_plot)

    if len(extruded_centers) > 0:
        ext_plot = downsample_points(extruded_centers, max_voxels_to_plot)
        ax.scatter(
            ext_plot[:, 0],
            ext_plot[:, 2],
            ext_plot[:, 1],
            s=3,
            alpha=0.20,
            color="tab:orange",
            label="Extruded prior",
        )
        all_points_list.append(ext_plot)

    if len(fused_centers) > 0:
        fused_plot = downsample_points(fused_centers, max_voxels_to_plot)
        ax.scatter(
            fused_plot[:, 0],
            fused_plot[:, 2],
            fused_plot[:, 1],
            s=2,
            alpha=0.25,
            color="tab:green",
            label="Fused volume",
        )
        all_points_list.append(fused_plot)

    # Occupied after unknown/fused so it appears clearly on top.
    if len(occupied_centers) > 0:
        occ_plot = downsample_points(occupied_centers, max_voxels_to_plot)
        ax.scatter(
            occ_plot[:, 0],
            occ_plot[:, 2],
            occ_plot[:, 1],
            s=5,
            alpha=0.90,
            color="tab:blue",
            label="Occupied voxels",
        )
        all_points_list.append(occ_plot)

    if len(observed_centers) > 0:
        obs_plot = downsample_points(observed_centers, max_voxels_to_plot)
        ax.scatter(
            obs_plot[:, 0],
            obs_plot[:, 2],
            obs_plot[:, 1],
            s=6,
            alpha=0.95,
            color="tab:red",
            label="Observed surface",
        )
        all_points_list.append(obs_plot)

    if bbox_corners is not None:
        plot_bbox_edges(ax, bbox_corners)
        all_points_list.append(np.asarray(bbox_corners))

    all_points = np.vstack(all_points_list)

    ax.set_title("Extrusion-Fusion Reconstruction Layers")
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Z [m]")
    ax.set_zlabel("World Y / Height [m]")

    ax.legend()

    set_axes_equal_xzy_display(ax, all_points)

    ax.format_coord = lambda x, y: ""

    plt.tight_layout()
    plt.show()


def set_axes_equal_xzy_display(ax, points_world):
    """
    Equal scaling for plots where display convention is:

    ax x = world X
    ax y = world Z
    ax z = world Y / height
    """

    if points_world is None or len(points_world) == 0:
        return

    x_data = points_world[:, 0]
    y_data = points_world[:, 2]
    z_data = points_world[:, 1]

    x_mid = 0.5 * (x_data.min() + x_data.max())
    y_mid = 0.5 * (y_data.min() + y_data.max())
    z_mid = 0.5 * (z_data.min() + z_data.max())

    max_range = max(
        np.ptp(x_data),
        np.ptp(y_data),
        np.ptp(z_data),
        1e-6,
    )

    half_range = max_range / 2.0

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)

    ax.set_box_aspect([1, 1, 1])


def set_axes_equal_xyz_display(ax, points_camera):
    """
    Equal scaling for camera plot where display convention is:

    ax x = camera X
    ax y = camera Z
    ax z = camera Y
    """

    if points_camera is None or len(points_camera) == 0:
        return

    x_data = points_camera[:, 0]
    y_data = points_camera[:, 2]
    z_data = points_camera[:, 1]

    x_mid = 0.5 * (x_data.min() + x_data.max())
    y_mid = 0.5 * (y_data.min() + y_data.max())
    z_mid = 0.5 * (z_data.min() + z_data.max())

    max_range = max(
        np.ptp(x_data),
        np.ptp(y_data),
        np.ptp(z_data),
        1e-6,
    )

    half_range = max_range / 2.0

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)

    ax.set_box_aspect([1, 1, 1])
    
    
    
def plot_unknown_surface_voxel_history(
    view_numbers,
    unknown_surface_counts,
    title="Unknown Surface Voxels After Each View",
):
    """
    Plot the number of unknown surface voxels after each view observation.
    """

    if len(view_numbers) == 0 or len(unknown_surface_counts) == 0:
        print("No unknown surface voxel history to plot.")
        return

    plt.figure(figsize=(7, 4))

    plt.plot(
        view_numbers,
        unknown_surface_counts,
        marker="o",
        linewidth=2,
    )

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
    """
    Plot total occupied + unknown volume against true Unity volume
    after each object view.
    """

    if len(view_numbers) == 0 or len(total_unknown_occupied_volumes) == 0:
        print("No volume history to plot.")
        return

    plt.figure(figsize=(7, 4))

    plt.plot(
        view_numbers,
        total_unknown_occupied_volumes,
        marker="o",
        linewidth=2,
        label="Occupied + unknown volume",
    )

    if true_volumes is not None and len(true_volumes) == len(view_numbers):
        plt.plot(
            view_numbers,
            true_volumes,
            linestyle="--",
            marker="s",
            linewidth=2,
            label="True Unity volume",
        )

    plt.title(title)
    plt.xlabel("Object view number")
    plt.ylabel("Volume [m³]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()