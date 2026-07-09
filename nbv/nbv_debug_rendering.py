import numpy as np
import matplotlib.pyplot as plt

from nbv_utils import as_points_array


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
    points = as_points_array(points)

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

    occupied_voxels = as_points_array(occupied_voxels)
    unknown_voxels = as_points_array(unknown_voxels)
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
