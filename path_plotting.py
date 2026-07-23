import matplotlib.pyplot as plt
import numpy as np
from nbv.nbv_distance import inflate_bbox, inflated_bbox_shortest_path_distance
from itertools import combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_bbox_vertices(bbox_min, bbox_max):
    """
    Return the eight vertices of a 3D axis-aligned bounding box.
    """
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    return np.array(
        [
            [xmin, ymin, zmin],  # 0
            [xmax, ymin, zmin],  # 1
            [xmax, ymax, zmin],  # 2
            [xmin, ymax, zmin],  # 3
            [xmin, ymin, zmax],  # 4
            [xmax, ymin, zmax],  # 5
            [xmax, ymax, zmax],  # 6
            [xmin, ymax, zmax],  # 7
        ],
        dtype=float,
    )


def get_bbox_faces(vertices):
    """
    Return the six faces of a bounding box from its eight vertices.
    """
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]


def draw_bbox_3d(
    ax,
    bbox_min,
    bbox_max,
    face_alpha=0.15,
    edge_linewidth=1.5,
    edge_linestyle="-",
    label=None,
):
    """
    Draw a transparent 3D axis-aligned bounding box.
    """
    vertices = get_bbox_vertices(
        bbox_min,
        bbox_max,
    )

    faces = get_bbox_faces(vertices)

    bbox_collection = Poly3DCollection(
        faces,
        alpha=face_alpha,
        linewidths=edge_linewidth,
        linestyles=edge_linestyle,
        edgecolors="black",
    )

    ax.add_collection3d(bbox_collection)

    # Add an invisible line so the box appears correctly in the legend.
    if label is not None:
        ax.plot(
            [],
            [],
            [],
            linewidth=edge_linewidth,
            linestyle=edge_linestyle,
            label=label,
        )


def set_3d_axes_equal(
    ax,
    x_values,
    y_values,
    z_values,
    padding_factor=0.15,
):
    """
    Set equal visual scaling for a Matplotlib 3D plot.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    z_values = np.asarray(z_values, dtype=float)

    x_min = np.min(x_values)
    x_max = np.max(x_values)

    y_min = np.min(y_values)
    y_max = np.max(y_values)

    z_min = np.min(z_values)
    z_max = np.max(z_values)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)

    maximum_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
        1e-6,
    )

    half_range = 0.5 * maximum_range
    half_range *= 1.0 + padding_factor

    ax.set_xlim(
        x_center - half_range,
        x_center + half_range,
    )

    ax.set_ylim(
        y_center - half_range,
        y_center + half_range,
    )

    ax.set_zlim(
        z_center - half_range,
        z_center + half_range,
    )

    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass

def plot_all_nbv_positions_3d(
    bbox_min,
    bbox_max,
    inflation_factor,
    nbv_waypoints,
    elevation=25,
    azimuth=-55,
    output_file_path=None,
):
    inflated_min, inflated_max = inflate_bbox(
        bbox_min,
        bbox_max,
        inflation_factor,
    )
    nbv_waypoints = np.asarray(nbv_waypoints, dtype=float)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Original bounding box.
    draw_bbox_3d(
        ax=ax,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        face_alpha=0.20,
        edge_linewidth=1.5,
        edge_linestyle="-",
        label="Original bounding box",
    )

    # Inflated bounding box.
    draw_bbox_3d(
        ax=ax,
        bbox_min=inflated_min,
        bbox_max=inflated_max,
        face_alpha=0.05,
        edge_linewidth=2.0,
        edge_linestyle="--",
        label=(
            f"Inflated bounding box "
            f"(factor = {inflation_factor})"
        ),
    )

    # Trajectory through all supplied positions.
    ax.plot(
        nbv_waypoints[:, 0],
        nbv_waypoints[:, 1],
        nbv_waypoints[:, 2],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="NBV trajectory",
        zorder=8,
    )

    # Initial position.
    initial_position = nbv_waypoints[0]

    ax.scatter(
        initial_position[0],
        initial_position[1],
        initial_position[2],
        s=180,
        marker="o",
        depthshade=False,
        label="Initial position",
        zorder=10,
    )

    ax.text(
        initial_position[0],
        initial_position[1],
        initial_position[2],
        "  Waypoint 0",
        fontsize=10,
    )

    # Selected NBV positions.
    for view_number, position in enumerate(
        nbv_waypoints[1:],
        start=1,
    ):
        ax.scatter(
            position[0],
            position[1],
            position[2],
            s=150,
            marker="*",
            depthshade=False,
            zorder=9,
        )

        ax.text(
            position[0],
            position[1],
            position[2],
            f"  Waypoint {view_number}",
            fontsize=9,
        )

    # Equal axis scaling.
    all_x = [
        bbox_min[0],
        bbox_max[0],
        inflated_min[0],
        inflated_max[0],
    ]

    all_y = [
        bbox_min[1],
        bbox_max[1],
        inflated_min[1],
        inflated_max[1],
    ]

    all_z = [
        bbox_min[2],
        bbox_max[2],
        inflated_min[2],
        inflated_max[2],
    ]

    all_x.extend(nbv_waypoints[:, 0].tolist())
    all_y.extend(nbv_waypoints[:, 1].tolist())
    all_z.extend(nbv_waypoints[:, 2].tolist())

    set_3d_axes_equal(
        ax=ax,
        x_values=all_x,
        y_values=all_y,
        z_values=all_z,
        padding_factor=0.15,
    )

    ax.set_title(
        "Next-Best-View Trajectory\n"
        f"Number of views = {len(nbv_waypoints)}"
    )

    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.set_zlabel("Z position [m]")

    ax.view_init(
        elev=elevation,
        azim=azimuth,
    )

    ax.legend(loc="best")

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