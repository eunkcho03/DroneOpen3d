import numpy as np
import heapq
import matplotlib.pyplot as plt

from itertools import combinations
from matplotlib.patches import Rectangle

def inflate_bbox(bbox_min, bbox_max, fac):
    center = (bbox_min + bbox_max) / 2.0
    half_size = (bbox_max - bbox_min) / 2.0
    inflated_half_size = fac * half_size
    return center - inflated_half_size, center + inflated_half_size

def unpack_2d_points(p1, p2):
    x1, z1 = p1
    x2, z2 = p2
    return x1, z1, x2, z2


def segment_intersects_rect_interior(p1, p2, rect):
    xmin, xmax, zmin, zmax = rect
    x1, z1, x2, z2 = unpack_2d_points(p1, p2)

    if (
        max(x1, x2) <= xmin
        or min(x1, x2) >= xmax
        or max(z1, z2) <= zmin
        or min(z1, z2) >= zmax
    ):
        return False

    dx = x2 - x1
    dz = z2 - z1

    p = [-dx, dx, -dz, dz]
    q = [x1 - xmin, xmax - x1, z1 - zmin, zmax - z1]

    u1, u2 = 0.0, 1.0

    for i in range(4):
        if p[i] == 0:
            if q[i] <= 0:
                return False
        else:
            t = q[i] / p[i]

            if p[i] < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)

    return u1 < u2


def inflated_bbox_shortest_path_distance(
    start_position,
    end_position,
    b_min,
    b_max,
    fac=1.2,
):
    b_min, b_max = inflate_bbox(b_min, b_max, fac)

    rect = (b_min[0], b_max[0], b_min[2], b_max[2])

    pts = [
        start_position[[0, 2]],
        end_position[[0, 2]],
    ] + [
        np.array([x, z])
        for x in rect[:2]
        for z in rect[2:]
    ]

    graph = {i: [] for i in range(6)}

    for (i, p1), (j, p2) in combinations(enumerate(pts), 2):
        if not segment_intersects_rect_interior(p1, p2, rect):
            distance = np.linalg.norm(p1 - p2)
            graph[i].append((j, distance))
            graph[j].append((i, distance))

    dist = {0: 0.0}
    prev = {}
    queue = [(0.0, 0)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == 1:
            break

        if current_distance > dist.get(current_node, float("inf")):
            continue

        for neighbour, edge_distance in graph[current_node]:
            new_distance = current_distance + edge_distance

            if new_distance < dist.get(neighbour, float("inf")):
                dist[neighbour] = new_distance
                prev[neighbour] = current_node
                heapq.heappush(queue, (new_distance, neighbour))

    if 1 not in dist:
        return float("inf"), []

    path_2d = []
    current_node = 1

    while current_node is not None:
        path_2d.append(pts[current_node])
        current_node = prev.get(current_node)

    path_2d.reverse()

    path_3d = [
        np.array([point[0], start_position[1], point[1]])
        for point in path_2d
    ]

    # Add the actual end position if the height changes.
    if not np.isclose(start_position[1], end_position[1]):
        path_3d.append(np.asarray(end_position))

    horizontal_distance = dist[1]
    vertical_distance = abs(end_position[1] - start_position[1])

    total_distance = horizontal_distance + vertical_distance

    return total_distance, path_3d


def plot_inflated_bbox_path(
    start_position,
    end_position,
    bbox_min,
    bbox_max,
    path,
    fac=1.2,
    total_distance=None,
    show_direct_path=True,
):
    start_position = np.asarray(start_position, dtype=float)
    end_position = np.asarray(end_position, dtype=float)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    inflated_min, inflated_max = inflate_bbox(
        bbox_min,
        bbox_max,
        fac,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Original bounding box.
    original_width = bbox_max[0] - bbox_min[0]
    original_depth = bbox_max[2] - bbox_min[2]

    original_rectangle = Rectangle(
        (bbox_min[0], bbox_min[2]),
        original_width,
        original_depth,
        fill=True,
        alpha=0.25,
        label="Original bounding box",
    )

    ax.add_patch(original_rectangle)

    # Inflated bounding box.
    inflated_width = inflated_max[0] - inflated_min[0]
    inflated_depth = inflated_max[2] - inflated_min[2]

    inflated_rectangle = Rectangle(
        (inflated_min[0], inflated_min[2]),
        inflated_width,
        inflated_depth,
        fill=False,
        linewidth=2.0,
        linestyle="--",
        label=f"Inflated bounding box (factor = {fac})",
    )

    ax.add_patch(inflated_rectangle)

    # Direct start-to-end line for comparison.
    if show_direct_path:
        ax.plot(
            [start_position[0], end_position[0]],
            [start_position[2], end_position[2]],
            linestyle=":",
            linewidth=1.5,
            label="Direct path",
        )

    # Plot the computed path.
    if path:
        path_array = np.asarray(path, dtype=float)

        # Only X and Z are needed for the top-down plot.
        path_x = path_array[:, 0]
        path_z = path_array[:, 2]

        ax.plot(
            path_x,
            path_z,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label="Shortest collision-free path",
        )

        # Label each waypoint.
        for index, point in enumerate(path_array):
            ax.annotate(
                f"P{index}",
                xy=(point[0], point[2]),
                xytext=(7, 7),
                textcoords="offset points",
            )

    # Start and end markers.
    ax.scatter(
        start_position[0],
        start_position[2],
        s=120,
        marker="o",
        label="Start",
        zorder=5,
    )

    ax.scatter(
        end_position[0],
        end_position[2],
        s=140,
        marker="*",
        label="End",
        zorder=5,
    )

    # Add some space around all geometry.
    all_x = [
        start_position[0],
        end_position[0],
        bbox_min[0],
        bbox_max[0],
        inflated_min[0],
        inflated_max[0],
    ]

    all_z = [
        start_position[2],
        end_position[2],
        bbox_min[2],
        bbox_max[2],
        inflated_min[2],
        inflated_max[2],
    ]

    x_range = max(all_x) - min(all_x)
    z_range = max(all_z) - min(all_z)
    margin = 0.15 * max(x_range, z_range, 1.0)

    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    title = "Shortest Path Around an Inflated Bounding Box"

    if total_distance is not None:
        title += f"\nTotal distance = {total_distance:.3f} m"

    ax.set_title(title)
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Z position [m]")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()
