import numpy as np
import heapq
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

def segment_intersects_rect(p1, p2, rect):
    xmin, xmax, zmin, zmax = rect
    x1, z1 = p1
    x2, z2 = p2

    # Quick bounding box overlap test
    if max(x1, x2) <= xmin or min(x1, x2) >= xmax or max(z1, z2) <= zmin or min(z1, z2) >= zmax:
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


def compute_distance(
    start_position,
    end_position,
    b_min,
    b_max,
    fac=1.2,
):
    # Extract 2D coordinates (X, Z)
    p_start = np.array([start_position[0], start_position[2]], dtype=float)
    p_end = np.array([end_position[0], end_position[2]], dtype=float)

    # Inflate BBox in 2D
    center = (b_min[[0, 2]] + b_max[[0, 2]]) * 0.5
    half_size = (b_max[[0, 2]] - b_min[[0, 2]]) * 0.5 * fac
    rect_min = center - half_size
    rect_max = center + half_size

    rect = (rect_min[0], rect_max[0], rect_min[1], rect_max[1])

    # 1. Direct path check
    if not segment_intersects_rect(p_start, p_end, rect):
        horiz_dist = np.hypot(*(p_end - p_start))
    else:
        # 2. Obstructed: evaluate candidate corner paths
        # Corners: 0:(xmin, zmin), 1:(xmin, zmax), 2:(xmax, zmin), 3:(xmax, zmax)
        corners = [
            np.array([rect[0], rect[2]]),
            np.array([rect[0], rect[3]]),
            np.array([rect[1], rect[2]]),
            np.array([rect[1], rect[3]]),
        ]

        best_dist = float("inf")

        # Check single-corner detour paths
        for c in corners:
            if not segment_intersects_rect(
                p_start, c, rect
            ) and not segment_intersects_rect(c, p_end, rect):
                d = np.hypot(*(c - p_start)) + np.hypot(*(p_end - c))
                if d < best_dist:
                    best_dist = d

        # Check two-corner detour paths (around adjacent sides)
        corner_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in corner_pairs:
            c1, c2 = corners[i], corners[j]
            # Try both orientations for the pair
            for c_first, c_second in [(c1, c2), (c2, c1)]:
                if (
                    not segment_intersects_rect(p_start, c_first, rect)
                    and not segment_intersects_rect(c_first, c_second, rect)
                    and not segment_intersects_rect(c_second, p_end, rect)
                ):

                    d = (
                        np.hypot(*(c_first - p_start))
                        + np.hypot(*(c_second - c_first))
                        + np.hypot(*(p_end - c_second))
                    )
                    if d < best_dist:
                        best_dist = d

        horiz_dist = best_dist

    total_distance = horiz_dist + abs(end_position[1] - start_position[1])

    return total_distance
