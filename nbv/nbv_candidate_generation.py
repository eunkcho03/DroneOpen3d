import numpy as np

from nbv.nbv_utils import look_at_quaternion


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
