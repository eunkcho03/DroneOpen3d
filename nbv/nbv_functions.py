"""
Backward-compatible NBV imports.

Keep this file name if your current main.py still does:
    from nbv_functions import generate_candidate_views_from_bbox, compute_next_best_view

Internally, the functions are now separated into focused modules.
"""

from nbv_utils import (
    normalize_vector,
    rotation_matrix_to_quaternion,
    look_at_quaternion,
    as_points_array,
    _as_points_array,
)
from nbv_candidate_generation import generate_candidate_views_from_bbox
from nbv_surface import (
    estimate_voxel_size_from_points,
    voxel_centers_to_grid_keys,
    extract_surface_voxels,
)
from nbv_projection import (
    compute_camera_basis,
    world_points_to_camera,
    project_points_to_image,
    compute_surface_projection_gain,
)
from nbv_debug_rendering import (
    compute_matplotlib_view_angles,
    compute_equal_axis_limits,
    render_candidate_view_figure,
)
from nbv_planning import compute_next_best_view
