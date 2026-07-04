import numpy as np


# ============================================================
# Quaternion utilities
# ============================================================

def normalize_vector(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < eps:
        return np.zeros_like(v)

    return v / norm


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].

    This format matches Unity's Quaternion convention:
        x, y, z, w
    """

    R = np.asarray(R, dtype=float)

    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)

    return q


def look_at_quaternion(camera_position, target_position, world_up=np.array([0.0, 1.0, 0.0])):
    """
    Compute quaternion [x, y, z, w] so that the camera faces the target.

    Assumption:
        - Unity-style coordinate system
        - Camera/object forward direction is local +Z
        - Y is world up
    """

    camera_position = np.asarray(camera_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = normalize_vector(target_position - camera_position)

    if np.linalg.norm(forward) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])

    right = np.cross(world_up, forward)

    # If camera is almost exactly above/below object, world_up becomes unstable.
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)

    right = normalize_vector(right)
    up = normalize_vector(np.cross(forward, right))

    # Columns are local x=right, y=up, z=forward
    R = np.column_stack((right, up, forward))

    return rotation_matrix_to_quaternion(R)


# ============================================================
# Input hardening helpers
# ============================================================

def _as_points_array(points):
    """
    Coerce input to an (N, 3) float array.

    Accepts:
        - (N, 3) arrays
        - a single point of shape (3,)  -> becomes (1, 3)
        - empty input                   -> becomes (0, 3)
    """
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return points.reshape(0, 3)

    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(f"Expected a 3D point, got shape {points.shape}")
        return points.reshape(1, 3)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) array, got shape {points.shape}")

    return points


# ============================================================
# Candidate view generation
# ============================================================

def generate_candidate_views_from_bbox(
    bbox_min,
    bbox_max,
    num_views=8,
    distance_factor=1.5,
    height_fraction=0.6,
    min_radius=1e-6,
):
    """
    Generate fixed candidate viewpoints around the object bounding box.

    Each candidate view includes:
        - view_id
        - angle_deg
        - position
        - look_at
        - rotation_quat: quaternion [x, y, z, w] that faces the object
    """

    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    if np.any(bbox_max < bbox_min):
        raise ValueError("bbox_max must be >= bbox_min in every axis")

    center = 0.5 * (bbox_min + bbox_max)
    size = bbox_max - bbox_min

    # Horizontal radius needed to be outside bbox footprint
    half_diag_xz = 0.5 * np.sqrt(size[0] ** 2 + size[2] ** 2)
    radius = half_diag_xz * distance_factor

    # Degenerate bbox (a point, or a vertical line): fail loudly instead of
    # silently stacking all cameras on the object center.
    if radius < min_radius:
        raise ValueError(
            "Bounding box has no horizontal extent; candidate camera radius "
            f"would be {radius}. Provide a valid bbox or a manual radius."
        )

    cam_y = bbox_min[1] + height_fraction * size[1]

    candidate_views = []

    for i in range(num_views):
        angle = 2.0 * np.pi * i / num_views

        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            cam_y,
            center[2] + radius * np.sin(angle),
        ])

        look_at = center.copy()

        rotation_quat = look_at_quaternion(
            camera_position=cam_pos,
            target_position=look_at,
        )

        candidate_views.append({
            "view_id": i,
            "angle_deg": np.degrees(angle),
            "position": cam_pos,
            "look_at": look_at,
            "rotation_quat": rotation_quat,
        })

    return candidate_views


# ============================================================
# Unknown surface voxel extraction
# ============================================================

def extract_outer_unknown_voxels(unknown_voxels, voxel_size, grid_eps=1e-6):
    """
    Extract only boundary/outer unknown voxels.

    A voxel is considered outer if at least one of its 6 direct neighbors
    is not also unknown.

    Input:
        unknown_voxels: Nx3 array of voxel center positions in world coordinates
        voxel_size: voxel size in meters
        grid_eps: relative tolerance for float jitter at cell boundaries

    Output:
        outer_unknown_voxels: Mx3 array of outer unknown voxel centers

    Notes on quantization:
        Uses floor(), NOT round(). Voxel centers typically sit at half-cell
        offsets (i + 0.5) * voxel_size; np.round() applies half-to-even
        ("banker's") rounding, which maps adjacent centers 0.5 -> 0 and
        1.5 -> 2, breaking all neighbor lookups and classifying nearly
        every voxel as outer. floor() maps any position inside a cell to
        that cell's index, which is what we want.
    """

    if voxel_size is None or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"voxel_size must be a positive finite number, got {voxel_size}")

    unknown_voxels = _as_points_array(unknown_voxels)

    if len(unknown_voxels) == 0:
        return unknown_voxels

    if not np.all(np.isfinite(unknown_voxels)):
        raise ValueError("unknown_voxels contains NaN or inf values")

    # Quantize to integer grid cells. Small epsilon absorbs float jitter
    # for positions sitting almost exactly on a cell boundary.
    grid = np.floor(unknown_voxels / voxel_size + grid_eps).astype(np.int64)

    # Shift to non-negative so we can pack (x, y, z) into a single int64
    # for fast vectorized membership tests. +1 margin so that neighbor
    # offsets of -1 never underflow the encoding.
    g = grid - grid.min(axis=0) + 1
    dims = g.max(axis=0) + 2  # +2 margin for +1 neighbor offsets

    def encode(cells):
        return (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]

    codes = np.sort(encode(g))

    neighbor_offsets = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.int64)

    is_outer = np.zeros(len(g), dtype=bool)

    for offset in neighbor_offsets:  # only 6 iterations; inner work is vectorized
        neighbor_codes = encode(g + offset)
        idx = np.searchsorted(codes, neighbor_codes)
        idx = np.clip(idx, 0, len(codes) - 1)
        # Neighbor missing from the unknown set -> this voxel is on the boundary
        is_outer |= codes[idx] != neighbor_codes

    return unknown_voxels[is_outer]


# ============================================================
# Current-position-aware NBV
# ============================================================

def compute_next_best_view(
    unknown_voxels,
    object_center,
    candidate_views,
    current_camera_position=None,
    voxel_size=None,
    use_outer_unknown_only=True,
    visited_view_ids=None,
    lambda_distance=1.0,
    distance_penalty_mode="ratio",
):
    """
    Compute the next best view.

    The score is based on:
        1. Number of visible unknown outer voxels
        2. Distance from the current camera position

    Recommended scoring:

        score = information_gain / (1 + lambda_distance * travel_distance)

    Inputs:
        unknown_voxels:
            Nx3 array of unknown voxel centers.

        object_center:
            3D position of object center.

        candidate_views:
            List of views generated by generate_candidate_views_from_bbox.
            NOTE: view dicts are annotated in place with "gain", "distance",
            and "score", and their "look_at"/"rotation_quat" are refreshed
            to face object_center.

        current_camera_position:
            Current camera/drone position.
            If None, distance penalty is not applied.

        voxel_size:
            Required if use_outer_unknown_only=True.

        use_outer_unknown_only:
            If True, only outer unknown voxels are used for scoring.

        visited_view_ids:
            Views already visited. These are skipped.

        lambda_distance:
            Weight for travel distance penalty. Must be >= 0 when
            distance_penalty_mode is "ratio".

        distance_penalty_mode:
            "ratio":
                score = gain / (1 + lambda_distance * distance)

            "subtract":
                score = gain - lambda_distance * distance

            "none":
                score = gain

    Output:
        (best_view, best_score)

        best_view is None (with best_score 0.0) when there is nothing to do:
        no unknown voxels, no scoring voxels, no candidate views, or all
        candidates already visited. Callers MUST handle the None case.
    """

    if distance_penalty_mode not in ("ratio", "subtract", "none"):
        raise ValueError(
            "distance_penalty_mode must be 'ratio', 'subtract', or 'none'"
        )

    if distance_penalty_mode == "ratio" and lambda_distance < 0:
        raise ValueError(
            "lambda_distance must be >= 0 in 'ratio' mode "
            "(a negative value can zero or flip the denominator)"
        )

    unknown_voxels = _as_points_array(unknown_voxels)
    object_center = np.asarray(object_center, dtype=float)

    if current_camera_position is not None:
        current_camera_position = np.asarray(current_camera_position, dtype=float)

    if visited_view_ids is None:
        visited_view_ids = set()
    else:
        visited_view_ids = set(visited_view_ids)

    if len(unknown_voxels) == 0:
        return None, 0.0

    # -------------------------------------------------
    # Select voxels used for scoring
    # -------------------------------------------------
    if use_outer_unknown_only:
        if voxel_size is None:
            raise ValueError(
                "voxel_size must be provided when use_outer_unknown_only=True"
            )

        scoring_voxels = extract_outer_unknown_voxels(
            unknown_voxels=unknown_voxels,
            voxel_size=voxel_size,
        )
    else:
        scoring_voxels = unknown_voxels

    if len(scoring_voxels) == 0:
        return None, 0.0

    # Precompute once: direction of every scoring voxel relative to the
    # object center. This turns the per-view gain computation into a single
    # matrix-vector product instead of a Python loop over all voxels.
    voxel_dirs = scoring_voxels - object_center  # (M, 3)

    best_score = -np.inf
    best_view = None

    # -------------------------------------------------
    # Score each candidate view
    # -------------------------------------------------
    for view in candidate_views:

        view_id = view["view_id"]

        # Skip already visited views
        if view_id in visited_view_ids:
            view["gain"] = 0
            view["distance"] = np.inf
            view["score"] = -np.inf
            continue

        camera_position = np.asarray(view["position"], dtype=float)

        # Make sure the candidate always faces the object
        view["look_at"] = object_center.copy()
        view["rotation_quat"] = look_at_quaternion(
            camera_position=camera_position,
            target_position=object_center,
        )

        # -------------------------------------------------
        # Information gain approximation (vectorized)
        #
        # A voxel counts if it lies in the hemisphere facing this camera:
        # dot(voxel - center, camera - center) > 0
        # -------------------------------------------------
        center_to_camera = camera_position - object_center
        gain = int(np.count_nonzero(voxel_dirs @ center_to_camera > 0.0))

        # -------------------------------------------------
        # Distance penalty from current camera position
        # -------------------------------------------------
        if current_camera_position is None:
            travel_distance = 0.0
        else:
            travel_distance = float(
                np.linalg.norm(camera_position - current_camera_position)
            )

        if distance_penalty_mode == "ratio":
            score = gain / (1.0 + lambda_distance * travel_distance)
        elif distance_penalty_mode == "subtract":
            score = gain - lambda_distance * travel_distance
        else:  # "none"
            score = gain

        view["gain"] = gain
        view["distance"] = travel_distance
        view["score"] = score

        if score > best_score:
            best_score = score
            best_view = view

    # All candidates visited (or no candidates at all): consistent None return
    if best_view is None:
        return None, 0.0

    return best_view, best_score