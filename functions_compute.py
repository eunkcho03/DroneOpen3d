import numpy as np
import cv2

DEPTH_MIN_VALID = 0.0
DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01

# Dense voxel-state values
FREE = 0
UNKNOWN = 1
OCCUPIED = 2


def make_valid_depth_mask(depth, far):
    depth = np.asarray(depth)
    return (
        np.isfinite(depth)
        & (depth > DEPTH_MIN_VALID)
        & (depth != DEPTH_INVALID_VALUE)
        & (depth < far - FAR_PLANE_MARGIN)
    )


def camera_intrinsics(width, height, fov_degrees):
    fy = height / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3)

    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def depth_to_camera_points(depth, fov_degrees, far, keep_invalid_depth=False):
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)

    if keep_invalid_depth:
        valid_mask = np.ones(depth.shape, dtype=bool)
    else:
        valid_mask = make_valid_depth_mask(depth, far)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = -(v[valid_mask] - cy) * z / fy

    points_camera = np.column_stack((x, y, z))
    return points_camera, valid_mask


def camera_to_world(points_camera, position, quaternion):
    points_camera = np.asarray(points_camera, dtype=float)
    if len(points_camera) == 0:
        return np.empty((0, 3))

    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return points_camera @ R.T + position


def world_to_camera(points_world, position, quaternion):
    points_world = np.asarray(points_world, dtype=float)
    if len(points_world) == 0:
        return np.empty((0, 3))

    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return (points_world - position) @ R


class DepthFrameProcessor:
    def __init__(
        self,
        depth,
        fov_degrees,
        far,
        position,
        quaternion,
        floor_height=0.0,
        height_threshold=0.02,
        verbose=True,
    ):
        self.depth = depth
        self.fov_degrees = fov_degrees
        self.far = far
        self.position = position
        self.quaternion = quaternion
        self.floor_height = floor_height
        self.height_threshold = height_threshold
        self.verbose = verbose

        self.points_camera = None
        self.points_world = None
        self.filtered_points = None
        self.valid_mask = None

    def run(self):
        self.points_camera, self.valid_mask = depth_to_camera_points(
            self.depth,
            self.fov_degrees,
            self.far,
        )
        self.points_world = camera_to_world(
            self.points_camera,
            self.position,
            self.quaternion,
        )
        self.filtered_points = self._filter_by_height(self.points_world)
        return self

    def _filter_by_height(self, points_world):
        if points_world is None or len(points_world) == 0:
            return np.empty((0, 3))

        min_height = self.floor_height + self.height_threshold
        filtered = points_world[points_world[:, 1] > min_height]

        if self.verbose:
            print(
                f"Height filter | input: {len(points_world)} | "
                f"filtered: {len(filtered)} | min Y: {min_height:.3f}"
            )

        return filtered


class ExtrusionFusionReconstruction:
    """
    Faster dense-grid version of your previous reconstruction.

    Main change:
        old: voxel_prob dict + occupied/free/unknown Python sets
        new: voxel_state 3D NumPy array

    State encoding:
        FREE     = 0
        UNKNOWN  = 1
        OCCUPIED = 2

    The public attributes are kept similar:
        occupied_centers, unknown_centers, free_centers, observed_centers
        occupied_volume, unknown_volume, free_volume, observed_volume
        occupied_indices, unknown_indices, free_indices, observed_indices

    Note:
        *_indices are now NumPy arrays with shape (N, 3), not Python sets.
        len(reconstruction.unknown_indices) still works.
    """

    def __init__(
        self,
        voxel_size=0.01,
        bbox_margin=0.05,
        floor_height=0.0,
        max_points=None,
        verbose=True,
        update_center_outputs=True,
        max_centers_to_store=None,
    ):
        self.voxel_size = voxel_size
        self.bbox_margin = bbox_margin
        self.floor_height = floor_height
        self.max_points = max_points
        self.verbose = verbose

        # If plotting becomes slow, set update_center_outputs=False or limit with max_centers_to_store.
        self.update_center_outputs = update_center_outputs
        self.max_centers_to_store = max_centers_to_store

        self.initialized = False
        self.bbox_min = None
        self.bbox_max = None
        self.bbox_corners = None
        self.grid_shape = None

        # Kept for compatibility with old code/logging.
        self.free_prob = 0.0
        self.unknown_prob = 0.5
        self.occupied_prob = 1.0
        self.occupied_threshold = 0.75

        # Dense state arrays.
        self.voxel_state = None       # uint8 grid: FREE / UNKNOWN / OCCUPIED
        self.observed_mask = None     # bool grid: has this voxel ever been observed as surface?

        # Compatibility attributes. These are NumPy arrays, not sets.
        self.occupied_indices = np.empty((0, 3), dtype=int)
        self.observed_indices = np.empty((0, 3), dtype=int)
        self.free_indices = np.empty((0, 3), dtype=int)
        self.unknown_indices = np.empty((0, 3), dtype=int)

        self.observed_centers = np.empty((0, 3))
        self.occupied_centers = np.empty((0, 3))
        self.free_centers = np.empty((0, 3))
        self.unknown_centers = np.empty((0, 3))

        self.observed_volume = 0.0
        self.occupied_volume = 0.0
        self.free_volume = 0.0
        self.unknown_volume = 0.0
        self.expected_volume = 0.0

    def add_view(
        self,
        points_world,
        depth=None,
        fov_degrees=None,
        far=None,
        position=None,
        quaternion=None,
        carve_margin=None,
        carve_with_invalid_depth=False,
    ):
        points_world = self._validate_points(points_world)

        if not self.initialized:
            if len(points_world) == 0:
                self._log("Skipping initialization: no valid object points.")
                return self
            self._initialize_bbox_from_detection(points_world)
            

        if self._has_camera_data(depth, fov_degrees, far, position, quaternion):
            self.carve_with_depth_image(
                depth=depth,
                fov_degrees=fov_degrees,
                far=far,
                position=position,
                quaternion=quaternion,
                carve_margin=carve_margin,
                carve_with_invalid_depth=carve_with_invalid_depth,
            )
        else:
            self._log("Skipping carving: missing depth/camera information.")

        new_observed_indices = self._voxelize(points_world)
        before_observed = int(np.count_nonzero(self.observed_mask))

        self._mark_observed_occupied(new_observed_indices)

        after_observed = int(np.count_nonzero(self.observed_mask))
        self._update_outputs()
        self._log_status("Processed view", new_observed=after_observed - before_observed)
        return self

    def carve_with_depth_image(
        self,
        depth,
        fov_degrees,
        far,
        position,
        quaternion,
        carve_margin=None,
        carve_with_invalid_depth=False,
        keep_invalid_depth=False,
    ):
        if not self.initialized or self.voxel_state is None:
            return self

        if carve_margin is None:
            carve_margin = 0.5 * self.voxel_size

        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2:
            return self

        height, width = depth.shape
        fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)

        if keep_invalid_depth:
            valid_depth_mask = np.ones(depth.shape, dtype=bool)
        else:
            valid_depth_mask = make_valid_depth_mask(depth, far)

        # Only project voxels that are not already free.
        active_indices = np.argwhere(self.voxel_state != FREE)
        if len(active_indices) == 0:
            return self

        centers_world = self._indices_to_centers_from_array(active_indices)
        centers_camera = world_to_camera(centers_world, position, quaternion)

        x = centers_camera[:, 0]
        y = centers_camera[:, 1]
        z = centers_camera[:, 2]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(x * fx / z + cx).astype(int)
            v = np.round(cy - y * fy / z).astype(int)

        projects_inside_image = (
            (z > 0.0)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )

        if not np.any(projects_inside_image):
            return self

        candidate_indices = active_indices[projects_inside_image]
        candidate_z = z[projects_inside_image]
        candidate_u = u[projects_inside_image]
        candidate_v = v[projects_inside_image]

        measured_depth = depth[candidate_v, candidate_u]
        valid_measured_depth = valid_depth_mask[candidate_v, candidate_u]

        carve_free_space = valid_measured_depth & (candidate_z < measured_depth - carve_margin)

        if carve_with_invalid_depth:
            carve_free_space |= (~valid_measured_depth) & (candidate_z < far - carve_margin)

        carve_indices = candidate_indices[carve_free_space]
        before_free = int(np.count_nonzero(self.voxel_state == FREE))

        if len(carve_indices) > 0:
            ix = carve_indices[:, 0]
            iy = carve_indices[:, 1]
            iz = carve_indices[:, 2]

            # Do not erase voxels that have been directly observed as object surface.
            can_carve = ~self.observed_mask[ix, iy, iz]
            ix = ix[can_carve]
            iy = iy[can_carve]
            iz = iz[can_carve]

            self.voxel_state[ix, iy, iz] = FREE

        # Safety: observed surface voxels always stay occupied.
        self.voxel_state[self.observed_mask] = OCCUPIED

        self._update_outputs()

        after_free = int(np.count_nonzero(self.voxel_state == FREE))
        newly_free = after_free - before_free

        self._log(
            f"Depth carving | projected voxels: {len(candidate_indices)} | "
            f"newly free: {newly_free} | "
            f"occupied volume: {self.occupied_volume:.6f} m³"
        )
        return self

    def get_unknown_surface_indices(self):
        """
        Return unknown voxels that touch free space in a 6-neighborhood.

        Faster than the old set-neighbor loop because it uses boolean slicing.
        Returned type is a set of (ix, iy, iz) tuples for compatibility.
        """
        if not self.initialized or self.voxel_state is None:
            return set()

        unknown = self.voxel_state == UNKNOWN
        free = self.voxel_state == FREE

        if not np.any(unknown) or not np.any(free):
            return set()

        surface = np.zeros(self.grid_shape, dtype=bool)

        # Neighbor in +/- x
        surface[1:, :, :] |= unknown[1:, :, :] & free[:-1, :, :]
        surface[:-1, :, :] |= unknown[:-1, :, :] & free[1:, :, :]

        # Neighbor in +/- y
        surface[:, 1:, :] |= unknown[:, 1:, :] & free[:, :-1, :]
        surface[:, :-1, :] |= unknown[:, :-1, :] & free[:, 1:, :]

        # Neighbor in +/- z
        surface[:, :, 1:] |= unknown[:, :, 1:] & free[:, :, :-1]
        surface[:, :, :-1] |= unknown[:, :, :-1] & free[:, :, 1:]

        indices = np.argwhere(surface)
        return {tuple(idx) for idx in indices}

    def count_unknown_surface_voxels(self):
        if not self.initialized or self.voxel_state is None:
            return 0

        unknown = self.voxel_state == UNKNOWN
        free = self.voxel_state == FREE

        if not np.any(unknown) or not np.any(free):
            return 0

        surface = np.zeros(self.grid_shape, dtype=bool)

        surface[1:, :, :] |= unknown[1:, :, :] & free[:-1, :, :]
        surface[:-1, :, :] |= unknown[:-1, :, :] & free[1:, :, :]

        surface[:, 1:, :] |= unknown[:, 1:, :] & free[:, :-1, :]
        surface[:, :-1, :] |= unknown[:, :-1, :] & free[:, 1:, :]

        surface[:, :, 1:] |= unknown[:, :, 1:] & free[:, :, :-1]
        surface[:, :, :-1] |= unknown[:, :, :-1] & free[:, :, 1:]

        return int(np.count_nonzero(surface))

    def get_unknown_surface_centers(self):
        surface_indices = np.array(list(self.get_unknown_surface_indices()), dtype=int)
        if len(surface_indices) == 0:
            return np.empty((0, 3))
        return self._indices_to_centers_from_array(surface_indices)

    def _initialize_bbox_from_detection(self, points_world):
        self.bbox_min, self.bbox_max, self.bbox_corners = self._compute_bbox(points_world)
        
        bbox_size = self.bbox_max - self.bbox_min
        bbox_volume = np.prod(bbox_size)

        self._log(
            f"Initial bbox | "
            f"min: {self.bbox_min} | "
            f"max: {self.bbox_max} | "
            f"size: {bbox_size} m | "
            f"volume: {bbox_volume:.6f} m³ | "
            f"voxel size: {self.voxel_size} m"
        )
    
        self.grid_shape = self._compute_grid_shape()

        # Fill whole bbox as UNKNOWN with one dense array allocation.
        self.voxel_state = np.full(self.grid_shape, UNKNOWN, dtype=np.uint8)
        self.observed_mask = np.zeros(self.grid_shape, dtype=bool)

        observed_indices = self._voxelize(points_world)
        self._mark_observed_occupied(observed_indices)

        self.initialized = True
        self._update_outputs()
        self._log_status("Initialized dense bbox from first object detection")
        return self

    def _mark_observed_occupied(self, indices):
        if indices is None or len(indices) == 0:
            return

        indices = np.asarray(indices, dtype=int)
        ix = indices[:, 0]
        iy = indices[:, 1]
        iz = indices[:, 2]

        self.observed_mask[ix, iy, iz] = True
        self.voxel_state[ix, iy, iz] = OCCUPIED

    def _validate_points(self, points_world):
        if points_world is None:
            return np.empty((0, 3))

        points_world = np.asarray(points_world, dtype=float)

        if len(points_world) == 0:
            return np.empty((0, 3))

        if self.max_points is not None and len(points_world) > self.max_points:
            selected = np.random.choice(len(points_world), self.max_points, replace=False)
            points_world = points_world[selected]

        return points_world

    def _compute_bbox(self, points_world):
        bbox_min = points_world.min(axis=0).copy()
        bbox_max = points_world.max(axis=0).copy()

        # Same behavior as your previous code:
        # add margin in x/z only, and force bottom to floor height.
        bbox_min[[0, 2]] -= self.bbox_margin
        bbox_max[[0, 2]] += self.bbox_margin
        bbox_min[1] = self.floor_height

        mn_x, mn_y, mn_z = bbox_min
        mx_x, mx_y, mx_z = bbox_max

        corners = np.array([
            [mn_x, mn_y, mn_z],
            [mx_x, mn_y, mn_z],
            [mx_x, mn_y, mx_z],
            [mn_x, mn_y, mx_z],
            [mn_x, mx_y, mn_z],
            [mx_x, mx_y, mn_z],
            [mx_x, mx_y, mx_z],
            [mn_x, mx_y, mx_z],
        ])

        return bbox_min, bbox_max, corners

    def _compute_grid_shape(self):
        return np.maximum(
            np.ceil((self.bbox_max - self.bbox_min) / self.voxel_size).astype(int),
            1,
        )

    def _voxelize(self, points_world):
        """
        Convert world points to unique integer voxel indices.

        Returns:
            indices: NumPy array with shape (N, 3), dtype int
        """
        if self.bbox_min is None or self.grid_shape is None:
            return np.empty((0, 3), dtype=int)

        points_world = np.asarray(points_world, dtype=float)
        if len(points_world) == 0:
            return np.empty((0, 3), dtype=int)

        indices = np.floor((points_world - self.bbox_min) / self.voxel_size).astype(int)

        inside = np.all((indices >= 0) & (indices < self.grid_shape), axis=1)
        indices = indices[inside]

        if len(indices) == 0:
            return np.empty((0, 3), dtype=int)

        return np.unique(indices, axis=0)

    def _indices_to_centers(self, indices):
        if indices is None or len(indices) == 0:
            return np.empty((0, 3))
        return self._indices_to_centers_from_array(indices)

    def _indices_to_centers_from_array(self, indices):
        indices = np.asarray(indices, dtype=float)
        return self.bbox_min + (indices + 0.5) * self.voxel_size

    def _limited_indices_from_mask(self, mask):
        indices = np.argwhere(mask)
        if self.max_centers_to_store is not None and len(indices) > self.max_centers_to_store:
            selected = np.random.choice(len(indices), self.max_centers_to_store, replace=False)
            indices = indices[selected]
        return indices

    def _update_outputs(self):
        if self.voxel_state is None:
            return

        voxel_volume = self.voxel_size ** 3

        occupied_mask = self.voxel_state == OCCUPIED
        unknown_mask = self.voxel_state == UNKNOWN
        free_mask = self.voxel_state == FREE

        observed_count = int(np.count_nonzero(self.observed_mask))
        occupied_count = int(np.count_nonzero(occupied_mask))
        unknown_count = int(np.count_nonzero(unknown_mask))
        free_count = int(np.count_nonzero(free_mask))

        self.observed_volume = observed_count * voxel_volume
        self.occupied_volume = occupied_count * voxel_volume
        self.unknown_volume = unknown_count * voxel_volume
        self.free_volume = free_count * voxel_volume

        # Same meaning as old sum(voxel_prob.values()) * voxel_volume:
        # occupied contributes 1.0, unknown contributes 0.5, free contributes 0.0.
        self.expected_volume = (occupied_count + 0.5 * unknown_count) * voxel_volume

        if not self.update_center_outputs:
            self.observed_indices = np.empty((0, 3), dtype=int)
            self.occupied_indices = np.empty((0, 3), dtype=int)
            self.unknown_indices = np.empty((0, 3), dtype=int)
            self.free_indices = np.empty((0, 3), dtype=int)
            self.observed_centers = np.empty((0, 3))
            self.occupied_centers = np.empty((0, 3))
            self.unknown_centers = np.empty((0, 3))
            self.free_centers = np.empty((0, 3))
            return

        self.observed_indices = self._limited_indices_from_mask(self.observed_mask)
        self.occupied_indices = self._limited_indices_from_mask(occupied_mask)
        self.unknown_indices = self._limited_indices_from_mask(unknown_mask)
        self.free_indices = self._limited_indices_from_mask(free_mask)

        self.observed_centers = self._indices_to_centers(self.observed_indices)
        self.occupied_centers = self._indices_to_centers(self.occupied_indices)
        self.unknown_centers = self._indices_to_centers(self.unknown_indices)
        self.free_centers = self._indices_to_centers(self.free_indices)

    @staticmethod
    def _has_camera_data(depth, fov_degrees, far, position, quaternion):
        return all(v is not None for v in [depth, fov_degrees, far, position, quaternion])

    def _log(self, message):
        if self.verbose:
            print(message)

    def _log_status(self, title, new_observed=None):
        if not self.verbose:
            return

        observed_count = int(np.count_nonzero(self.observed_mask)) if self.observed_mask is not None else 0
        occupied_count = int(np.count_nonzero(self.voxel_state == OCCUPIED)) if self.voxel_state is not None else 0
        unknown_count = int(np.count_nonzero(self.voxel_state == UNKNOWN)) if self.voxel_state is not None else 0
        free_count = int(np.count_nonzero(self.voxel_state == FREE)) if self.voxel_state is not None else 0

        message = (
            f"{title} | initialized: {self.initialized} | "
            f"observed: {observed_count} | "
            f"occupied: {occupied_count} | "
            f"unknown: {unknown_count} | "
            f"free: {free_count} | "
            f"occupied volume: {self.occupied_volume:.6f} m³"
        )

        if new_observed is not None:
            message += f" | new observed: {new_observed}"

        print(message)
