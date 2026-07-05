import numpy as np
import cv2

DEPTH_MIN_VALID = 0.0
DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01

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
    if norm < 1e-12: return np.eye(3)
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
    if len(points_camera) == 0: return np.empty((0, 3))
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return points_camera @ R.T + position

def world_to_camera(points_world, position, quaternion):
    points_world = np.asarray(points_world, dtype=float)
    if len(points_world) == 0: return np.empty((0, 3))
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return (points_world - position) @ R

class DepthFrameProcessor:
    def __init__(self, depth, fov_degrees, far, position, quaternion, floor_height=0.0, height_threshold=0.02, verbose=True):
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
        self.points_camera, self.valid_mask = depth_to_camera_points(self.depth, self.fov_degrees, self.far)
        self.points_world = camera_to_world(self.points_camera, self.position, self.quaternion)
        self.filtered_points = self._filter_by_height(self.points_world)
        return self

    def _filter_by_height(self, points_world):
        if points_world is None or len(points_world) == 0:
            return np.empty((0, 3))
        min_height = self.floor_height + self.height_threshold
        filtered = points_world[points_world[:, 1] > min_height]
        if self.verbose:
            print(f"Height filter | input: {len(points_world)} | filtered: {len(filtered)} | min Y: {min_height:.3f}")
        return filtered

class ExtrusionFusionReconstruction:
    def __init__(self, voxel_size=0.01, bbox_margin=0.05, floor_height=0.0, max_points=None, verbose=True):
        self.voxel_size = voxel_size
        self.bbox_margin = bbox_margin
        self.floor_height = floor_height
        self.max_points = max_points
        self.verbose = verbose

        self.initialized = False
        self.bbox_min = None
        self.bbox_max = None
        self.bbox_corners = None
        self.grid_shape = None

        self.free_prob = 0.0
        self.unknown_prob = 0.5
        self.occupied_prob = 1.0
        self.occupied_threshold = 0.75

        self.voxel_prob = {}
        self.occupied_indices = set()
        self.observed_indices = set()
        self.free_indices = set()
        self.unknown_indices = set()

        self.observed_centers = np.empty((0, 3))
        self.occupied_centers = np.empty((0, 3))
        self.free_centers = np.empty((0, 3))
        self.unknown_centers = np.empty((0, 3))

        self.observed_volume = 0.0
        self.occupied_volume = 0.0
        self.free_volume = 0.0
        self.unknown_volume = 0.0
        self.expected_volume = 0.0

    def add_view(self, points_world, depth=None, fov_degrees=None, far=None, position=None, quaternion=None, carve_margin=None, carve_with_invalid_depth=False):
        points_world = self._validate_points(points_world)

        if not self.initialized:
            self._initialize_bbox_from_detection(points_world)

        if self._has_camera_data(depth, fov_degrees, far, position, quaternion):
            self.carve_with_depth_image(
                depth=depth, fov_degrees=fov_degrees, far=far,
                position=position, quaternion=quaternion,
                carve_margin=carve_margin, carve_with_invalid_depth=carve_with_invalid_depth,
                keep_invalid_depth=True
            )
        else:
            self._log("Skipping carving: missing depth/camera information.")

        new_observed = self._voxelize(points_world)
        before_observed = len(self.observed_indices)

        self.observed_indices |= new_observed
        for idx in new_observed:
            self.voxel_prob[idx] = self.occupied_prob

        self._refresh_state_sets_from_probabilities()
        self._update_outputs()
        self._log_status("Processed view", new_observed=len(self.observed_indices) - before_observed)
        return self

    def carve_with_depth_image(self, depth, fov_degrees, far, position, quaternion, carve_margin=None, carve_with_invalid_depth=False, keep_invalid_depth=False):
        if not self.initialized or not self.voxel_prob:
            return self

        if carve_margin is None: carve_margin = 0.5 * self.voxel_size
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2: return self

        height, width = depth.shape
        fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
        
        if keep_invalid_depth:
            valid_depth_mask = np.ones(depth.shape, dtype=bool)
        else:
            valid_depth_mask = make_valid_depth_mask(depth, far)

        indices = np.array([idx for idx, prob in self.voxel_prob.items() if prob > self.free_prob], dtype=int)
        if len(indices) == 0: return self

        centers_world = self._indices_to_centers_from_array(indices)
        centers_camera = world_to_camera(centers_world, position, quaternion)

        x, y, z = centers_camera[:, 0], centers_camera[:, 1], centers_camera[:, 2]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(x * fx / z + cx).astype(int)
            v = np.round(cy - y * fy / z).astype(int)

        projects_inside_image = ((z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height))
        if not np.any(projects_inside_image): return self

        candidate_indices = indices[projects_inside_image]
        candidate_z = z[projects_inside_image]
        candidate_u = u[projects_inside_image]
        candidate_v = v[projects_inside_image]

        measured_depth = depth[candidate_v, candidate_u]
        valid_measured_depth = valid_depth_mask[candidate_v, candidate_u]

        carve_free_space = valid_measured_depth & (candidate_z < measured_depth - carve_margin)
        if carve_with_invalid_depth:
            carve_free_space |= (~valid_measured_depth) & (candidate_z < far - carve_margin)

        remove_indices = {tuple(idx) for idx in candidate_indices[carve_free_space]}
        before_free = len(self.free_indices)

        for idx in remove_indices:
            if idx not in self.observed_indices:
                self.voxel_prob[idx] = self.free_prob

        for idx in self.observed_indices:
            self.voxel_prob[idx] = self.occupied_prob

        self._refresh_state_sets_from_probabilities()
        self._update_outputs()

        newly_free = len(self.free_indices) - before_free
        self._log(f"Depth carving | projected voxels: {len(candidate_indices)} | newly free: {newly_free} | occupied volume: {self.occupied_volume:.6f} m³")
        return self

    def _initialize_bbox_from_detection(self, points_world):
        self.bbox_min, self.bbox_max, self.bbox_corners = self._compute_bbox(points_world)
        self.grid_shape = self._compute_grid_shape()

        all_bbox_indices = self._fill_bbox()
        self.voxel_prob = {idx: self.unknown_prob for idx in all_bbox_indices}

        self.observed_indices = self._voxelize(points_world)
        for idx in self.observed_indices:
            self.voxel_prob[idx] = self.occupied_prob

        self.initialized = True
        self._refresh_state_sets_from_probabilities()
        self._update_outputs()
        self._log_status("Initialized probabilistic bbox from first object detection")
        return self

    def _fill_bbox(self):
        nx, ny, nz = self.grid_shape
        return {(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)}

    def _validate_points(self, points_world):
        if points_world is None: return np.empty((0, 3))
        points_world = np.asarray(points_world, dtype=float)
        if len(points_world) == 0: return np.empty((0, 3))
        if self.max_points is not None and len(points_world) > self.max_points:
            selected = np.random.choice(len(points_world), self.max_points, replace=False)
            points_world = points_world[selected]
        return points_world

    def _compute_bbox(self, points_world):
        bbox_min = points_world.min(axis=0).copy()
        bbox_max = points_world.max(axis=0).copy()
        bbox_min[[0, 2]] -= self.bbox_margin
        bbox_max[[0, 2]] += self.bbox_margin
        bbox_min[1] = self.floor_height
        mn_x, mn_y, mn_z = bbox_min
        mx_x, mx_y, mx_z = bbox_max
        corners = np.array([
            [mn_x, mn_y, mn_z], [mx_x, mn_y, mn_z], [mx_x, mn_y, mx_z], [mn_x, mn_y, mx_z],
            [mn_x, mx_y, mn_z], [mx_x, mx_y, mn_z], [mx_x, mx_y, mx_z], [mn_x, mx_y, mx_z],
        ])
        return bbox_min, bbox_max, corners

    def _compute_grid_shape(self):
        return np.maximum(np.ceil((self.bbox_max - self.bbox_min) / self.voxel_size).astype(int), 1)

    def _voxelize(self, points_world):
        if self.bbox_min is None or self.grid_shape is None: return set()
        points_world = np.asarray(points_world, dtype=float)
        if len(points_world) == 0: return set()
        indices = np.floor((points_world - self.bbox_min) / self.voxel_size).astype(int)
        inside = np.all((indices >= 0) & (indices < self.grid_shape), axis=1)
        indices = indices[inside]
        if len(indices) == 0: return set()
        return {tuple(idx) for idx in np.unique(indices, axis=0)}

    def _indices_to_centers(self, index_set):
        if not index_set: return np.empty((0, 3))
        indices = np.array(list(index_set), dtype=int)
        return self._indices_to_centers_from_array(indices)

    def _indices_to_centers_from_array(self, indices):
        indices = np.asarray(indices, dtype=float)
        return self.bbox_min + (indices + 0.5) * self.voxel_size

    def _refresh_state_sets_from_probabilities(self):
        self.free_indices = {idx for idx, prob in self.voxel_prob.items() if prob <= self.free_prob}
        self.unknown_indices = {idx for idx, prob in self.voxel_prob.items() if self.free_prob < prob < self.occupied_threshold}
        self.occupied_indices = {idx for idx, prob in self.voxel_prob.items() if prob >= self.occupied_threshold}

    def _update_outputs(self):
        voxel_volume = self.voxel_size ** 3
        self.observed_centers = self._indices_to_centers(self.observed_indices)
        self.occupied_centers = self._indices_to_centers(self.occupied_indices)
        self.free_centers = self._indices_to_centers(self.free_indices)
        self.unknown_centers = self._indices_to_centers(self.unknown_indices)

        self.observed_volume = len(self.observed_indices) * voxel_volume
        self.occupied_volume = len(self.occupied_indices) * voxel_volume
        self.free_volume = len(self.free_indices) * voxel_volume
        self.unknown_volume = len(self.unknown_indices) * voxel_volume
        self.expected_volume = sum(self.voxel_prob.values()) * voxel_volume

    @staticmethod
    def _has_camera_data(depth, fov_degrees, far, position, quaternion):
        return all(v is not None for v in [depth, fov_degrees, far, position, quaternion])

    def _log(self, message):
        if self.verbose: print(message)

    def _log_status(self, title, new_observed=None):
        if not self.verbose: return
        message = (f"{title} | initialized: {self.initialized} | observed: {len(self.observed_indices)} | occupied: {len(self.occupied_indices)} | unknown: {len(self.unknown_indices)} | free: {len(self.free_indices)} | occupied volume: {self.occupied_volume:.6f} m³")
        if new_observed is not None: message += f" | new observed: {new_observed}"
        print(message)

    def get_unknown_surface_indices(self):
        """A voxel is an unknown surface voxel if at least one 6-neighbor is FREE space."""
        if not self.unknown_indices: return set()
        neighbor_offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        unknown_surface_indices = set()

        for idx in self.unknown_indices:
            ix, iy, iz = idx
            for dx, dy, dz in neighbor_offsets:
                neighbor = (ix + dx, iy + dy, iz + dz)
                if neighbor in self.free_indices:
                    unknown_surface_indices.add(idx)
                    break
        return unknown_surface_indices

    def count_unknown_surface_voxels(self):
        return len(self.get_unknown_surface_indices())