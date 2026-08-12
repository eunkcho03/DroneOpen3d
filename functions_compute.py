import numpy as np
import numba 
from functions_camera import camera_intrinsics, make_valid_depth_mask, world_to_camera, filter_by_height

# Dense voxel-state values
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

class ExtrusionFusionReconstruction:
    def __init__(
        self,
        voxel_size_ratio=0.01,
        bbox_margin=0.05,
        floor_height=0.0,
        min_z_height=0.01,
    ):
        #self.voxel_size = 1
        self.voxel_size_ratio = voxel_size_ratio
        self.bbox_margin = bbox_margin
        self.floor_height = floor_height
        self.min_z_height = min_z_height
        self.initialized = False
        
                
    def initialize(self, points_world):
        points_world = np.asarray(points_world)
        self.bbox_min, self.bbox_max, self.bbox_corners, self.bbox_center = self._compute_bbox(points_world)
        self.grid_shape = self._compute_grid_shape()
        self.voxel_state = np.full(self.grid_shape, UNKNOWN, dtype=np.uint8)
        self.observed_mask = np.zeros(self.grid_shape, dtype=bool)
        self.initialized = True
        self.carve_margin = 0.5 * self.voxel_size
        self._update_outputs()
        self._log_status('Initialized bbox from first object detection')

    def carve_with_depth_image(
        self,
        depth,
        fov_degrees,
        far,
        position,
        quaternion,
        floor_height,
        height_threshold
    ):
        depth = np.asarray(depth, dtype=float)
        height, width = depth.shape
        fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
        valid_depth_mask = make_valid_depth_mask(depth, far)

        ix, iy, iz = np.where(self.voxel_state != FREE)
        indices = np.column_stack((ix, iy, iz))
        centers_world = self._indices_to_centers_from_array(indices)

        # 2. Project into camera space
        centers_camera = world_to_camera(centers_world, position, quaternion)

        x = centers_camera[:, 0]
        y = centers_camera[:, 1]
        z = centers_camera[:, 2]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(x * fx / z + cx).astype(int)
            v = np.round(y * fy / z + cy).astype(int)

        projects_inside_image = (
            (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        )

        candidate_indices = indices[projects_inside_image]
        candidate_z = z[projects_inside_image]
        candidate_u = u[projects_inside_image]
        candidate_v = v[projects_inside_image]

        measured_depth = depth[candidate_v, candidate_u]
        valid_measured_depth = valid_depth_mask[candidate_v, candidate_u]

        # 3. Evaluate carving rule
        filtered_points = filter_by_height(depth, fov_degrees, far, position, quaternion, floor_height, height_threshold)
        self.mark_observed_occupied(filtered_points)

        carve_free_space = np.where(
            valid_measured_depth,
            candidate_z < measured_depth - self.carve_margin,
            candidate_z < far,
        )

        remove_indices = candidate_indices[carve_free_space]
        
        removed_count = 0
        for idx in remove_indices:
            cx, cy, cz = idx
            if not self.observed_mask[cx, cy, cz]:
                self.voxel_state[cx, cy, cz] = FREE
                removed_count += 1
        
        self._update_outputs()
        self._log_status('depth_carving')

        return self

    def mark_observed_occupied(self, filtered_points):
        self.expand_grid_if_needed(filtered_points)
        indices = self._voxelize(filtered_points)
        ix = indices[:, 0]
        iy = indices[:, 1]
        iz = indices[:, 2]
        self.observed_mask[ix, iy, iz] = True
        self.voxel_state[ix, iy, iz] = OCCUPIED


    def _compute_bbox(self, points_world):
        bbox_min = points_world.min(axis=0).copy()
        bbox_max = points_world.max(axis=0).copy()
        
        initial_bbox_dimensions = bbox_max - bbox_min
        shortest_length = initial_bbox_dimensions.min()
        self.voxel_size = self.voxel_size_ratio * shortest_length

        bbox_min[[0, 2]] -= self.bbox_margin
        bbox_max[[0, 2]] += self.bbox_margin
        bbox_min[1] = self.floor_height

        corners = self.get_corners(bbox_min, bbox_max)

        center = 0.5 * (bbox_min + bbox_max)

        return bbox_min, bbox_max, corners, center


    def get_corners(self, bbox_min, bbox_max):
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
        
        return corners

    def expand_grid_if_needed(self, points_world):
        points_min = points_world.min(axis=0)
        points_max = points_world.max(axis=0)
        
        needs_min_expand = points_min < self.bbox_min
        needs_max_expand = points_max > self.bbox_max
        
        if not (np.any(needs_min_expand) or np.any(needs_max_expand)):
            return 
        
        pad_min_voxels = np.maximum(0, np.ceil((self.bbox_min - points_min) / self.voxel_size).astype(int))
        pad_max_voxels = np.maximum(0, np.ceil((points_max - self.bbox_max) / self.voxel_size).astype(int))
        
        self.bbox_min -= pad_min_voxels * self.voxel_size
        self.bbox_max += pad_max_voxels * self.voxel_size
        
        pad_widths = list(zip(pad_min_voxels, pad_max_voxels))
        
        self.voxel_state = np.pad(self.voxel_state, pad_widths, mode='constant', constant_values=UNKNOWN)
        self.observed_mask = np.pad(self.observed_mask, pad_widths, mode='constant', constant_values=False)
        self.grid_shape = np.array(self.voxel_state.shape, dtype=int)
        self.bbox_corners = self.get_corners(self.bbox_min, self.bbox_max)

    def _compute_grid_shape(self):
        return np.maximum(
            np.ceil((self.bbox_max - self.bbox_min) / self.voxel_size).astype(int), 1,
        )

    def _voxelize(self, points_world):
        indices = np.floor((points_world - self.bbox_min) / self.voxel_size).astype(int)
        inside = np.all((indices >= 0) & (indices < self.grid_shape), axis=1)
        indices = indices[inside]
        return np.unique(indices, axis=0)

    def _indices_to_centers_from_array(self, indices):
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        indices = indices.reshape(-1, 3)
        return self.bbox_min + (indices + 0.5) * self.voxel_size

    def _update_outputs(self):
        voxel_volume = self.voxel_size ** 3

        occupied_mask = self.voxel_state == OCCUPIED
        unknown_mask = self.voxel_state == UNKNOWN

        self.occupied_count = int(np.count_nonzero(occupied_mask))
        self.unknown_count = int(np.count_nonzero(unknown_mask))

        self.occupied_volume = self.occupied_count * voxel_volume
        self.unknown_volume = self.unknown_count * voxel_volume

        self.occupied_indices = np.argwhere(occupied_mask)
        self.unknown_indices = np.argwhere(unknown_mask)

        self.occupied_centers = self._indices_to_centers_from_array(self.occupied_indices)
        self.unknown_centers = self._indices_to_centers_from_array(self.unknown_indices)

    def _log_status(self, title, new_observed=None):
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
            f"occupied + unknown volume: {self.occupied_volume+ self.unknown_volume:.6f} m³"
        )

        if new_observed is not None:
            message += f" | new observed: {new_observed}"

        print(message)
    
    def surface_data(self):
        unknown_list, occupied_list = get_surface_data_numba(self.voxel_state, FREE, UNKNOWN, OCCUPIED)
        unknown_indices = np.array(unknown_list, dtype=np.int32)
        occupied_indices = np.array(occupied_list, dtype=np.int32)
        self.surface_centers(unknown_indices, occupied_indices)
        return unknown_indices, occupied_indices

    def surface_centers(self):
        unknown_list, occupied_list = get_surface_data_numba(self.voxel_state, FREE, UNKNOWN, OCCUPIED)
        self.unknown_indices = np.array(unknown_list, dtype=np.int32)
        self.occupied_indices = np.array(occupied_list, dtype=np.int32)
        self.occupied_surface_centers = self._indices_to_centers_from_array(self.occupied_indices)
        self.unknown_surface_centers = self._indices_to_centers_from_array(self.unknown_indices)

@numba.njit(fastmath=True)
def get_surface_data_numba(voxel_state, FREE=0, UNKNOWN=1, OCCUPIED=2):
    nx, ny, nz = voxel_state.shape
    unknown_list = numba.typed.List()
    occupied_list = numba.typed.List()

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                state = voxel_state[x, y, z]

                if state == FREE:
                    continue
                
                touches_free = (
                    (x > 0 and voxel_state[x - 1, y, z] == FREE)
                    or (x < nx - 1 and voxel_state[x + 1, y, z] == FREE)
                    or (y > 0 and voxel_state[x, y - 1, z] == FREE)
                    or (y < ny - 1 and voxel_state[x, y + 1, z] == FREE)
                    or (z > 0 and voxel_state[x, y, z - 1] == FREE)
                    or (z < nz - 1 and voxel_state[x, y, z + 1] == FREE)
                )
                
                touches_bbox_boundary = (
                    x == 0
                    or x == nx - 1
                    or y == 0
                    or y == ny - 1
                    or z == 0
                    or z == nz - 1
                )

                if state == UNKNOWN:
                    if touches_free or touches_bbox_boundary:
                        unknown_list.append((x, y, z))
                elif state == OCCUPIED:
                    if touches_free or touches_bbox_boundary:
                        occupied_list.append((x, y, z))
    return unknown_list, occupied_list