import numpy as np
import open3d as o3d


DEPTH_INVALID_VALUE = -1.0
FAR_PLANE_MARGIN = 0.01


def make_valid_depth_mask(depth, far):
    depth = np.asarray(depth)
    return (
        np.isfinite(depth)
        & (depth > 0.0)
        & (depth != DEPTH_INVALID_VALUE)
        & (depth < far - FAR_PLANE_MARGIN)
    )


def camera_intrinsic(width, height, fov_degrees):
    fy = height / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def camera_matrix(position, quaternion):
    qx, qy, qz, qw = quaternion
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def depth_for_open3d(depth, far):
    depth = np.asarray(depth, dtype=np.float32).copy()
    depth[~make_valid_depth_mask(depth, far)] = 0.0
    return depth


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    points_h = np.column_stack([points, np.ones(len(points))])
    return (points_h @ matrix.T)[:, :3]


def world_to_camera(points_world, position, quaternion):
    return transform_points(points_world, np.linalg.inv(camera_matrix(position, quaternion)))


class DepthFrameProcessor:
    """
    Convert one Unity depth frame into an Open3D point cloud.
    """

    def __init__(
        self,
        depth,
        fov_degrees,
        far,
        position,
        quaternion,
        floor_height=0.0,
        height_threshold=0.01,
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

        self.valid_mask = None
        self.pcd_camera = o3d.geometry.PointCloud()
        self.pcd_world = o3d.geometry.PointCloud()
        self.pcd_filtered = o3d.geometry.PointCloud()
        self.points_camera = np.empty((0, 3))
        self.points_world = np.empty((0, 3))
        self.filtered_points = np.empty((0, 3))

    def run(self):
        height, width = self.depth.shape
        self.valid_mask = make_valid_depth_mask(self.depth, self.far)

        depth_image = o3d.geometry.Image(depth_for_open3d(self.depth, self.far))
        intrinsic = camera_intrinsic(width, height, self.fov_degrees)

        self.pcd_camera = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic,
            depth_scale=1.0,
            depth_trunc=self.far,
            stride=1,
        )
        self.points_camera = np.asarray(self.pcd_camera.points)

        self.pcd_world = o3d.geometry.PointCloud(self.pcd_camera)
        self.pcd_world.transform(camera_matrix(self.position, self.quaternion))
        self.points_world = np.asarray(self.pcd_world.points)

        min_y = self.floor_height + self.height_threshold
        self.filtered_points = self.points_world[self.points_world[:, 1] > min_y]
        self.pcd_filtered.points = o3d.utility.Vector3dVector(self.filtered_points)

        if self.verbose:
            print(
                f"Height filter | input: {len(self.points_world)} | "
                f"filtered: {len(self.filtered_points)} | min Y: {min_y:.3f}"
            )

        return self


class Open3DVoxelReconstruction:
    """
    Open3D-based voxel reconstruction with the same useful outputs as the old code:
    occupied voxels, unknown voxels, free voxels, bbox, volumes, and unknown-surface count.

    Open3D is used for point-cloud creation and observed surface voxelization.
    The unknown/free grid is still stored as index sets because Open3D VoxelGrid only
    stores occupied voxels.
    """

    def __init__(
        self,
        voxel_size=0.01,
        bbox_margin=0.05,
        floor_height=0.0,
        max_points=None,
        verbose=True,
    ):
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

        self.all_indices = set()
        self.observed_indices = set()
        self.occupied_indices = set()
        self.unknown_indices = set()
        self.free_indices = set()

        self.extruded_indices = set()
        self.fused_indices = set()

        self.observed_centers = np.empty((0, 3))
        self.occupied_centers = np.empty((0, 3))
        self.unknown_centers = np.empty((0, 3))
        self.free_centers = np.empty((0, 3))
        self.fused_centers = np.empty((0, 3))
        self.extruded_centers = np.empty((0, 3))

        self.observed_volume = 0.0
        self.fused_volume = 0.0
        self.unknown_volume = 0.0
        self.free_volume = 0.0
        self.expected_volume = 0.0
        self.extruded_volume = 0.0

        self.pcd = o3d.geometry.PointCloud()
        self.voxel_grid = o3d.geometry.VoxelGrid()

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
        points_world = self._prepare_points(points_world)
        if len(points_world) == 0:
            return self

        if not self.initialized:
            self._initialize_from_points(points_world)

        if depth is not None:
            self.carve_with_depth_image(
                depth=depth,
                fov_degrees=fov_degrees,
                far=far,
                position=position,
                quaternion=quaternion,
                carve_margin=carve_margin,
                carve_with_invalid_depth=carve_with_invalid_depth,
            )

        new_observed = self._voxelize_with_open3d(points_world)
        self.observed_indices |= new_observed
        self.occupied_indices |= new_observed
        self.free_indices -= self.occupied_indices
        self.unknown_indices = self.all_indices - self.free_indices - self.occupied_indices
        self._update_outputs()

        if self.verbose:
            print(
                f"Processed view | observed: {len(self.observed_indices)} | "
                f"occupied: {len(self.occupied_indices)} | unknown: {len(self.unknown_indices)} | "
                f"free: {len(self.free_indices)}"
            )

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
    ):
        carve_margin = 0.5 * self.voxel_size if carve_margin is None else carve_margin
        depth = np.asarray(depth, dtype=float)
        height, width = depth.shape
        intrinsic = camera_intrinsic(width, height, fov_degrees)
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()

        candidate_indices = np.array(list(self.all_indices - self.free_indices), dtype=int)
        centers_world = self._indices_to_centers_from_array(candidate_indices)
        centers_camera = world_to_camera(centers_world, position, quaternion)

        x, y, z = centers_camera[:, 0], centers_camera[:, 1], centers_camera[:, 2]
        u = np.round(x * fx / z + cx).astype(int)
        v = np.round(cy - y * fy / z).astype(int)

        inside = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        candidate_indices = candidate_indices[inside]
        candidate_z = z[inside]
        u = u[inside]
        v = v[inside]

        measured_depth = depth[v, u]
        valid = make_valid_depth_mask(depth, far)[v, u]
        carve = valid & (candidate_z < measured_depth - carve_margin)

        if carve_with_invalid_depth:
            carve |= (~valid) & (candidate_z < far - carve_margin)

        carved_indices = {tuple(idx) for idx in candidate_indices[carve]}
        carved_indices -= self.observed_indices

        self.free_indices |= carved_indices
        self.occupied_indices -= self.free_indices
        self.unknown_indices = self.all_indices - self.free_indices - self.occupied_indices
        self._update_outputs()

        if self.verbose:
            print(f"Depth carving | projected: {len(candidate_indices)} | newly free: {len(carved_indices)}")

        return self

    def get_unknown_surface_indices(self):
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        surface = set()
        for ix, iy, iz in self.unknown_indices:
            for dx, dy, dz in offsets:
                if (ix + dx, iy + dy, iz + dz) not in self.unknown_indices:
                    surface.add((ix, iy, iz))
                    break
        return surface

    def count_unknown_surface_voxels(self):
        return len(self.get_unknown_surface_indices())

    def draw_open3d(self):
        self.voxel_grid = self._make_open3d_voxel_grid(self.occupied_indices)
        o3d.visualization.draw_geometries([self.voxel_grid])

    def _initialize_from_points(self, points_world):
        self.bbox_min, self.bbox_max, self.bbox_corners = self._compute_bbox(points_world)
        self.grid_shape = np.maximum(
            np.ceil((self.bbox_max - self.bbox_min) / self.voxel_size).astype(int),
            1,
        )
        self.all_indices = self._fill_bbox_indices()
        self.unknown_indices = set(self.all_indices)
        self.initialized = True
        self._update_outputs()

        if self.verbose:
            print(f"Initialized bbox grid | shape: {self.grid_shape} | voxels: {len(self.all_indices)}")

    def _prepare_points(self, points_world):
        points = np.asarray(points_world, dtype=float)
        if self.max_points is not None and len(points) > self.max_points:
            ids = np.random.choice(len(points), self.max_points, replace=False)
            points = points[ids]
        return points

    def _compute_bbox(self, points_world):
        bbox_min = points_world.min(axis=0).copy()
        bbox_max = points_world.max(axis=0).copy()
        bbox_min[[0, 2]] -= self.bbox_margin
        bbox_max[[0, 2]] += self.bbox_margin
        bbox_min[1] = self.floor_height
        return bbox_min, bbox_max, self._bbox_corners_from_bounds(bbox_min, bbox_max)

    def _fill_bbox_indices(self):
        nx, ny, nz = self.grid_shape
        grid = np.indices((nx, ny, nz)).reshape(3, -1).T
        return {tuple(idx) for idx in grid}

    def _voxelize_with_open3d(self, points_world):
        self.pcd.points = o3d.utility.Vector3dVector(points_world)
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            self.pcd,
            self.voxel_size,
            self.bbox_min,
            self.bbox_min + self.grid_shape * self.voxel_size,
        )
        return {tuple(voxel.grid_index) for voxel in self.voxel_grid.get_voxels()}

    def _indices_to_centers(self, indices_set):
        if not indices_set:
            return np.empty((0, 3))
        return self._indices_to_centers_from_array(np.array(list(indices_set), dtype=int))

    def _indices_to_centers_from_array(self, indices):
        return self.bbox_min + (np.asarray(indices, dtype=float) + 0.5) * self.voxel_size

    def _make_open3d_voxel_grid(self, indices_set):
        centers = self._indices_to_centers(indices_set)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_size)

    def _update_outputs(self):
        voxel_volume = self.voxel_size ** 3
        self.fused_indices = set(self.occupied_indices)
        self.extruded_indices = set(self.occupied_indices)

        self.observed_centers = self._indices_to_centers(self.observed_indices)
        self.occupied_centers = self._indices_to_centers(self.occupied_indices)
        self.fused_centers = self.occupied_centers
        self.extruded_centers = self.occupied_centers
        self.unknown_centers = self._indices_to_centers(self.unknown_indices)
        self.free_centers = self._indices_to_centers(self.free_indices)

        self.observed_volume = len(self.observed_indices) * voxel_volume
        self.fused_volume = len(self.fused_indices) * voxel_volume
        self.extruded_volume = self.fused_volume
        self.unknown_volume = len(self.unknown_indices) * voxel_volume
        self.free_volume = len(self.free_indices) * voxel_volume
        self.expected_volume = self.fused_volume + 0.5 * self.unknown_volume

    @staticmethod
    def _bbox_corners_from_bounds(bbox_min, bbox_max):
        mn_x, mn_y, mn_z = bbox_min
        mx_x, mx_y, mx_z = bbox_max
        return np.array([
            [mn_x, mn_y, mn_z], [mx_x, mn_y, mn_z], [mx_x, mn_y, mx_z], [mn_x, mn_y, mx_z],
            [mn_x, mx_y, mn_z], [mx_x, mx_y, mn_z], [mx_x, mx_y, mx_z], [mn_x, mx_y, mx_z],
        ])


# Compatibility with your older main.py name.
ExtrusionFusionReconstruction = Open3DVoxelReconstruction
