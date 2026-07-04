import numpy as np
import open3d as o3d


UNKNOWN = -1
EMPTY = 0
FILLED = 1


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)

    x, y, z, w = q

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        [0,                       0,                       0],
    ])[:3, :3]


def make_open3d_intrinsic(width, height, fov_degrees):
    fy = height / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    return o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def depth_to_open3d_point_cloud(
    depth,
    fov_degrees,
    far,
    depth_scale=1.0,
    depth_trunc=None,
    stride=2,
):
    """
    Convert depth image to Open3D point cloud.

    Assumption:
        depth is already in meters.

    If your depth is in millimeters, use depth_scale=1000.0.
    """

    if depth_trunc is None:
        depth_trunc = far

    depth = np.asarray(depth, dtype=np.float32)

    height, width = depth.shape
    intrinsic = make_open3d_intrinsic(width, height, fov_degrees)

    depth_o3d = o3d.geometry.Image(depth)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_o3d,
        intrinsic=intrinsic,
        extrinsic=np.eye(4),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        stride=stride,
        project_valid_depth_only=True,
    )

    return pcd


def transform_point_cloud_to_world(pcd, position, quaternion):
    """
    Transform Open3D point cloud from camera frame to world frame.
    """

    rotation = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position

    pcd_world = o3d.geometry.PointCloud(pcd)
    pcd_world.transform(T)

    return pcd_world


def filter_object_points(points_world, floor_height=0.0, height_threshold=0.02):
    """
    Remove floor points.
    """

    if len(points_world) == 0:
        return np.empty((0, 3))

    min_height = floor_height + height_threshold

    return points_world[points_world[:, 1] > min_height]


class Open3DOccupancyVoxelMap:
    """
    Simple occupancy voxel map using Open3D for point cloud conversion.

    State meaning:
        UNKNOWN = -1
        EMPTY   = 0
        FILLED  = 1

    Unknown voxels are implicit:
        If a voxel index is not in self.voxels, it is UNKNOWN.
    """

    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        self.voxels = {}

    def world_to_voxel(self, point):
        point = np.asarray(point, dtype=float)
        return tuple(np.floor(point / self.voxel_size).astype(int))

    def voxel_to_world(self, voxel_index):
        voxel_index = np.asarray(voxel_index, dtype=float)
        return (voxel_index + 0.5) * self.voxel_size

    def get_state(self, voxel_index):
        return self.voxels.get(tuple(voxel_index), UNKNOWN)

    def update_from_depth(
        self,
        depth,
        fov_degrees,
        far,
        position,
        quaternion,
        floor_height=0.0,
        height_threshold=0.02,
        stride=2,
    ):
        """
        Main update function.

        Input:
            depth       : 2D depth image in meters
            fov_degrees : camera vertical FOV
            far         : far clipping distance
            position    : camera world position [x, y, z]
            quaternion  : camera world rotation [qx, qy, qz, qw]

        Output:
            object_points_world
        """

        pcd_camera = depth_to_open3d_point_cloud(
            depth=depth,
            fov_degrees=fov_degrees,
            far=far,
            depth_scale=1.0,
            depth_trunc=far,
            stride=stride,
        )

        pcd_world = transform_point_cloud_to_world(
            pcd=pcd_camera,
            position=position,
            quaternion=quaternion,
        )

        points_world = np.asarray(pcd_world.points)

        object_points_world = filter_object_points(
            points_world=points_world,
            floor_height=floor_height,
            height_threshold=height_threshold,
        )

        self.carve_depth_rays(
            camera_position=np.asarray(position, dtype=float),
            surface_points=object_points_world,
        )

        return object_points_world

    def carve_depth_rays(self, camera_position, surface_points):
        """
        Depth carving.

        For every observed surface point:
            voxels before surface point = EMPTY
            voxel containing surface point = FILLED
        """

        if len(surface_points) == 0:
            return

        for point in surface_points:
            ray_voxels = self.voxels_along_ray(camera_position, point)

            if len(ray_voxels) == 0:
                continue

            empty_voxels = ray_voxels[:-1]
            filled_voxel = ray_voxels[-1]

            for voxel in empty_voxels:
                if self.get_state(voxel) != FILLED:
                    self.voxels[voxel] = EMPTY

            self.voxels[filled_voxel] = FILLED

    def voxels_along_ray(self, start_point, end_point):
        """
        Simple ray traversal by sampling points along the ray.

        This is easier than full 3D DDA and good enough for a clean thesis prototype.
        """

        start_point = np.asarray(start_point, dtype=float)
        end_point = np.asarray(end_point, dtype=float)

        direction = end_point - start_point
        distance = np.linalg.norm(direction)

        if distance < 1e-9:
            return []

        step_size = self.voxel_size * 0.5
        num_steps = max(int(distance / step_size), 1)

        sampled_points = np.linspace(start_point, end_point, num_steps)

        voxels = []
        previous_voxel = None

        for point in sampled_points:
            voxel = self.world_to_voxel(point)

            if voxel != previous_voxel:
                voxels.append(voxel)
                previous_voxel = voxel

        return voxels

    def get_filled_voxels(self):
        return {
            voxel
            for voxel, state in self.voxels.items()
            if state == FILLED
        }

    def get_empty_voxels(self):
        return {
            voxel
            for voxel, state in self.voxels.items()
            if state == EMPTY
        }

    def get_filled_centers(self):
        return np.array([
            self.voxel_to_world(voxel)
            for voxel in self.get_filled_voxels()
        ])

    def get_empty_centers(self):
        return np.array([
            self.voxel_to_world(voxel)
            for voxel in self.get_empty_voxels()
        ])

    def get_filled_volume(self):
        return len(self.get_filled_voxels()) * self.voxel_size ** 3

    def create_open3d_voxel_grid(self):
        """
        Create an Open3D VoxelGrid only for visualization of FILLED voxels.
        """

        filled_centers = self.get_filled_centers()

        if len(filled_centers) == 0:
            return o3d.geometry.VoxelGrid()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filled_centers)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=self.voxel_size,
        )

        return voxel_grid

    def summary(self):
        filled = len(self.get_filled_voxels())
        empty = len(self.get_empty_voxels())

        print(f"Filled voxels: {filled}")
        print(f"Empty voxels:  {empty}")
        print(f"Filled volume: {self.get_filled_volume():.6f} m³")