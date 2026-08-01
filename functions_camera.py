import numpy as np
from scipy.spatial.transform import Rotation as R


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

def compute_lookat_quaternion(cam_pos, center: np.ndarray, world_up=np.array([0, 1, 0])):
    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)
    rot_matrix =  np.column_stack((right, true_up, forward))

    return R.from_matrix(rot_matrix).as_quat()

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

def depth_to_camera_points(depth, fov_degrees, far):
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
    valid_mask = make_valid_depth_mask(depth, far)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy

    points_camera = np.column_stack((x, y, z))
    return points_camera

def camera_to_world(points_camera, position, quaternion):
    points_camera = np.asarray(points_camera, dtype=float)
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return points_camera @ R.T + position

def world_to_camera(points_world, position, quaternion):
    points_world = np.asarray(points_world, dtype=float)
    R = quaternion_to_rotation_matrix(*quaternion)
    position = np.asarray(position, dtype=float)
    return (points_world - position) @ R

def filter_by_height(depth, fov_degrees, far, position, quaternion, floor_height, height_threshold):
    points_camera = depth_to_camera_points(depth, fov_degrees, far)
    points_world = camera_to_world(points_camera, position, quaternion)
    min_height = floor_height + height_threshold 
    return points_world[points_world[:, 1] > min_height ]

class Initialization:
    def __init__(self, min_height, max_elev_deg=30, min_points=100, min_alt_buffer=1.5):
        self.min_height = min_height
        self.max_elev_rad = np.deg2rad(max_elev_deg)
        self.min_points = min_points
        self.min_alt_buffer = min_alt_buffer
    
    def verify_num_points(self, points_world):
        if len(points_world) < self.min_points:
            return False
        return True

    def verify_altitude(self, points_world, camera_pos):
        median_y = np.median(points_world[:, 1])
        if camera_pos[1] < median_y * self.min_alt_buffer:
            return False
        return True
    
    def verification(self, points_world, camera_pos, quaternion, width, height, fov_degrees):
        if not self.verify_num_points(points_world):
            return False
        if not self.verify_altitude(points_world, camera_pos):
            return False
        if not self.verify_full_visibility(points_world, camera_pos, quaternion, width, height, fov_degrees):
            return False
        return True

    def verify_full_visibility(self, points_world, camera_pos, quaternion, width, height, fov_degrees, border_pixel=20):
        points_camera = world_to_camera(points_world, camera_pos, quaternion)
        
        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]
        
        fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
        
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy
        
        all_u_valid = np.all((u >= border_pixel) & (u < width - border_pixel))
        all_v_valid = np.all((v >= border_pixel) & (v < height - border_pixel))
        
        return all_u_valid and all_v_valid

    def get_object_properties(self, points_world):
        object_min = points_world.min(axis=0)
        object_max = points_world.max(axis=0)
        object_center = (object_min + object_max) / 2
        return object_min, object_max, object_center

    def initial_position(self, points_world):
        views = []
        _, object_max, object_center = self.get_object_properties(points_world)
        y_init = max(object_max[1] * self.min_alt_buffer, self.min_height)
        dy = y_init - object_center[1]
        r_init = dy / np.sin(self.max_elev_rad)
        r_azi = r_init * np.cos(self.max_elev_rad)
        
        initial_position = np.array([object_center[0] , y_init, object_center[2]+ r_azi])
        initial_quaternion = compute_lookat_quaternion(initial_position, object_center)

        views.append({"view_id":0,"pos":initial_position, "quat": initial_quaternion})        
        return views
    
    
    
    