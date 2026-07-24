from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from functions_camera import world_to_camera
from nbv_distance import compute_distance


@dataclass
class CameraIntrinsics:
    fov_deg: float
    near: float
    width: int
    height: int


class NextBestViewPlanner:
    def __init__(
        self,
        camera: CameraIntrinsics,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        center: np.ndarray,
        margin: float = 1.0,
        min_height: float = 0.0,
        max_elev_rad: float = np.pi / 3,
        azi_min_step_rad: float = np.deg2rad(10),
        elev_min_step_rad: float = np.deg2rad(10),
    ):
        self.camera = camera
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.center = center
        self.margin = margin
        self.min_height = min_height
        self.max_elev_rad = max_elev_rad
        self.azi_min_step_rad = azi_min_step_rad
        self.elev_min_step_rad = elev_min_step_rad

        self.r_obj = 0.5 * np.linalg.norm(self.bbox_max - self.bbox_min)
        self.view_radius = self.r_obj / np.sin(0.5 * np.deg2rad(self.camera.fov_deg)) * self.margin

    @staticmethod
    def compute_lookat_quaternion(cam_pos: np.ndarray, center: np.ndarray, up=np.array([0, 1, 0])) -> np.ndarray:
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        rot_matrix = np.eye(3)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = true_up
        rot_matrix[:, 2] = -forward

        return R.from_matrix(rot_matrix).as_quat()

    def generate_candidate_views(self, n_elev: int, n_azi: int):
        d_off = self.min_height - self.center[1]
        sin_val = np.clip(d_off / self.view_radius, -1.0, 1.0)
        elev_min_rad = max(np.arcsin(sin_val), -np.pi / 6)

        elevation_span = self.max_elev_rad - elev_min_rad
        n_azi_eff = min(n_azi, max(1, int(np.floor(2 * np.pi / self.azi_min_step_rad))))
        n_elev_eff = min(n_elev, max(1, int(np.floor(elevation_span / self.elev_min_step_rad))))

        azi_rad = np.linspace(0, 2 * np.pi, n_azi_eff, endpoint=False)
        elev_rad = (
            np.array([0.5 * (elev_min_rad + self.max_elev_rad)])
            if n_elev_eff == 1
            else np.linspace(elev_min_rad, self.max_elev_rad, n_elev_eff)
        )

        views = []
        view_id = 0
        for elev in elev_rad:
            r_h = self.view_radius * np.cos(elev)
            r_v = self.view_radius * np.sin(elev)
            for azi in azi_rad:
                cam_pos = np.array([
                    self.center[0] + r_h * np.cos(azi),
                    self.center[1] + r_v,
                    self.center[2] + r_h * np.sin(azi),
                ], dtype=float)

                quat = self.compute_lookat_quaternion(cam_pos, self.center)
                views.append({'view_id': view_id, 'pos': cam_pos, 'quat': quat})
                view_id += 1

        return views

    def compute_gain(
        self,
        occupied_points_w: np.ndarray,
        unknown_points_w: np.ndarray,
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ):
        surface_points_w = np.vstack([occupied_points_w, unknown_points_w])
        labels = np.concatenate([
            np.zeros(len(occupied_points_w), dtype=np.int8),
            np.ones(len(unknown_points_w), dtype=np.int8),
        ])

        cam_sur = world_to_camera(surface_points_w, cam_pos, cam_quat)

        x, y, z_depth = cam_sur[:, 0], cam_sur[:, 1], cam_sur[:, 2]

        aspect_ratio = self.camera.width / self.camera.height
        theta_v = np.deg2rad(self.camera.fov_deg) * 0.5
        theta_h = np.arctan(np.tan(theta_v) * aspect_ratio)

        valid = (
            (z_depth > self.camera.near)
            & (np.abs(x / z_depth) <= np.tan(theta_h))
            & (np.abs(y / z_depth) <= np.tan(theta_v))
        )

        x_norm = x[valid] / z_depth[valid] / np.tan(theta_h)
        y_norm = y[valid] / z_depth[valid] / np.tan(theta_v)

        u_val = np.clip(((x_norm + 1) * 0.5) * (self.camera.width - 1), 0, self.camera.width - 1).astype(int)
        v_val = np.clip((1 - (y_norm + 1) * 0.5) * (self.camera.height - 1), 0, self.camera.height - 1).astype(int)

        z_val = z_depth[valid]
        labels_val = labels[valid]

        pixel_idx = v_val * self.camera.width + u_val
        order = np.lexsort((z_val, pixel_idx))

        pixel_sorted = pixel_idx[order]
        labels_sorted = labels_val[order]

        first_mask = np.ones(len(pixel_sorted), dtype=bool)
        first_mask[1:] = pixel_sorted[1:] != pixel_sorted[:-1]

        visible_labels = labels_sorted[first_mask]
        return int(np.count_nonzero(visible_labels == 1))

    def select_best_view(
        self,
        occupied_points_w: np.ndarray,
        unknown_points_w: np.ndarray,
        cam_cur_pos: np.ndarray,
        n_elev: int,
        n_azi: int,
        fac_infl: float,
        distance_penalty: bool = True,
    ):
        views = self.generate_candidate_views(n_elev, n_azi)
        best_score = -1.0
        best_view = None

        for view in views:
            gain_abs = self.compute_gain(occupied_points_w, unknown_points_w, view['pos'], view['quat'])
            dist = compute_distance(cam_cur_pos, view['pos'], self.bbox_min, self.bbox_max, fac_infl)

            score = (gain_abs / dist) if distance_penalty else float(gain_abs)

            if score > best_score:
                best_score = score
                best_view = view

        return best_score, best_view['view_id'], best_view