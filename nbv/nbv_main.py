from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from functions_camera import world_to_camera
from nbv.nbv_distance import compute_distance


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
    ):
        self.camera = camera
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.center = center
        self.margin = margin
        self.min_height = min_height

        self.r_obj = 0.5 * np.linalg.norm(self.bbox_max - self.bbox_min)
        self.view_radius = self.r_obj / np.sin(0.5 * np.deg2rad(self.camera.fov_deg)) * self.margin

    @staticmethod
    def compute_lookat_quaternion(cam_pos: np.ndarray, center: np.ndarray, world_up=np.array([0, 1, 0])) -> np.ndarray:
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        true_up = np.cross(forward, right)
        rot_matrix =  np.column_stack((right, true_up, forward))

        return R.from_matrix(rot_matrix).as_quat()

    def generate_candidate_views(self, n_elev: int, n_azi: int, elev_min_deg=-30, elev_max_deg=30):
        d_off = self.min_height - self.center[1]
        sin_val = np.clip(d_off / self.view_radius, -1.0, 1.0)        
        elev_min_rad = np.arcsin(sin_val)

        azi_rad = np.linspace(0, 2 * np.pi, n_azi, endpoint=False)
        elev_rad = np.linspace(np.deg2rad(elev_min_deg), np.deg2rad(elev_max_deg), n_elev)
        elev_rad = elev_rad[elev_rad>=elev_min_rad]

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
        visualize_candidates: bool = True,
    ):
        views = self.generate_candidate_views(n_elev, n_azi)
        best_score = -1.0
        best_view = None
        best_dist = 0
        for view in views:
            gain_abs = self.compute_gain(occupied_points_w, unknown_points_w, view['pos'], view['quat'])
            #print('gain', gain_abs)
            if visualize_candidates:
                            self.visualize_view(
                                occupied_points_w, 
                                unknown_points_w, 
                                view["pos"], 
                                view["quat"], 
                                view["view_id"]
                            )
            dist = compute_distance(cam_cur_pos, view['pos'], self.bbox_min, self.bbox_max, fac_infl)

            score = (gain_abs / dist) if distance_penalty else float(gain_abs)

            if score > best_score:
                best_score = score
                best_view = view
                best_dist = dist

        return best_score, best_view, best_dist
    
    def visualize_view(
            self,
            occupied_points_w: np.ndarray,
            unknown_points_w: np.ndarray,
            cam_pos: np.ndarray,
            cam_quat: np.ndarray,
            view_id: int = None,
        ):
            """Renders a 2D segmentation map of what the camera sees."""
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

            u_val = np.clip(
                ((x_norm + 1) * 0.5) * (self.camera.width - 1),
                0,
                self.camera.width - 1,
            ).astype(int)
            v_val = np.clip(
                (1 - (y_norm + 1) * 0.5) * (self.camera.height - 1),
                0,
                self.camera.height - 1,
            ).astype(int)

            z_val = z_depth[valid]
            labels_val = labels[valid]

            pixel_idx = v_val * self.camera.width + u_val
            order = np.lexsort((z_val, pixel_idx))

            pixel_sorted = pixel_idx[order]
            labels_sorted = labels_val[order]

            first_mask = np.ones(len(pixel_sorted), dtype=bool)
            first_mask[1:] = pixel_sorted[1:] != pixel_sorted[:-1]

            image_grid = (
                np.ones((self.camera.height, self.camera.width), dtype=int) * -1
            )
            u_visible = pixel_sorted[first_mask] % self.camera.width
            v_visible = pixel_sorted[first_mask] // self.camera.width

            image_grid[v_visible, u_visible] = labels_sorted[first_mask]

            plt.figure(figsize=(8, 6))
            plt.imshow(image_grid, cmap="coolwarm", vmin=-1, vmax=1)
            
            title_str = "Camera Projection View"
            if view_id is not None:
                title_str += f" (Candidate View ID: {view_id})"
            plt.title(title_str)
            
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.colorbar(ticks=[-1, 0, 1], label="-1: Bg, 0: Occupied, 1: Unknown")
            plt.show()  # This blocks execution until the window is closed