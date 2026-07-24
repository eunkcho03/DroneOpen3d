import numpy as np
from scipy.spatial.transform import Rotation as R
from functions_camera import world_to_camera
from nbv_distance import compute_distance

def compute_lookat_quaternion(cam_pos, center, up=np.array([0, 1, 0])):
    forward = center - cam_pos
    norm_f = np.linalg.norm(forward)
    forward = forward / norm_f
        
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    rot_matrix = np.eye(3)
    rot_matrix[:, 0] = right
    rot_matrix[:, 1] = true_up
    rot_matrix[:, 2] = -forward 

    return R.from_matrix(rot_matrix).as_quat()

def candidate_view_radius(bbox_max, bbox_min, fov_deg, margin):

    r_obj = 0.5 * np.linalg.norm(bbox_max-bbox_min)

    return r_obj / np.sin(0.5 * np.deg2rad(fov_deg)) * margin



def generate_candidate_views(bbox_max, bbox_min, fov_deg, margin, 
                             n_elev, n_azi, min_height, max_elev_rad, 
                             center, azi_min_step_rad, elev_min_step_rad):
    r_obj = 0.5 * np.linalg.norm(bbox_max-bbox_min)
    r = r_obj / np.sin(0.5 * np.deg2rad(fov_deg)) * margin

    d_off = min_height - center[1]
    sin_val = np.clip(d_off / r, -1.0, 1.0)
    elev_min_rad = max(np.arcsin(sin_val), - np.pi/6) # -30 degrees lowest elevation to reach
    
    # minimum meaningful angular spacing
    elevation_span = max_elev_rad - elev_min_rad
    n_azi_eff = min(n_azi, max(1, int(np.floor(2*np.pi / azi_min_step_rad))))
    n_elev_eff = min(n_elev, max(1, int(np.floor(elevation_span / elev_min_step_rad))))    
    azi_rad = np.linspace(0, 2*np.pi, n_azi_eff, endpoint=False)

    if n_elev_eff == 1: 
        elev_rad = np.array([0.5 * (elev_min_rad + max_elev_rad)])
    else: 
        elev_rad = np.linspace(elev_min_rad, max_elev_rad, n_elev_eff)
    
    views = []
    view_id = 0
    # candidate views
    for elev in elev_rad:
        r_h = r * np.cos(elev)
        r_v = r * np.sin(elev)
        for azi in azi_rad:
            cam_pos = np.array([center[0] + r_h * np.cos(azi), 
                                center[1] + r_v, 
                                center[2] + r_h * np.sin(azi)
                            ], dtype=float)
            quat = compute_lookat_quaternion(cam_pos, center)
            views.append({
                'view_id': view_id,
                'pos': cam_pos, 
                'quat': quat
            })
            view_id = view_id + 1
    return views

def surface_points_labels(occupied_points_w, unknown_points_w):
    surface_points_w = np.vstack([occupied_points_w, unknown_points_w])
    labels = np.concatenate([
        np.zeros(len(occupied_points_w), dtype=np.int8),
        np.ones(len(unknown_points_w), dtype=np.int8)
    ])
    return surface_points_w, labels

def compute_gain(occupied_points_w, unknown_points_w, cam_pos, cam_quat, 
                 fov_deg, near, width, height):
    surface_points_w, labels = surface_points_labels(occupied_points_w, unknown_points_w)
    cam_sur = world_to_camera(surface_points_w, cam_pos, cam_quat)
    
    x = cam_sur[:, 0]
    y = cam_sur[:, 1]
    z_depth = cam_sur[:, 2]  # Depth positive in front of camera
    
    aspect_ratio = width / height
    theta_v = np.deg2rad(fov_deg) * 0.5
    theta_h = np.arctan(np.tan(theta_v) * aspect_ratio)
    
    # Filter valid visible points in view frustum
    valid = ((z_depth > near) 
             & (np.abs(x / z_depth) <= np.tan(theta_h))
             & (np.abs(y / z_depth) <= np.tan(theta_v)))

    x_norm = (x[valid] / z_depth[valid] / np.tan(theta_h))
    y_norm = (y[valid] / z_depth[valid] / np.tan(theta_v))
    
    u_val = np.clip(((x_norm + 1) * 0.5) * (width - 1), 0, width - 1).astype(int)
    v_val = np.clip((1 - (y_norm + 1) * 0.5) * (height - 1), 0, height - 1).astype(int)
    
    z_val = z_depth[valid]
    labels_val = labels[valid]
    
    # Z-buffering logic per pixel
    pixel_idx = v_val * width + u_val 
    order = np.lexsort((z_val, pixel_idx))
    
    pixel_sorted = pixel_idx[order]
    labels_sorted = labels_val[order]
    
    # Keep only closest point per pixel
    first_mask = np.ones(len(pixel_sorted), dtype=bool)
    first_mask[1:] = pixel_sorted[1:] != pixel_sorted[:-1]
    
    visible_labels = labels_sorted[first_mask]
    visible_unknown = int(np.count_nonzero(visible_labels == 1))
    visible_known = int(np.count_nonzero(visible_labels == 0))
    
    total_visible = visible_unknown + visible_known
    #gain_ratio = (visible_unknown / total_visible) if total_visible > 0 else 0.0
    
    return visible_unknown

# choose_best_nbv
def nbv_main(bbox_max, bbox_min, fov_deg, margin, 
            n_elev, n_azi, min_height, max_elev_rad, 
            center, azi_min_step_rad, elev_min_step_rad, 
            occupied_points_w, unknown_points_w, near, width, height, cam_cur_pos, fac_infl, distance_penalty):
    views = generate_candidate_views(bbox_max, bbox_min, fov_deg, margin, 
                             n_elev, n_azi, min_height, max_elev_rad, 
                             center, azi_min_step_rad, elev_min_step_rad)
    best_score = 0
    best_view_id = 0
    for view in views:
        cam_can_pos = view['pos']
        quat = view['quat']
        view_id = view['view_id']
        
        gain_abs = compute_gain(occupied_points_w, unknown_points_w, 
                                cam_can_pos, quat, fov_deg, near, width, height)
        dist = compute_distance(cam_cur_pos, cam_can_pos, bbox_min, bbox_max, fac_infl)
        if distance_penalty:
            score = gain_abs /dist
        else:
            score = gain_abs
        
        if score > best_score:
            best_score = score
            best_view_id = view_id
        
    return best_score, best_view_id, 
            