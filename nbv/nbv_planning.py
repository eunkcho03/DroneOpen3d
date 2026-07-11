import os
import numpy as np

from nbv.nbv_utils import as_points_array, look_at_quaternion
from nbv.nbv_surface import estimate_voxel_size_from_points, extract_surface_voxels
from nbv.nbv_projection import compute_surface_projection_gain
from nbv.nbv_debug_rendering import render_candidate_view_figure
from nbv.nbv_distance import inflated_bbox_shortest_path_distance, plot_inflated_bbox_path

def compute_next_best_view(
    occupied_voxels,
    unknown_voxels,
    object_center,
    candidate_views,
    bbox_inflation_factor,
    b_min,
    b_max,
    current_camera_position=None,
    visited_view_ids=None,
    lambda_distance=1.0,
    save_debug_images=False,
    output_folder="potential_views",
    fov_degrees=58.0,
    image_width=160,
    image_height=160,
    aspect_ratio=16.0 / 9.0,
    voxel_size=None,
    use_distance_penalty=False,
    print_scores=True,
    plot_path=False,
):

    if occupied_voxels is None:
        occupied_voxels = np.empty((0, 3))

    if unknown_voxels is None:
        unknown_voxels = np.empty((0, 3))

    occupied_voxels = as_points_array(occupied_voxels)
    unknown_voxels = as_points_array(unknown_voxels)
    object_center = np.asarray(object_center, dtype=float)

    if current_camera_position is not None:
        current_camera_position = np.asarray(current_camera_position, dtype=float)

    if visited_view_ids is None:
        visited_view_ids = set()
    else:
        visited_view_ids = set(visited_view_ids)

    if len(unknown_voxels) == 0:
        print("No unknown voxels left. NBV not needed.")
        return None, 0.0

    if voxel_size is None:
        if len(occupied_voxels) > 0 and len(unknown_voxels) > 0:
            voxel_size = estimate_voxel_size_from_points(
                np.vstack([occupied_voxels, unknown_voxels])
            )
        elif len(unknown_voxels) > 0:
            voxel_size = estimate_voxel_size_from_points(unknown_voxels)
        else:
            voxel_size = estimate_voxel_size_from_points(occupied_voxels)

    # Use shared origin so occupied and unknown voxel grids align.
    if len(occupied_voxels) > 0 and len(unknown_voxels) > 0:
        grid_origin = np.vstack([occupied_voxels, unknown_voxels]).min(axis=0)
    elif len(unknown_voxels) > 0:
        grid_origin = unknown_voxels.min(axis=0)
    elif len(occupied_voxels) > 0:
        grid_origin = occupied_voxels.min(axis=0)
    else:
        grid_origin = np.zeros(3)

    occupied_surface_voxels = extract_surface_voxels(
        occupied_voxels,
        voxel_size=voxel_size,
        origin=grid_origin,
    )

    unknown_surface_voxels = extract_surface_voxels(
        unknown_voxels,
        voxel_size=voxel_size,
        origin=grid_origin,
    )

    if len(unknown_surface_voxels) == 0:
        print("No unknown surface voxels left. NBV not needed.")
        return None, 0.0

    if save_debug_images:
        os.makedirs(output_folder, exist_ok=True)

    print("")
    print("====================================================")
    print("Surface-based NBV")
    print(f"Voxel size used: {voxel_size}")
    print(f"Occupied voxels: {len(occupied_voxels)}")
    print(f"Unknown voxels: {len(unknown_voxels)}")
    print(f"Occupied surface voxels: {len(occupied_surface_voxels)}")
    print(f"Unknown surface voxels: {len(unknown_surface_voxels)}")
    print("====================================================")

    best_view = None
    best_score = -np.inf

    if print_scores:
        print("")
        print("--- Candidate Views: Surface-Based Gain ---")

    for view in candidate_views:
        view_id = view["view_id"]

        if view_id in visited_view_ids:
            view.update({
                "gain_ratio": 0.0,
                "gain_score_raw": 0.0,
                "visible_unknown_surface_pixels": 0,
                "visible_known_surface_pixels": 0,
                "visible_total_surface_pixels": 0,
                "unknown_surface_in_fov": 0,
                "known_surface_in_fov": 0,
                "distance": np.inf,
                "score": -np.inf,
            })

            if print_scores:
                print(f"View {view_id:02d} | visited | score = -inf")

            continue

        camera_position = np.asarray(view["position"], dtype=float)
        look_at = np.asarray(view["look_at"], dtype=float) if "look_at" in view else object_center.copy()

        view["position"] = camera_position
        view["look_at"] = look_at
        view["rotation_quat"] = look_at_quaternion(
            camera_position=camera_position,
            target_position=look_at,
        )

        gain_result = compute_surface_projection_gain(
            occupied_surface_voxels=occupied_surface_voxels,
            unknown_surface_voxels=unknown_surface_voxels,
            camera_position=camera_position,
            look_at=look_at,
            fov_degrees=fov_degrees,
            image_width=image_width,
            image_height=image_height,
            aspect_ratio=aspect_ratio,
        )

        raw_gain_score = float(gain_result["gain_score"])
        gain_ratio = float(gain_result["gain_ratio"])

        visible_unknown = int(gain_result["visible_unknown_surface_pixels"])
        visible_known = int(gain_result["visible_known_surface_pixels"])
        visible_total = int(gain_result["visible_total_surface_pixels"])

        unknown_in_fov = int(gain_result["unknown_surface_in_fov"])
        known_in_fov = int(gain_result["known_surface_in_fov"])
            
        travel_distance, path = inflated_bbox_shortest_path_distance(
            start_position=current_camera_position,
            end_position=camera_position,
            b_min=b_min,
            b_max=b_max,
            fac=bbox_inflation_factor,
        )
        
        if plot_path:
            plot_inflated_bbox_path(
                start_position=current_camera_position, 
                end_position=camera_position, 
                bbox_min=b_min, 
                bbox_max=b_max, 
                path=path, 
                fac=bbox_inflation_factor, 
                total_distance = travel_distance, 
                show_direct_path=True
                )

        if use_distance_penalty:
            score = raw_gain_score / (1.0 + lambda_distance * travel_distance)
        else:
            score = raw_gain_score

        view.update({
            "gain_ratio": gain_ratio,
            "gain_score_raw": raw_gain_score,
            "visible_unknown_surface_pixels": visible_unknown,
            "visible_known_surface_pixels": visible_known,
            "visible_total_surface_pixels": visible_total,
            "unknown_surface_in_fov": unknown_in_fov,
            "known_surface_in_fov": known_in_fov,
            "distance": travel_distance,
            "score": score,
        })

        if print_scores:
            print(
                f"View {view_id:02d} | "
                f"angle = {view.get('angle_deg', 0.0):.1f} deg | "
                f"height_frac = {view.get('height_fraction', -1):.2f} | "
                f"visible unknown surface = {visible_unknown} | "
                f"visible known surface = {visible_known} | "
                f"ratio = {gain_ratio:.3f} | "
                f"score = {score:.4f} | "
                f"distance = {travel_distance:.2f}"
            )

        if save_debug_images:
            filepath = os.path.join(
                output_folder,
                f"candidate_{view_id:02d}_surface_unknown_{visible_unknown}_ratio_{gain_ratio:.3f}_score_{score:.4f}.png"
            )

            render_candidate_view_figure(
                occupied_voxels=occupied_surface_voxels,
                unknown_voxels=unknown_surface_voxels,
                camera_position=camera_position,
                look_at=look_at,
                zoom_margin=0.08,
                figsize=(6, 6),
                marker_size=2,
                save_path=filepath,
                dpi=150,
                close_after_save=True,
                show_camera_triangle=True,
            )

        if score > best_score:
            best_score = score
            best_view = view

    if best_view is None:
        return None, 0.0

    print("")
    print("====================================================")
    print("Selected NBV")
    print(f"View ID: {best_view['view_id']}")
    print(f"Angle: {best_view.get('angle_deg', 0.0):.1f} deg")
    print(f"Height fraction: {best_view.get('height_fraction', -1):.2f}")
    print(f"Visible unknown surface pixels: {best_view['visible_unknown_surface_pixels']}")
    print(f"Visible known surface pixels: {best_view['visible_known_surface_pixels']}")
    print(f"Gain ratio: {best_view['gain_ratio']:.3f}")
    print(f"Best score: {best_score:.4f}")
    print("====================================================")
    print("")

    return best_view, best_score
