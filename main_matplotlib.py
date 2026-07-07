import matplotlib.pyplot as plt
import numpy as np

from functions_print import (
    plot_voxel_centers_3d,
    print_depth_info,
    plot_unknown_surface_voxel_history,
    plot_total_unknown_occupied_vs_true_volume,
    plot_3d_scatter,
)

from functions_connect import (
    create_server,
    receive_header,
    receive_depth,
    send_next_view,
    connect_to_unity_nbv,
)

from functions_compute import (
    DepthFrameProcessor,
    ExtrusionFusionReconstruction,
)

from nbv_functions import (
    generate_candidate_views_from_bbox,
    compute_next_best_view,
)

HOST = "127.0.0.1"
DEPTH_PORT = 9010
NBV_HOST = "127.0.0.1"
NBV_PORT = 9020

FLOOR_HEIGHT = 0.0
OBJECT_HEIGHT_THRESHOLD = 1e-3
VOXEL_SIZE = 0.0025
BBOX_MARGIN = 0.05

CARVE_MARGIN = 0 # 0.5 * VOXEL_SIZE
CARVE_WITH_INVALID_DEPTH = True

USE_NBV = True
NBV_NUM_VIEWS = 8
NBV_MARGIN_FACTOR = 1.2
NBV_HEIGHT_FRACTION = 2
NBV_LAMBDA_DISTANCE = 1.5
NBV_SAVE_DEBUG_IMAGES = True
NBV_OUTPUT_FOLDER = "potential_views"
SEND_NBV_TO_UNITY = True

# --------------------------------------------------
# Movement gating settings
# --------------------------------------------------
NBV_POSITION_TOLERANCE = 0.03          # meters
NBV_STABLE_FRAMES_REQUIRED = 1      # require N frames near target before updating


def print_candidate_views(candidate_views):
    print("\n--- Candidate Views ---")
    for view in candidate_views:
        print(
            f"View {view['view_id']:02d} | "
            f"angle = {view['angle_deg']:.1f} deg | "
            f"score = {view.get('score', None)}"
        )


def is_drone_at_target(current_position, target_position, tolerance):
    current_position = np.asarray(current_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)

    distance = np.linalg.norm(current_position - target_position)
    return distance <= tolerance, distance


def main():
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)
    nbv_sock = None

    if SEND_NBV_TO_UNITY:
        try:
            nbv_sock = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
            print(f"Connected to Unity NBV receiver on {NBV_HOST}:{NBV_PORT}")
        except Exception as e:
            print(f"Could not connect to Unity NBV receiver: {e}")
            nbv_sock = None

    frame_count = 0
    detected_view_count = 0

    unknown_surface_view_numbers = []
    unknown_surface_counts = []

    volume_view_numbers = []
    total_unknown_occupied_volume_history = []
    true_volume_for_total_history = []

    visited_view_ids = set()

    # --------------------------------------------------
    # New NBV movement state
    # --------------------------------------------------
    waiting_for_nbv_move = False
    pending_nbv_position = None
    stable_target_frame_count = 0

    recon = ExtrusionFusionReconstruction(
        voxel_size=VOXEL_SIZE,
        bbox_margin=BBOX_MARGIN,
        floor_height=FLOOR_HEIGHT,
    )

    try:
        while True:
            header_data = receive_header(depth_conn)
            if header_data is None:
                break

            (
                width,
                height,
                fov,
                near,
                far,
                pos_x,
                pos_y,
                pos_z,
                rot_x,
                rot_y,
                rot_z,
                rot_w,
                true_volume,
            ) = header_data

            depth = receive_depth(depth_conn, width, height)
            if depth is None:
                break

            position = (pos_x, pos_y, pos_z)
            quaternion = (rot_x, rot_y, rot_z, rot_w)

            current_position_np = np.asarray(position, dtype=float)

            print_depth_info(
                depth,
                width,
                height,
                fov,
                near,
                far,
                position,
                quaternion,
            )

            if waiting_for_nbv_move:
                arrived, distance_to_target = is_drone_at_target(
                    current_position=current_position_np,
                    target_position=pending_nbv_position,
                    tolerance=NBV_POSITION_TOLERANCE,
                )

                print(
                    f"Waiting for drone to reach NBV target... "
                    f"distance = {distance_to_target:.4f} m"
                )

                if arrived:
                    stable_target_frame_count += 1
                    print(
                        f"Drone is near target. "
                        f"Stable frame {stable_target_frame_count}/"
                        f"{NBV_STABLE_FRAMES_REQUIRED}"
                    )

                    if stable_target_frame_count < NBV_STABLE_FRAMES_REQUIRED:
                        frame_count += 1
                        continue

                    print("Drone reached NBV target. Reconstruction update enabled.")
                    waiting_for_nbv_move = False
                    pending_nbv_position = None
                    stable_target_frame_count = 0

                else:
                    stable_target_frame_count = 0
                    print("Drone still moving. Skipping reconstruction update.")
                    frame_count += 1
                    continue

            frame = DepthFrameProcessor(
                depth=depth,
                fov_degrees=fov,
                far=far,
                position=position,
                quaternion=quaternion,
                floor_height=FLOOR_HEIGHT,
                height_threshold=OBJECT_HEIGHT_THRESHOLD,
            ).run()

            if len(frame.filtered_points) == 0:
                frame_count += 1
                continue

            # --------------------------------------------------
            # Reconstruction is only updated here.
            # Because of the waiting logic above, this only happens
            # after the drone has reached the requested NBV position.
            # --------------------------------------------------
            recon.add_view(
                points_world=frame.filtered_points,
                depth=depth,
                fov_degrees=fov,
                far=far,
                position=position,
                quaternion=quaternion,
                carve_margin=CARVE_MARGIN,
                carve_with_invalid_depth=CARVE_WITH_INVALID_DEPTH,
            )

            detected_view_count += 1

            print(
                "\n===================================================="
                f"\nReconstruction updated with detected view {detected_view_count}"
                f"\nCamera position used: {position}"
                "\n===================================================="
            )

            # --------------------------------------------------
            # Plot occupied and unknown voxel centers
            # --------------------------------------------------
            plot_voxel_centers_3d(
                 occupied_centers=recon.occupied_centers,
                 unknown_centers=recon.unknown_centers,
                 bbox_corners=recon.bbox_corners,
                 title=f"Occupied and Unknown Voxels - View {detected_view_count}",
             )

            volume_view_numbers.append(detected_view_count)

            total_unknown_occupied_volume = (
                recon.occupied_volume + recon.unknown_volume
            )

            print(
                f"Volume summary | "
                f"occupied: {recon.occupied_volume:.6f} m³ | "
                f"unknown: {recon.unknown_volume:.6f} m³ | "
                f"occupied + unknown: {total_unknown_occupied_volume:.6f} m³ | "
                f"true Unity volume: {true_volume:.6f} m³"
            )

            total_unknown_occupied_volume_history.append(
                total_unknown_occupied_volume
            )

            true_volume_for_total_history.append(
                true_volume if true_volume >= 0 else np.nan
            )

            plot_total_unknown_occupied_vs_true_volume(
                view_numbers=volume_view_numbers,
                total_unknown_occupied_volumes=total_unknown_occupied_volume_history,
                true_volumes=true_volume_for_total_history,
            )

            unknown_surface_count = recon.count_unknown_surface_voxels()

            unknown_surface_view_numbers.append(detected_view_count)
            unknown_surface_counts.append(unknown_surface_count)

            plot_unknown_surface_voxel_history(
                view_numbers=unknown_surface_view_numbers,
                unknown_surface_counts=unknown_surface_counts,
            )

            # --------------------------------------------------
            # NBV selection
            # --------------------------------------------------
            if (
                USE_NBV
                and hasattr(recon, "unknown_centers")
                and len(recon.unknown_centers) > 0
            ):
                object_center = 0.5 * (recon.bbox_min + recon.bbox_max)

                candidate_views = generate_candidate_views_from_bbox(
                        bbox_min=recon.bbox_min,
                        bbox_max=recon.bbox_max,
                        fov_degrees=fov,
                        num_azimuth_views=8,
                        height_fractions=(0.2, 0.60, 1.0, 1.2),
                        margin_factor=1.2,
                    )

                if len(visited_view_ids) < len(candidate_views):
                    best_view, best_score = compute_next_best_view(
                        occupied_voxels=recon.occupied_centers,
                        unknown_voxels=recon.unknown_centers,
                        object_center=object_center,
                        candidate_views=candidate_views,
                        current_camera_position=current_position_np,
                        visited_view_ids=visited_view_ids,
                        lambda_distance=NBV_LAMBDA_DISTANCE,
                        save_debug_images=NBV_SAVE_DEBUG_IMAGES,
                        output_folder=NBV_OUTPUT_FOLDER,
                    )

                    print_candidate_views(candidate_views)

                    if best_view is not None:
                        visited_view_ids.add(best_view["view_id"])

                        if SEND_NBV_TO_UNITY and nbv_sock is not None:
                            send_next_view(nbv_sock, best_view, best_score)

                            # --------------------------------------------------
                            # After sending NBV, block future reconstruction
                            # updates until Unity reports this new position.
                            # --------------------------------------------------
                            pending_nbv_position = np.asarray(best_view["position"], dtype=float)

                            waiting_for_nbv_move = True
                            stable_target_frame_count = 0

                            print(
                                "\nSent NBV target to Unity."
                                f"\nWaiting for drone to move to: {pending_nbv_position}"
                                "\nReconstruction updates are paused until arrival."
                            )

            frame_count += 1

    finally:
        if nbv_sock is not None:
            nbv_sock.close()

        depth_conn.close()
        depth_server.close()


if __name__ == "__main__":
    main()