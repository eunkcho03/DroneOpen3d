import numpy as np
import os
from backup.nbv_functions import compute_next_best_view
from functions_print import (
    plot_voxel_centers_3d,
    print_depth_info,
    plot_unknown_surface_voxel_history,
    plot_total_unknown_occupied_vs_true_volume,
    plot_cum_nbv_gain_distance_history_one,
    plot_cum_nbv_gain_distance_history_sbs,
    save_nbv_history_to_excel,
    save_volume_history_to_excel,
    plot_nbv_evaluation_history,
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

from nbv.nbv_candidate_generation import generate_candidate_views_from_bbox
from nbv.nbv_planning import compute_next_best_view
from path_plotting import plot_all_nbv_positions_3d


HOST = "127.0.0.1"
DEPTH_PORT = 9010
NBV_HOST = "127.0.0.1"
NBV_PORT = 9020

FLOOR_HEIGHT = 0.0
OBJECT_HEIGHT_THRESHOLD = 1e-3
VOXEL_SIZE = 0.005
BBOX_MARGIN = 0.05

CARVE_MARGIN = 0
CARVE_WITH_INVALID_DEPTH = True

USE_NBV = True
NBV_NUM_VIEWS = 8
NBV_INFLATION_FACTOR = 1.5
NBV_MARGIN_FACTOR = 1.2
NBV_LAMBDA_DISTANCE = 5.0
NBV_SAVE_DEBUG_IMAGES = False
NBV_OUTPUT_FOLDER = "potential_views_debug"
NBV_PROJECTION_WIDTH = 160
NBV_PROJECTION_HEIGHT = 160
NBV_ASPECT_RATIO = 16.0 / 9.0
NBV_USE_DISTANCE_PENALTY = False
NBV_POSITION_TOLERANCE = 0.03
NBV_STABLE_FRAMES_REQUIRED = 1

SEND_NBV_TO_UNITY = True
PLOT_PATH = False  
PLOT_VOXEL_CENTERS = False 
PLOT_HISTORY = True
PLOT_3D_PATHS = False
SAVE_HISTORY_TO_EXCEL = False


def is_drone_at_target(current_position, target_position, tolerance):
    current_position = np.asarray(current_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)

    distance = np.linalg.norm(current_position - target_position)

    return distance <= tolerance, distance

# Early stopping
UNKNOWN_SURFACE_RATIO_THRESHOLD = 0.03
VOLUME_CHANGE_THRESHOLD = 0.03
VOLUME_STABLE_LIMIT = 2


def main():
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)
    nbv_sock = None

    if SEND_NBV_TO_UNITY:
        try:
            nbv_sock = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
            print(
                f"Connected to Unity NBV receiver on "
                f"{NBV_HOST}:{NBV_PORT}"
            )
        except Exception as e:
            print(f"Could not connect to Unity NBV receiver: {e}")

    frame_count = 0
    detected_view_count = 0

    previous_volume_estimate = None
    stable_volume_count = 0

    unknown_surface_counts = []
    unknown_surface_ratio_history = []

    volume_view_numbers = []
    total_unknown_occupied_volume_history = []
    true_volume_for_total_history = []

    nbv_distance = []
    nbv_gain = []
    nbv_all_paths = []
    nbv_simple_paths = []

    visited_view_ids = set()

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

            depth = receive_depth(
                depth_conn,
                width,
                height,
            )

            if depth is None:
                break

            position = (pos_x, pos_y, pos_z)
            quaternion = (rot_x, rot_y, rot_z, rot_w)

            current_position_np = np.asarray(
                position,
                dtype=float,
            )

            current_aspect_ratio = (
                float(width) / float(height)
                if height > 0
                else NBV_ASPECT_RATIO
            )

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

            # --------------------------------------------------
            # Wait until the drone reaches the previous NBV
            # --------------------------------------------------
            if waiting_for_nbv_move:
                arrived, distance_to_target = is_drone_at_target(
                    current_position=current_position_np,
                    target_position=pending_nbv_position,
                    tolerance=NBV_POSITION_TOLERANCE,
                )

                print(
                    "Waiting for drone to reach NBV target... "
                    f"distance = {distance_to_target:.4f} m"
                )

                if not arrived:
                    stable_target_frame_count = 0

                    print(
                        "Drone still moving. "
                        "Skipping reconstruction update."
                    )

                    frame_count += 1
                    continue

                stable_target_frame_count += 1

                print(
                    "Drone is near target. Stable frame "
                    f"{stable_target_frame_count}/"
                    f"{NBV_STABLE_FRAMES_REQUIRED}"
                )

                if (
                    stable_target_frame_count
                    < NBV_STABLE_FRAMES_REQUIRED
                ):
                    frame_count += 1
                    continue

                print(
                    "Drone reached NBV target. "
                    "Reconstruction update enabled."
                )

                waiting_for_nbv_move = False
                pending_nbv_position = None
                stable_target_frame_count = 0

            # --------------------------------------------------
            # Process depth frame
            # --------------------------------------------------
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
            # Reconstruction update
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
                f"\nReconstruction updated with detected view "
                f"{detected_view_count}"
                f"\nCamera position used: {position}"
                "\n===================================================="
            )

            if PLOT_VOXEL_CENTERS:
                plot_voxel_centers_3d(
                    occupied_centers=recon.occupied_centers,
                    unknown_centers=recon.unknown_centers,
                    bbox_corners=recon.bbox_corners,
                    title=(
                        "Occupied and Unknown Voxels - "
                        f"View {detected_view_count}"
                    ),
                )

            # --------------------------------------------------
            # Volume history
            # --------------------------------------------------
            total_unknown_occupied_volume = (
                recon.occupied_volume
                + recon.unknown_volume
            )

            volume_view_numbers.append(
                detected_view_count
            )

            total_unknown_occupied_volume_history.append(
                total_unknown_occupied_volume
            )

            true_volume_for_total_history.append(
                true_volume
            )

            print(
                f"Volume summary | "
                f"occupied: {recon.occupied_volume:.6f} m³ | "
                f"unknown: {recon.unknown_volume:.6f} m³ | "
                f"occupied + unknown: "
                f"{total_unknown_occupied_volume:.6f} m³ | "
                f"true Unity volume: {true_volume:.6f} m³"
            )

            # --------------------------------------------------
            # Volume stability
            # --------------------------------------------------
            relative_change = None

            if previous_volume_estimate is not None:
                relative_change = (
                    abs(
                        total_unknown_occupied_volume
                        - previous_volume_estimate
                    )
                    / max(
                        abs(previous_volume_estimate),
                        1e-12,
                    )
                )

                if (
                    relative_change
                    < VOLUME_CHANGE_THRESHOLD
                ):
                    stable_volume_count += 1
                else:
                    stable_volume_count = 0

                print(
                    f"Volume change: "
                    f"{relative_change * 100:.3f}% | "
                    f"Stable estimates: "
                    f"{stable_volume_count}/"
                    f"{VOLUME_STABLE_LIMIT}"
                )

            previous_volume_estimate = (
                total_unknown_occupied_volume
            )

            volume_stable = (
                stable_volume_count
                >= VOLUME_STABLE_LIMIT
            )

            # --------------------------------------------------
            # Surface-completeness stopping criterion
            # --------------------------------------------------
            unknown_surface_count = (
                recon.count_unknown_surface_voxels()
            )

            occupied_surface_count = (
                recon.count_occupied_surface_voxels()
            )

            total_surface_count = (
                unknown_surface_count
                + occupied_surface_count
            )

            unknown_surface_ratio = (
                unknown_surface_count
                / max(total_surface_count, 1)
            )

            unknown_surface_counts.append(
                unknown_surface_count
            )

            unknown_surface_ratio_history.append(
                unknown_surface_ratio
            )

            surface_complete = (
                unknown_surface_ratio
                < UNKNOWN_SURFACE_RATIO_THRESHOLD
            )

            no_unknown_surface = (
                unknown_surface_count == 0
            )

            print(
                f"Surface summary | "
                f"unknown surface: "
                f"{unknown_surface_count} | "
                f"occupied surface: "
                f"{occupied_surface_count} | "
                f"total surface: "
                f"{total_surface_count} | "
                f"unknown ratio: "
                f"{unknown_surface_ratio * 100:.3f}%"
            )

            print(
                f"Stopping checks | "
                f"surface complete: {surface_complete} | "
                f"volume stable: {volume_stable}"
            )

            should_stop = (
                    volume_stable or surface_complete
            )

            stop_reason = None

            # --------------------------------------------------
            # NBV selection
            # Compute final NBV for evaluation, but do not send it
            # when should_stop is True.
            # --------------------------------------------------
            best_view = None
            best_score = None
            path_to_best_view = None

            candidate_views_exhausted = False

            if (
                hasattr(recon, "unknown_centers")
                and len(recon.unknown_centers) > 0
            ):
                object_center = 0.5 * (
                    recon.bbox_min
                    + recon.bbox_max
                )

                candidate_views = (
                    generate_candidate_views_from_bbox(
                        bbox_min=recon.bbox_min,
                        bbox_max=recon.bbox_max,
                        fov_degrees=fov,
                        num_azimuth_views=NBV_NUM_VIEWS,
                        height_fractions=(
                            0.2,
                            0.60,
                            1.0,
                            1.2,
                        ),
                        margin_factor=NBV_MARGIN_FACTOR,
                    )
                )

                candidate_views_exhausted = (
                    len(visited_view_ids)
                    >= len(candidate_views)
                )

                if not candidate_views_exhausted:
                    (
                        best_view,
                        best_score,
                        distance,
                        gain,
                        path_to_best_view,
                    ) = compute_next_best_view(
                        occupied_voxels=(
                            recon.occupied_centers
                        ),
                        unknown_voxels=(
                            recon.unknown_centers
                        ),
                        object_center=object_center,
                        candidate_views=candidate_views,
                        current_camera_position=(
                            current_position_np
                        ),
                        visited_view_ids=(
                            visited_view_ids
                        ),
                        lambda_distance=(
                            NBV_LAMBDA_DISTANCE
                        ),
                        fov_degrees=fov,
                        image_width=(
                            NBV_PROJECTION_WIDTH
                        ),
                        image_height=(
                            NBV_PROJECTION_HEIGHT
                        ),
                        aspect_ratio=(
                            current_aspect_ratio
                        ),
                        voxel_size=VOXEL_SIZE,
                        use_distance_penalty=(
                            NBV_USE_DISTANCE_PENALTY
                        ),
                        save_debug_images=(
                            NBV_SAVE_DEBUG_IMAGES
                        ),
                        output_folder=(
                            NBV_OUTPUT_FOLDER
                        ),
                        print_scores=True,
                        bbox_inflation_factor=(
                            NBV_INFLATION_FACTOR
                        ),
                        b_min=recon.bbox_min,
                        b_max=recon.bbox_max,
                        plot_path=PLOT_PATH,
                    )

                    nbv_distance.append(distance)
                    nbv_gain.append(gain)

                    if path_to_best_view is not None:
                        nbv_all_paths.extend(
                            path_to_best_view
                        )

                        if len(nbv_simple_paths) == 0:
                            nbv_simple_paths.extend(
                                [
                                    path_to_best_view[0],
                                    path_to_best_view[-1],
                                ]
                            )
                        else:
                            nbv_simple_paths.append(
                                path_to_best_view[-1]
                            )

                    if PLOT_HISTORY:
                        plot_nbv_evaluation_history(
                            view_numbers=(
                                volume_view_numbers
                            ),
                            nbv_gains=nbv_gain,
                            nbv_distances=nbv_distance,
                            estimated_volumes=(
                                total_unknown_occupied_volume_history
                            ),
                            true_volumes=(
                                true_volume_for_total_history
                            ),
                            output_file_path=None,
                        )

                    if SAVE_HISTORY_TO_EXCEL:
                        os.makedirs(
                            "history",
                            exist_ok=True,
                        )

                        save_nbv_history_to_excel(
                            view_numbers=(
                                volume_view_numbers
                            ),
                            nbv_gains=nbv_gain,
                            nbv_distances=nbv_distance,
                            output_file_path=(
                                "history/"
                                "nbv_history_DPEN_"
                                f"{NBV_USE_DISTANCE_PENALTY}"
                                ".xlsx"
                            ),
                        )

                        save_volume_history_to_excel(
                            view_numbers=(
                                volume_view_numbers
                            ),
                            total_unknown_occupied_volumes=(
                                total_unknown_occupied_volume_history
                            ),
                            true_volumes=(
                                true_volume_for_total_history
                            ),
                            output_file_path=(
                                "history/"
                                "volume_history_DPEN_"
                                f"{NBV_USE_DISTANCE_PENALTY}"
                                ".xlsx"
                            ),
                        )

                    if (
                        PLOT_3D_PATHS
                        and path_to_best_view is not None
                    ):
                        plot_all_nbv_positions_3d(
                            bbox_min=recon.bbox_min,
                            bbox_max=recon.bbox_max,
                            inflation_factor=(
                                NBV_INFLATION_FACTOR
                            ),
                            nbv_waypoints=(
                                nbv_all_paths
                            ),
                            elevation=25,
                            azimuth=-55,
                            output_file_path=None,
                        )

                        plot_all_nbv_positions_3d(
                            bbox_min=recon.bbox_min,
                            bbox_max=recon.bbox_max,
                            inflation_factor=(
                                NBV_INFLATION_FACTOR
                            ),
                            nbv_waypoints=(
                                nbv_simple_paths
                            ),
                            elevation=25,
                            azimuth=-55,
                            output_file_path=None,
                        )

            # Stop if no unvisited candidate remains.
            if candidate_views_exhausted:
                should_stop = True
                stop_reason = (
                    "All generated candidate views "
                    "have been visited."
                )

            # --------------------------------------------------
            # Save final results and stop
            # --------------------------------------------------
            if should_stop:
                print(
                    "\n===================================================="
                    "\nStopping reconstruction:"
                    f"\nReason: {stop_reason}"
                    f"\nUnknown surface ratio: "
                    f"{unknown_surface_ratio * 100:.3f}%"
                    f"\nVolume stable count: "
                    f"{stable_volume_count}/"
                    f"{VOLUME_STABLE_LIMIT}"
                    f"\nFinal detected view: "
                    f"{detected_view_count}"
                    "\nThe final NBV was computed for evaluation "
                    "but was not sent."
                    "\n===================================================="
                )

                results_folder = "results"

                os.makedirs(
                    results_folder,
                    exist_ok=True,
                )

                plot_nbv_evaluation_history(
                    view_numbers=volume_view_numbers,
                    nbv_gains=nbv_gain,
                    nbv_distances=nbv_distance,
                    estimated_volumes=(
                        total_unknown_occupied_volume_history
                    ),
                    true_volumes=(
                        true_volume_for_total_history
                    ),
                    output_file_path=os.path.join(
                        results_folder,
                        "nbv_evaluation_history.png",
                    ),
                )

                plot_all_nbv_positions_3d(
                    bbox_min=recon.bbox_min,
                    bbox_max=recon.bbox_max,
                    inflation_factor=(
                        NBV_INFLATION_FACTOR
                    ),
                    nbv_waypoints=nbv_all_paths,
                    elevation=25,
                    azimuth=-55,
                    output_file_path=os.path.join(
                        results_folder,
                        "complete_nbv_path.png",
                    ),
                )

                plot_all_nbv_positions_3d(
                    bbox_min=recon.bbox_min,
                    bbox_max=recon.bbox_max,
                    inflation_factor=(
                        NBV_INFLATION_FACTOR
                    ),
                    nbv_waypoints=nbv_simple_paths,
                    elevation=25,
                    azimuth=-55,
                    output_file_path=os.path.join(
                        results_folder,
                        "nbv_positions_path.png",
                    ),
                )

                break

            # --------------------------------------------------
            # Send the next NBV only if reconstruction continues
            # --------------------------------------------------
            if best_view is not None:
                visited_view_ids.add(
                    best_view["view_id"]
                )

                if (
                    SEND_NBV_TO_UNITY
                    and nbv_sock is not None
                ):
                    send_next_view(
                        nbv_sock,
                        best_view,
                        best_score,
                    )

                    pending_nbv_position = np.asarray(
                        best_view["position"],
                        dtype=float,
                    )

                    waiting_for_nbv_move = True
                    stable_target_frame_count = 0

                    print(
                        "\nSent NBV target to Unity."
                        f"\nWaiting for drone to move to: "
                        f"{pending_nbv_position}"
                        "\nReconstruction updates are paused "
                        "until arrival."
                    )

            frame_count += 1

    finally:
        if nbv_sock is not None:
            nbv_sock.close()

        depth_conn.close()
        depth_server.close()

if __name__ == "__main__":
    main()