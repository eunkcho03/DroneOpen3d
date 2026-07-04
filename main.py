import matplotlib.pyplot as plt
import numpy as np

from functions_print import (
    plot_voxel_centers_3d,
    print_depth_info,
    plot_unknown_surface_voxel_history,
    plot_total_unknown_occupied_vs_true_volume,
)

from functions_connect import (
    create_server,
    receive_header,
    receive_depth,
    send_next_view,
    connect_to_unity_nbv,
)

from functions_compute2 import Open3DOccupancyVoxelMap


from nbv_functions import (
    generate_candidate_views_from_bbox,
    compute_next_best_view,
)


# ---------------------------------------------------------
# Depth receiving settings
# Unity depth sender -> Python
# ---------------------------------------------------------
HOST = "127.0.0.1"
DEPTH_PORT = 9010


# ---------------------------------------------------------
# NBV sending settings
# Python -> Unity NBV receiver
# ---------------------------------------------------------
NBV_HOST = "127.0.0.1"
NBV_PORT = 9020


# ---------------------------------------------------------
# Reconstruction settings
# ---------------------------------------------------------
FLOOR_HEIGHT = 0.0
OBJECT_HEIGHT_THRESHOLD = 1e-3
VOXEL_SIZE = 0.005
BBOX_MARGIN = 0.05

PLOT_DEBUG_FIRST_DETECTED_FRAME = True
PLOT_EVERY_FRAME = True

# Half-voxel margin is usually stable for depth carving.
CARVE_MARGIN = 0.5 * VOXEL_SIZE

# Keep False first.
# True is more aggressive and may carve too much when Unity returns invalid/far depth.
CARVE_WITH_INVALID_DEPTH = False

# Plot mode:
# - "occupied": show only definitely occupied voxels
# - "unknown": show unknown voxels
# - "both": show occupied and unknown voxels together
PLOT_MODE = "both"


# ---------------------------------------------------------
# NBV settings
# ---------------------------------------------------------
USE_NBV = True

NBV_NUM_VIEWS = 18
NBV_DISTANCE_FACTOR = 2
NBV_HEIGHT_FRACTION = 0.8

# Distance-aware NBV settings
# Higher value means the drone prefers closer views more strongly.
NBV_LAMBDA_DISTANCE = 1.0

# Options:
# - "ratio":    score = gain / (1 + lambda * distance)
# - "subtract": score = gain - lambda * distance
# - "none":     score = gain
NBV_DISTANCE_PENALTY_MODE = "ratio"

# True only when Unity has NBVReceiver running on NBV_PORT.
SEND_NBV_TO_UNITY = True


def print_candidate_views(candidate_views):
    print("\n--- Candidate Views ---")
    for view in candidate_views:
        print(
            f"View {view['view_id']:02d} | "
            f"angle = {view['angle_deg']:.1f} deg | "
            f"position = {view['position']} | "
            f"look_at = {view['look_at']} | "
            f"gain = {view.get('gain', None)} | "
            f"distance = {view.get('distance', None)} | "
            f"score = {view.get('score', None)}"
        )


def main():
    # ---------------------------------------------------------
    # 1. Receive depth from Unity on DEPTH_PORT
    # ---------------------------------------------------------
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)

    # ---------------------------------------------------------
    # 2. Send NBV commands to Unity on NBV_PORT
    # ---------------------------------------------------------
    nbv_sock = None

    if SEND_NBV_TO_UNITY:
        try:
            nbv_sock = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
            print(f"Connected to Unity NBV receiver on {NBV_HOST}:{NBV_PORT}")
        except Exception as e:
            print("\nWARNING: Could not connect to Unity NBV receiver.")
            print("NBV computation will still run, but commands will not be sent.")
            print("Error:", e)
            nbv_sock = None

    frame_count = 0
    detected_view_count = 0

    # ---------------------------------------------------------
    # History for plots
    # ---------------------------------------------------------
    unknown_surface_view_numbers = []
    unknown_surface_counts = []

    volume_view_numbers = []
    total_unknown_occupied_volume_history = []
    true_volume_for_total_history = []

    # ---------------------------------------------------------
    # Keep track of which candidate NBV views have already been selected.
    # This prevents the same next-best-view from being selected repeatedly.
    # ---------------------------------------------------------
    visited_view_ids = set()

    recon = ExtrusionFusionReconstruction(
        voxel_size=VOXEL_SIZE,
        bbox_margin=BBOX_MARGIN,
        floor_height=FLOOR_HEIGHT,
        max_points=None,
        verbose=True,
    )

    try:
        while True:
            print("\nWaiting for Unity to send a depth frame...")

            header_data = receive_header(depth_conn)
            if header_data is None:
                print("No header received. Closing.")
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
                print("No depth frame received. Closing.")
                break

            # Replace invalid Unity depth value with far plane.
            depth = np.where(depth == -1, far, depth)

            position = (pos_x, pos_y, pos_z)
            quaternion = (rot_x, rot_y, rot_z, rot_w)

            print("\n====================================================")
            print("Received frame:", frame_count)
            print("Detected object views:", detected_view_count)
            print("Camera position:", position)
            print("Camera quaternion:", quaternion)
            print("True Unity volume:", true_volume, "m³")
            print("Visited NBV view IDs:", sorted(list(visited_view_ids)))
            print("====================================================")

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

            # ---------------------------------------------------------
            # Process current depth frame
            # ---------------------------------------------------------
            frame = DepthFrameProcessor(
                depth=depth,
                fov_degrees=fov,
                far=far,
                position=position,
                quaternion=quaternion,
                floor_height=FLOOR_HEIGHT,
                height_threshold=OBJECT_HEIGHT_THRESHOLD,
                verbose=True,
            ).run()

            print("Camera-frame points:   ", len(frame.points_camera))
            print("World-frame points:    ", len(frame.points_world))
            print("Filtered object points:", len(frame.filtered_points))

            # ---------------------------------------------------------
            # Object detection condition
            # ---------------------------------------------------------
            if len(frame.filtered_points) == 0:
                print("No object detected in this frame. Reconstruction unchanged.")
                frame_count += 1
                continue

            # ---------------------------------------------------------
            # Reconstruction update
            # ---------------------------------------------------------
            if not recon.initialized:
                print("\nObject detected. Initializing probabilistic bbox reconstruction...")
                print("Initial bbox voxels are UNKNOWN = 0.5, not occupied.")
                print("Directly observed surface voxels become OCCUPIED = 1.0.")
            else:
                print("\nObject detected. Updating probabilistic voxel grid using depth image...")
                print("Voxels in front of the measured depth become FREE = 0.0.")

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

            # ---------------------------------------------------------
            # Plot occupied + unknown volume against true Unity volume
            # ---------------------------------------------------------
            volume_view_numbers.append(detected_view_count)

            total_unknown_occupied_volume = recon.fused_volume + recon.unknown_volume
            total_unknown_occupied_volume_history.append(total_unknown_occupied_volume)

            if true_volume >= 0:
                true_volume_for_total_history.append(true_volume)
            else:
                true_volume_for_total_history.append(np.nan)

            print("  True Unity volume:            ", round(true_volume, 6), "m³")
            print("  Occupied volume:              ", round(recon.fused_volume, 6), "m³")
            print("  Unknown volume:               ", round(recon.unknown_volume, 6), "m³")
            print("  Occupied + unknown volume:    ", round(total_unknown_occupied_volume, 6), "m³")

            plot_total_unknown_occupied_vs_true_volume(
                view_numbers=volume_view_numbers,
                total_unknown_occupied_volumes=total_unknown_occupied_volume_history,
                true_volumes=true_volume_for_total_history,
            )

            # ---------------------------------------------------------
            # Plot unknown surface voxel count after each object view
            # ---------------------------------------------------------
            unknown_surface_count = recon.count_unknown_surface_voxels()

            unknown_surface_view_numbers.append(detected_view_count)
            unknown_surface_counts.append(unknown_surface_count)

            print("  Unknown surface voxels:       ", unknown_surface_count)

            plot_unknown_surface_voxel_history(
                view_numbers=unknown_surface_view_numbers,
                unknown_surface_counts=unknown_surface_counts,
            )

            # ---------------------------------------------------------
            # Print current reconstruction state
            # ---------------------------------------------------------
            print("\nCurrent reconstruction state:")
            print("  Raw frames received:          ", frame_count + 1)
            print("  Object views used:            ", detected_view_count)

            print("  Observed surface voxels:      ", len(recon.observed_indices))
            print("  Occupied voxels:              ", len(recon.occupied_indices))
            print("  Fused voxels:                 ", len(recon.fused_indices))

            if hasattr(recon, "unknown_indices"):
                print("  Unknown voxels:               ", len(recon.unknown_indices))

            if hasattr(recon, "free_indices"):
                print("  Free voxels:                  ", len(recon.free_indices))

            print("  Observed surface volume:      ", round(recon.observed_volume, 6), "m³")
            print("  Occupied volume:              ", round(recon.fused_volume, 6), "m³")

            if hasattr(recon, "expected_volume"):
                print("  Expected probabilistic volume:", round(recon.expected_volume, 6), "m³")

            if hasattr(recon, "unknown_volume"):
                print("  Unknown volume:               ", round(recon.unknown_volume, 6), "m³")

            if hasattr(recon, "free_volume"):
                print("  Free volume:                  ", round(recon.free_volume, 6), "m³")

            print("  Bounding box min:             ", recon.bbox_min)
            print("  Bounding box max:             ", recon.bbox_max)
            print("  Grid shape:                   ", recon.grid_shape)

            # ---------------------------------------------------------
            # Next Best View computation
            # ---------------------------------------------------------
            if USE_NBV:
                if hasattr(recon, "unknown_centers") and len(recon.unknown_centers) > 0:
                    object_center = 0.5 * (recon.bbox_min + recon.bbox_max)

                    candidate_views = generate_candidate_views_from_bbox(
                        bbox_min=recon.bbox_min,
                        bbox_max=recon.bbox_max,
                        num_views=NBV_NUM_VIEWS,
                        distance_factor=NBV_DISTANCE_FACTOR,
                        height_fraction=NBV_HEIGHT_FRACTION,
                    )

                    if len(visited_view_ids) >= len(candidate_views):
                        print("\nNBV skipped: all candidate views have already been visited.")

                    else:
                        best_view, best_score = compute_next_best_view(
                            unknown_voxels=recon.unknown_centers,
                            object_center=object_center,
                            candidate_views=candidate_views,
                            current_camera_position=position,
                            voxel_size=VOXEL_SIZE,
                            use_outer_unknown_only=True,
                            visited_view_ids=visited_view_ids,
                            lambda_distance=NBV_LAMBDA_DISTANCE,
                            distance_penalty_mode=NBV_DISTANCE_PENALTY_MODE,
                        )

                        print_candidate_views(candidate_views)

                        if best_view is None:
                            print("\nNBV skipped: no valid unvisited candidate view found.")

                        else:
                            print("\n--- Next Best View ---")
                            print("Best view ID:  ", best_view["view_id"])
                            print("Angle deg:     ", best_view["angle_deg"])
                            print("Position:      ", best_view["position"])
                            print("Look at:       ", best_view["look_at"])
                            print("Gain:          ", best_view.get("gain"))
                            print("Distance:      ", best_view.get("distance"))
                            print("NBV score:     ", best_score)

                            if "rotation_quat" in best_view:
                                print("Rotation quat: ", best_view["rotation_quat"])

                            visited_view_ids.add(best_view["view_id"])

                            print("Updated visited NBV view IDs:", sorted(list(visited_view_ids)))

                            if SEND_NBV_TO_UNITY:
                                if nbv_sock is not None:
                                    try:
                                        print("\nSending NBV result to Unity...")
                                        send_next_view(nbv_sock, best_view, best_score)
                                        print("NBV result sent.")
                                    except Exception as e:
                                        print("\nWARNING: Failed to send NBV result to Unity.")
                                        print("Error:", e)
                                        nbv_sock = None
                                else:
                                    print("\nNBV not sent: no active NBV socket.")

                else:
                    print("\nNBV skipped: recon.unknown_centers is not available or empty.")

            # ---------------------------------------------------------
            # Plot reconstruction
            # ---------------------------------------------------------
            if PLOT_EVERY_FRAME:
                if PLOT_MODE == "occupied":
                    print("Plotting occupied voxels only...")
                    plot_voxel_centers_3d(
                        occupied_centers=recon.fused_centers,
                        bbox_corners=recon.bbox_corners,
                        title="Occupied Voxels",
                    )

                elif PLOT_MODE == "unknown":
                    if hasattr(recon, "unknown_centers"):
                        print("Plotting unknown voxels only...")
                        plot_voxel_centers_3d(
                            unknown_centers=recon.unknown_centers,
                            bbox_corners=recon.bbox_corners,
                            title="Unknown Voxels",
                        )
                    else:
                        print("unknown_centers is not available. Are you using functions_compute_probabilistic?")

                elif PLOT_MODE == "both":
                    if hasattr(recon, "unknown_centers"):
                        print("Plotting occupied and unknown voxels in one figure...")
                        plot_voxel_centers_3d(
                            occupied_centers=recon.fused_centers,
                            unknown_centers=recon.unknown_centers,
                            bbox_corners=recon.bbox_corners,
                            title="Occupied and Unknown Voxels",
                        )
                    else:
                        print("unknown_centers is not available. Plotting occupied voxels only.")
                        plot_voxel_centers_3d(
                            occupied_centers=recon.fused_centers,
                            bbox_corners=recon.bbox_corners,
                            title="Occupied Voxels",
                        )

                else:
                    print(f"Unknown PLOT_MODE = {PLOT_MODE}. Skipping plot.")

            frame_count += 1

    finally:
        if nbv_sock is not None:
            nbv_sock.close()

        depth_conn.close()
        depth_server.close()

        print("Connections closed.")


if __name__ == "__main__":
    main()