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
from functions_compute import DepthFrameProcessor, Open3DVoxelReconstruction
from nbv_functions import generate_candidate_views_from_bbox, compute_next_best_view


HOST = "127.0.0.1"
DEPTH_PORT = 9010
NBV_HOST = "127.0.0.1"
NBV_PORT = 9020

FLOOR_HEIGHT = 0.0
OBJECT_HEIGHT_THRESHOLD = 1e-3
VOXEL_SIZE = 0.005
BBOX_MARGIN = 0.05
CARVE_MARGIN = 0.5 * VOXEL_SIZE
CARVE_WITH_INVALID_DEPTH = False

PLOT_EVERY_FRAME = True
PLOT_MODE = "both"  # "occupied", "unknown", or "both"

USE_NBV = True
SEND_NBV_TO_UNITY = True
NBV_NUM_VIEWS = 18
NBV_DISTANCE_FACTOR = 2
NBV_HEIGHT_FRACTION = 0.8
NBV_LAMBDA_DISTANCE = 1.0
NBV_DISTANCE_PENALTY_MODE = "ratio"


def print_candidate_views(candidate_views):
    print("\n--- Candidate Views ---")
    for view in candidate_views:
        print(
            f"View {view['view_id']:02d} | angle = {view['angle_deg']:.1f} deg | "
            f"gain = {view.get('gain')} | distance = {view.get('distance')} | score = {view.get('score')}"
        )


def connect_nbv_socket():
    if not SEND_NBV_TO_UNITY:
        return None
    try:
        sock = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
        print(f"Connected to Unity NBV receiver on {NBV_HOST}:{NBV_PORT}")
        return sock
    except Exception as error:
        print("NBV socket not connected. NBV will be computed but not sent.")
        print("Error:", error)
        return None


def maybe_send_nbv(nbv_sock, best_view, best_score):
    if SEND_NBV_TO_UNITY and nbv_sock is not None:
        send_next_view(nbv_sock, best_view, best_score)
        print("NBV result sent to Unity.")


def compute_and_send_nbv(recon, position, visited_view_ids, nbv_sock):
    if not USE_NBV or len(recon.unknown_centers) == 0:
        return

    object_center = 0.5 * (recon.bbox_min + recon.bbox_max)
    candidate_views = generate_candidate_views_from_bbox(
        bbox_min=recon.bbox_min,
        bbox_max=recon.bbox_max,
        num_views=NBV_NUM_VIEWS,
        distance_factor=NBV_DISTANCE_FACTOR,
        height_fraction=NBV_HEIGHT_FRACTION,
    )

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
        print("NBV skipped: no valid unvisited candidate view.")
        return

    print("\n--- Next Best View ---")
    print("Best view ID:", best_view["view_id"])
    print("Angle deg:", best_view["angle_deg"])
    print("Position:", best_view["position"])
    print("Look at:", best_view["look_at"])
    print("Gain:", best_view.get("gain"))
    print("Distance:", best_view.get("distance"))
    print("NBV score:", best_score)

    visited_view_ids.add(best_view["view_id"])
    maybe_send_nbv(nbv_sock, best_view, best_score)


def plot_reconstruction(recon):
    if not PLOT_EVERY_FRAME:
        return

    occupied = recon.fused_centers if PLOT_MODE in ["occupied", "both"] else None
    unknown = recon.unknown_centers if PLOT_MODE in ["unknown", "both"] else None

    plot_voxel_centers_3d(
        occupied_centers=occupied,
        unknown_centers=unknown,
        bbox_corners=recon.bbox_corners,
        title="Open3D Occupied and Unknown Voxels",
    )


def main():
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)
    nbv_sock = connect_nbv_socket()

    recon = Open3DVoxelReconstruction(
        voxel_size=VOXEL_SIZE,
        bbox_margin=BBOX_MARGIN,
        floor_height=FLOOR_HEIGHT,
        verbose=True,
    )

    frame_count = 0
    detected_view_count = 0
    visited_view_ids = set()

    unknown_surface_view_numbers = []
    unknown_surface_counts = []
    volume_view_numbers = []
    total_unknown_occupied_volume_history = []
    true_volume_for_total_history = []

    try:
        while True:
            print("\nWaiting for Unity to send a depth frame...")
            header = receive_header(depth_conn)
            if header is None:
                break

            (
                width, height, fov, near, far,
                pos_x, pos_y, pos_z,
                rot_x, rot_y, rot_z, rot_w,
                true_volume,
            ) = header

            depth = receive_depth(depth_conn, width, height)
            if depth is None:
                break

            position = (pos_x, pos_y, pos_z)
            quaternion = (rot_x, rot_y, rot_z, rot_w)

            print("\n====================================================")
            print("Received frame:", frame_count)
            print("Detected object views:", detected_view_count)
            print("Camera position:", position)
            print("True Unity volume:", true_volume, "m³")
            print("Visited NBV view IDs:", sorted(visited_view_ids))
            print("====================================================")

            print_depth_info(depth, width, height, fov, near, far, position, quaternion)

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

            print("Camera-frame points:", len(frame.points_camera))
            print("World-frame points:", len(frame.points_world))
            print("Filtered object points:", len(frame.filtered_points))

            if len(frame.filtered_points) == 0:
                print("No object detected in this frame. Reconstruction unchanged.")
                frame_count += 1
                continue

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

            total_unknown_occupied_volume = recon.fused_volume + recon.unknown_volume
            unknown_surface_count = recon.count_unknown_surface_voxels()

            volume_view_numbers.append(detected_view_count)
            total_unknown_occupied_volume_history.append(total_unknown_occupied_volume)
            true_volume_for_total_history.append(true_volume if true_volume >= 0 else np.nan)

            unknown_surface_view_numbers.append(detected_view_count)
            unknown_surface_counts.append(unknown_surface_count)

            print("\nCurrent reconstruction state:")
            print("  Object views used:", detected_view_count)
            print("  Observed surface voxels:", len(recon.observed_indices))
            print("  Occupied voxels:", len(recon.occupied_indices))
            print("  Unknown voxels:", len(recon.unknown_indices))
            print("  Free voxels:", len(recon.free_indices))
            print("  True Unity volume:", round(true_volume, 6), "m³")
            print("  Occupied volume:", round(recon.fused_volume, 6), "m³")
            print("  Unknown volume:", round(recon.unknown_volume, 6), "m³")
            print("  Occupied + unknown volume:", round(total_unknown_occupied_volume, 6), "m³")
            print("  Expected volume:", round(recon.expected_volume, 6), "m³")
            print("  Unknown surface voxels:", unknown_surface_count)
            print("  Bounding box min:", recon.bbox_min)
            print("  Bounding box max:", recon.bbox_max)
            print("  Grid shape:", recon.grid_shape)

            plot_total_unknown_occupied_vs_true_volume(
                view_numbers=volume_view_numbers,
                total_unknown_occupied_volumes=total_unknown_occupied_volume_history,
                true_volumes=true_volume_for_total_history,
            )
            plot_unknown_surface_voxel_history(
                view_numbers=unknown_surface_view_numbers,
                unknown_surface_counts=unknown_surface_counts,
            )

            compute_and_send_nbv(recon, position, visited_view_ids, nbv_sock)
            plot_reconstruction(recon)

            frame_count += 1

    finally:
        if nbv_sock is not None:
            nbv_sock.close()
        depth_conn.close()
        depth_server.close()
        print("Connections closed.")


if __name__ == "__main__":
    main()
