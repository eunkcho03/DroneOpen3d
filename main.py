import numpy as np
import os
from nbv.nbv_main import NextBestViewPlanner, CameraIntrinsics
from functions_compute import ExtrusionFusionReconstruction
from functions_camera import  filter_by_height
from functions_connect import (
    create_server,
    receive_header,
    receive_depth,
    send_next_view,
    connect_to_unity_nbv,
)
from functions_print import plot_voxel_centers_3d, plot_nbv_evaluation_history

# Unity connection
HOST = "127.0.0.1"
DEPTH_PORT = 9010
NBV_HOST = "127.0.0.1"
NBV_PORT = 9020

# input
VOXEL_SIZE = 0.005
BBOX_MARGIN = 0.05
FLOOR_HEIGHT = 1e-2
HEIGHT_THRESHOLD = 0.0
NBV_POSITION_TOLERANCE = 0.05

# nbv input
NBV_RADIUS_MARGIN = 1.0
NBV_MIN_HEIGHT = 0.0
NUM_ELEV = 3
NUM_AZIM = 8
FAC_INFL = 1.2
DISTANCE_PENALTY = False

# stopping condition
VOLUME_CHANGE_THRESHOLD = 0.03
VOLUME_STABLE_LIMIT = 2

# plotting option
PLOT_VOXEL_CENTERS = False
PLOT_INTERMEDIATE_RESULTS = True

def is_drone_at_target(current_position, target_position, tolerance):
    distance = np.linalg.norm(current_position - target_position)
    return distance <= tolerance, distance

def main():
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)
    nbv_socket = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
    print(f"Connected to Unity NBV receiver on "
          f"{NBV_HOST}:{NBV_PORT}")
    
    recon = ExtrusionFusionReconstruction(
        voxel_size=VOXEL_SIZE,
        bbox_margin=BBOX_MARGIN,
        floor_height=FLOOR_HEIGHT,
        min_z_height=HEIGHT_THRESHOLD,
    )
    waiting_for_nbv_move = False
    pending_nbv_position = None
    detected_view_count = 0
    stable_volume_count = 0
    
    # volume
    view_history = []
    volume_history = []
    true_volume_history = []    
    prev_volume = None
    
    # nbv history
    nbv_distance = []
    nbv_gain = []
    
    try:
        while True:
            width, height, fov, near, far, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, true_volume = receive_header(depth_conn)
            camera = CameraIntrinsics(fov_deg=fov, 
                                      near=near, 
                                      width=width,
                                      height=height)
            depth = receive_depth(depth_conn, width, height)
            position = np.asarray((pos_x, pos_y, pos_z), dtype=float)
            quat = (rot_x, rot_y, rot_z, rot_w)
            
            if not recon.initialized:
                filtered_points = filter_by_height(
                    depth=depth, 
                    fov_degrees=fov, 
                    far=far, 
                    position=position,
                    quaternion=quat, 
                    floor_height=FLOOR_HEIGHT,
                    height_threshold=HEIGHT_THRESHOLD
                )
                
                if len(filtered_points) == 0:
                    continue
                recon.initialize(filtered_points)
            
            if waiting_for_nbv_move:
                is_at_target, distance = is_drone_at_target(position, pending_nbv_position, NBV_POSITION_TOLERANCE)
                if not is_at_target:
                    print(f'Drone still moving... Distance to target: {distance:.3f}m')
                    continue
                else:
                    waiting_for_nbv_move = False
                    pending_nbv_position = None
            
            recon.carve_with_depth_image(
                depth=depth,
                fov_degrees=fov,
                far=far,
                position=position,
                quaternion=quat,
                floor_height=FLOOR_HEIGHT,
                height_threshold=HEIGHT_THRESHOLD
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
                    unknown_centers = recon.unknown_centers, 
                    bbox_corners = recon.bbox_corners,
                    title=(
                        "Occupied and Unknown Voxels - "
                        f"View {detected_view_count}"
                    ),
                )
            
            view_history.append(detected_view_count)
            volume_history.append(recon.occupied_volume + recon.unknown_volume)
            true_volume_history.append(true_volume)
            
            if prev_volume is not None:
                relative_change = abs(volume_history[-1]-prev_volume) /max(abs(prev_volume), 1e-12)
                
                if relative_change < VOLUME_CHANGE_THRESHOLD:
                    stable_volume_count +=1
                else:
                    stable_volume_count = 0
                
                print(
                    f"Volume change: "
                    f"{relative_change * 100:.3f}% | "
                    f"Stable estimates: "
                    f"{stable_volume_count}/"
                    f"{VOLUME_STABLE_LIMIT}"
                )
            prev_volume = volume_history[-1]
            
            if stable_volume_count >= VOLUME_STABLE_LIMIT:
                print(
                    "\n===================================================="
                    "\nStopping reconstruction"
                    "\n===================================================="
                )
                results_folder = "results"
                os.makedirs(results_folder, exist_ok=True)
                plot_nbv_evaluation_history(
                    view_numbers=view_history,
                    nbv_gains=nbv_gain,
                    nbv_distances=nbv_distance,
                    estimated_volumes=volume_history,
                    true_volumes=true_volume_history,
                    output_file_path=os.path.join(results_folder, 'nbv_evaluation_history.png')
                )
                break 
                            
            nbv = NextBestViewPlanner(
                camera=camera,
                bbox_min=recon.bbox_min,
                bbox_max=recon.bbox_max,
                center=recon.bbox_center,
            )
            recon.surface_centers()
            
            best_score, best_view, best_dist = nbv.select_best_view(
                occupied_points_w=recon.occupied_surface_centers,
                unknown_points_w=recon.unknown_surface_centers,
                cam_cur_pos=position,
                n_elev=NUM_ELEV,
                n_azi=NUM_AZIM,
                fac_infl=FAC_INFL,
                distance_penalty=DISTANCE_PENALTY,
                visualize_candidates=False,
            )

            nbv_gain.append(best_score)
            nbv_distance.append(best_dist)
            pending_nbv_position = np.asarray(best_view['pos'], dtype=float)
            waiting_for_nbv_move = True
            
            send_next_view(nbv_socket, best_view)

            print(
                "\nSent NBV target to Unity."
                f"\nWaiting for drone to move to: "
                f"{pending_nbv_position}"
                "\nReconstruction updates are paused until arrival"
            )
            
            if PLOT_INTERMEDIATE_RESULTS:
                plot_nbv_evaluation_history(
                    view_numbers=view_history,
                    nbv_gains=nbv_gain,
                    nbv_distances=nbv_distance,
                    estimated_volumes=volume_history,
                    true_volumes=true_volume_history,
                    output_file_path=None,
                )

            
    finally:
        nbv_socket.close()
        depth_conn.close()
        depth_server.close()

if __name__ == "__main__":
    main()