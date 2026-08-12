import numpy as np
import os
import time
from functions_compute import ExtrusionFusionReconstruction
from functions_camera import  filter_by_height, Initialization, compute_lookat_quaternion
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
VOXEL_SIZE = 0.015
BBOX_MARGIN = 0.05
FLOOR_HEIGHT = 0.0
HEIGHT_THRESHOLD = 1e-3
NBV_POSITION_TOLERANCE = 0.05
MAX_ELEV_DEG = 30


# fixed view and init parameters
NBV_MIN_HEIGHT = 1e-2
SIZE_MARGIN = 1.5 # Multiplier to ensure the entire object fits within the camera FOV
MIN_ALT_BUFFER = 1.2

# Object Type
OBJECT_TYPE = "cylinder_0.1m"

# plotting option
PLOT_VOXEL_CENTERS = False
PLOT_INTERMEDIATE_RESULTS = False

def is_drone_at_target(current_position, target_position, tolerance):
    distance = np.linalg.norm(current_position - target_position)
    return distance <= tolerance, distance

def main():
    depth_server, depth_conn = create_server(HOST, DEPTH_PORT)
    nbv_socket = connect_to_unity_nbv(NBV_HOST, NBV_PORT)
    print(f"Connected to Unity view receiver on {NBV_HOST}:{NBV_PORT}")
    
    recon = ExtrusionFusionReconstruction(
        voxel_size_ratio=VOXEL_SIZE,
        bbox_margin=BBOX_MARGIN,
        floor_height=FLOOR_HEIGHT,
        min_z_height=HEIGHT_THRESHOLD,
    )
    
    initial = Initialization(
        min_height = NBV_MIN_HEIGHT,
        max_elev_deg = MAX_ELEV_DEG,
    )
    
    waiting_for_camera_move = False
    pending_camera_position = None
    detected_view_count = 0
    
    # volume and history tracking
    view_history = []
    volume_history = []
    unknown_voxel_history = []
    position_history = []
    refinement_time_history = []
    
    # initialisation state
    initial_target_sent = False
    initial_target_pos = None
    min_alt_buffer = MIN_ALT_BUFFER

    
    # fixed views generation state
    fixed_views = []
    views_generated = False
    
    try:
        while True:
            width, height, fov, near, far, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, true_volume = receive_header(depth_conn)
            position = np.asarray((pos_x, pos_y, pos_z), dtype=float)
            quat = (rot_x, rot_y, rot_z, rot_w)
            depth = receive_depth(depth_conn, width, height)
            
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
                    print("Warning: No points detected at spawn position. Cannot determine object size.")
                    continue
                
                initial_views = initial.initial_position(filtered_points, min_alt_buffer, position)
                send_next_view(nbv_socket, initial_views[0])

                
                is_at_target, distance = is_drone_at_target(position, initial_views[0]['pos'], NBV_POSITION_TOLERANCE)
                
                if not is_at_target:
                    print(f'Drone still moving to initial view... Distance to target: {distance:.3f}m')
                    continue
                
                print("Camera reached relative initial target.")
                                    
                recon.initialize(filtered_points)
                print('INITIAL_UNKNOWN_COUNT', recon.unknown_count)
                    
            if waiting_for_camera_move:
                is_at_target, distance = is_drone_at_target(position, pending_camera_position, NBV_POSITION_TOLERANCE)
                if not is_at_target:
                    print(f'Drone still moving... Distance to target: {distance:.3f}m')
                    continue
                else:
                    waiting_for_camera_move = False
                    pending_camera_position = None
            
            start_time = time.time()
            
            recon.carve_with_depth_image(
                depth=depth,
                fov_degrees=fov,
                far=far,
                position=position,
                quaternion=quat,
                floor_height=FLOOR_HEIGHT,
                height_threshold=HEIGHT_THRESHOLD
            )
            
            end_time = time.time()
            refinement_time = end_time - start_time
            refinement_time_history.append(refinement_time)
            
            unknown_voxel_history.append(recon.unknown_count)
            view_history.append(detected_view_count)
            position_history.append(position)
            volume_history.append(recon.occupied_volume + recon.unknown_volume)
            
            print(
                "\n===================================================="
                f"\nReconstruction updated with detected view {detected_view_count}"
                f"\nRefinement completed in {refinement_time:.4f} seconds"
                f"\nCamera position used: {position}"
                "\n===================================================="
            )
            
            # Generate fixed views based on the initialized precise bounding box
            if not views_generated:
                center = recon.bbox_center
                extents = recon.bbox_max - recon.bbox_min
                max_dim = np.max(extents)
                
                fov_rad = np.radians(fov)
                orbit_radius = (max_dim / 2.0) / np.tan(fov_rad / 2.0) * SIZE_MARGIN
                orbit_radius = max(orbit_radius, 0.5)
                view_id_counter = 1
                for el in [30, 0, -30]:
                    for az in range(0, 360, 45):
                        el_rad = np.radians(el)
                        az_rad = np.radians(az)
                        
                        vx = center[0] + orbit_radius * np.cos(el_rad) * np.cos(az_rad)
                        vz = center[2] + orbit_radius * np.cos(el_rad) * np.sin(az_rad)
                        vy = center[1] + orbit_radius * np.sin(el_rad)
                        
                        vy = max(vy, NBV_MIN_HEIGHT) 
                        cam_pos = [vx, vy, vz]
                        target_quat = compute_lookat_quaternion(np.array(cam_pos), center)
                        
                        fixed_views.append({
                            'view_id': view_id_counter, 
                            'pos': cam_pos, 
                            'quat': target_quat
                        })
                        
                        view_id_counter += 1
                        
                views_generated = True
                print(f"Generated 24 fixed views at relative radius: {orbit_radius:.3f}m")

            if len(fixed_views) == 0:
                print(
                    "\n===================================================="
                    "\n VOXEL SIZE: ", VOXEL_SIZE,
                    "\n OBJECT TYPE: ", OBJECT_TYPE,
                    "\n===================================================="
                )
                
                total_refinement_time = np.sum(refinement_time_history)
                average_time_per_view = np.mean(refinement_time_history)
                total_views_processed = len(refinement_time_history)
                
                print(f"Total Refinement Time: {total_refinement_time:.4f} seconds")
                print(f"Average Time per View: {average_time_per_view:.4f} seconds")
                print(f"Total Views Processed: {total_views_processed}")
                print('FINAL VOLUME ESTIMATE', volume_history[-1])
                print('FINAL ERROR', ((volume_history[-1] - true_volume) / true_volume * 100))
                print(f"Absolute voxel size: {recon.voxel_size:.4f} meters")

                results_folder = "voxel_resolution_results"
                os.makedirs(results_folder, exist_ok=True)

                plot_nbv_evaluation_history(
                    unknown_voxel_history=unknown_voxel_history,
                    position_history=position_history,
                    volume_history=volume_history,
                    true_volume=true_volume,
                    view_numbers=view_history,
                    output_file_path=os.path.join(results_folder, f'view_evaluation_history_{VOXEL_SIZE}_{OBJECT_TYPE}.png'),
                )
                
                timing_file = os.path.join(results_folder, f'refinement_times_{VOXEL_SIZE}_{OBJECT_TYPE}.npy')
                np.save(timing_file, np.array(refinement_time_history))
                print(f"Saved refinement timing history to {timing_file}")
                
                print('UNKNOWN_COUNT HISTORY', unknown_voxel_history)
                print('REFINEMENT_TIME HISTORY (s)', [f"{t:.4f}" for t in refinement_time_history])

                break 
                            
            recon.surface_centers()
            if PLOT_VOXEL_CENTERS:
                plot_voxel_centers_3d(
                    occupied_centers=recon.occupied_surface_centers, 
                    unknown_centers = recon.unknown_surface_centers, 
                    bbox_corners = recon.bbox_corners,
                    title=(
                        "Occupied and Unknown Voxels - "
                        f"View {detected_view_count}"
                    ),
                )
            best_view = fixed_views.pop(0)
            
            pending_camera_position = np.asarray(best_view['pos'], dtype=float)
            waiting_for_camera_move = True
            
            send_next_view(nbv_socket, best_view)
            detected_view_count += 1

            print(
                "\nSent fixed target to Unity."
                f"\nWaiting for drone to move to: "
                f"{pending_camera_position}"
                "\nReconstruction updates are paused until arrival"
            )
            
            if PLOT_INTERMEDIATE_RESULTS:
                plot_nbv_evaluation_history(
                    unknown_voxel_history=unknown_voxel_history,
                    position_history=position_history,
                    volume_history=volume_history,
                    true_volume=true_volume,
                    view_numbers=view_history,
                )

    finally:
        nbv_socket.close()
        depth_conn.close()
        depth_server.close()

if __name__ == "__main__":
    main()