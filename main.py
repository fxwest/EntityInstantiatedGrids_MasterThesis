# -------------------------------
# ----------- Import ------------
# -------------------------------
import PointCloud as pc
import PointCloudVisu as pcv
import PointCloudProcessing as pcp
import ClusterTracking as ct


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NUM_MAX_FRAMES = 10
PCD_FOLDER = r"C:\Users\Q554273\OneDrive - BMW Group\Selbststudium\_Master\LokalisierungBewegungsplanungFusion\Fallstudie\Data\2011_09_26_drive_0052_extract\2011_09_26\2011_09_26_drive_0052_extract\velodyne_points\data"
TRIM_X_AXIS = [1.5, 15]
TRIM_Y_AXIS = [-1.5, 5.6]
TRIM_Z_AXIS = [-2.5, 3]                                                                                                 # Height of LiDAR is 1.73m


# -------------------------------
# -------- Main Function --------
# -------------------------------
def main():
    raw_pc_trace = pc.PointCloudTrace(PCD_FOLDER, NUM_MAX_FRAMES)
    trimmed_pc_frame_list = pcp.trim_fov(raw_pc_trace.raw_pc_frame_list, TRIM_X_AXIS, TRIM_Y_AXIS, TRIM_Z_AXIS)
    ground_pc_frame_list, outlier_pc_frame_list = pcp.get_ground_plane_ransac(trimmed_pc_frame_list, distance_threshold=0.3)
    clustered_pc_frame_list, dbscan_labels_frame_list, clusters_frame_list, used_tracker_ids = pcp.get_entity_cluster(outlier_pc_frame_list, eps=0.9, min_points=15)
    clusters_frame_list = ct.get_cluster_tracks(clusters_frame_list, used_tracker_ids)
    grids_frame_list = pcp.get_entity_grids(clusters_frame_list, trimmed_pc_frame_list)

    # TODO -> DONE: Get coordinate origin based on edge or centroid of clusters
    # TODO -> DONE: Get grid for each cluster with the determined coordinate origin
    # TODO -> DONE: Grid size is cluster size + offset
    # TODO -> DONE: Origin for each cluster shall be on the ground/lowest point of the cluster (remove RANSAC to get also ground considered? lowers cell is always ground?)
    # TODO -> DONE: Fill grid cell, if cell is filled (return Voxel)
    # TODO: Add occluded status to Voxel (if occupied cells are blocking the cells behind)
    # TODO -> DONE: First everything is only frame based but in the second step there needs to be a history/tracking for each grid/cluster with a KALMAN Filter e.g.

    pc_frames = [ground_pc_frame_list, clustered_pc_frame_list]
    centroid_cross_frame_list = [[entity_grid.centroid_coord_cross for entity_grid in grid_frame if entity_grid] for grid_frame in grids_frame_list]

    flag_show_empty_cells = False
    voxel_cell_visu_frames_list = []
    for idx, grid_frame in enumerate(grids_frame_list):
        voxel_cell_visu_list = []
        for entity_grid in grid_frame:
            if entity_grid is None:
                continue
            for voxel_cell_x in entity_grid.voxel_grid.grid_array:
                for voxel_cell_y in voxel_cell_x:
                    for voxel_cell in voxel_cell_y:
                        if not voxel_cell.cell_status == pcp.CellStatus.FREE:
                            voxel_cell_visu_list.insert(0, voxel_cell.visu_cell)                                        # An Anfang der Liste, damit belegte Zellen die anderen übermalen
                        else:
                            if flag_show_empty_cells:
                                voxel_cell_visu_list.append(voxel_cell.visu_cell)
        print(f"Frame {idx + 1} has {len(voxel_cell_visu_list)} Voxels")
        voxel_cell_visu_frames_list.append(voxel_cell_visu_list)


    bounding_boxes_frames = [[cluster.bounding_box for cluster in cluster_frame if cluster.bounding_box] for cluster_frame in clusters_frame_list]
    pcv.LidarViewer(pc_frames, raw_pc_trace.num_frames, raw_pc_trace.num_max_frames, centroid_frames=centroid_cross_frame_list, grid_frames=voxel_cell_visu_frames_list) #, bb_frames=bounding_boxes_frames)


# -------------------------------
# ----- Call Main Function ------
# -------------------------------
if __name__ == '__main__':
    main()
