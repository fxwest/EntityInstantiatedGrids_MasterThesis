# -------------------------------
# ----------- Import ------------
# -------------------------------
import PointCloud as pc
import PointCloudVisu as pcv
import PointCloudProcessing as pcp


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NUM_MAX_FRAMES = 5
PCD_FOLDER = r"C:\Users\Q554273\OneDrive - BMW Group\Selbststudium\_Master\LokalisierungBewegungsplanungFusion\Fallstudie\Data\2011_09_26_drive_0052_extract\2011_09_26\2011_09_26_drive_0052_extract\velodyne_points\data"
TRIM_X_AXIS = [1.5, 15]
TRIM_Y_AXIS = [-1.5, 1.6]
TRIM_Z_AXIS = [-2.5, 3]                                                                                                 # Height of LiDAR is 1.73m


# -------------------------------
# -------- Main Function --------
# -------------------------------
def main():
    raw_pc_trace = pc.PointCloudTrace(PCD_FOLDER, NUM_MAX_FRAMES)
    trimmed_pc_frame_list = pcp.trim_fov(raw_pc_trace.raw_pc_frame_list, TRIM_X_AXIS, TRIM_Y_AXIS, TRIM_Z_AXIS)
    ground_pc_frame_list, outlier_pc_frame_list = pcp.get_ground_plane_ransac(trimmed_pc_frame_list, distance_threshold=0.3)
    clustered_pc_frame_list, dbscan_labels_frame_list, clusters_frame_list = pcp.get_entity_cluster(outlier_pc_frame_list, eps=0.9, min_points=15)
    grids_frame_list = pcp.get_entity_grids(clusters_frame_list)

    # TODO -> DONE: Get coordinate origin based on edge or centroid of clusters
    # TODO -> DONE: Get grid for each cluster with the determined coordinate origin
    # TODO -> DONE: Grid size is cluster size + offset
    # TODO: Origin for each cluster shall be on the ground/lowest point of the cluster (remove RANSAC to get also ground considered? lowers cell is always ground?)
    # TODO: Fill grid cell, if cell is filled (return Voxel)
    # TODO: First everything is only frame based but in the second step there needs to be a history/tracking for each grid/cluster with a KALMAN Filter e.g.

    pc_frames = [ground_pc_frame_list, clustered_pc_frame_list]
    centroid_cross_frame_list = [[entity_grid.centroid_coord_cross for entity_grid in grid_frame if entity_grid] for grid_frame in grids_frame_list]
    grids_line_set_frame_list = [[entity_grid.grid_line_set for entity_grid in grid_frame if entity_grid] for grid_frame in grids_frame_list]
    bounding_boxes_frames = [[cluster.bounding_box for cluster in cluster_frame if cluster.bounding_box] for cluster_frame in clusters_frame_list]
    pcv.LidarViewer(pc_frames, raw_pc_trace.num_frames, raw_pc_trace.num_max_frames, centroid_frames=centroid_cross_frame_list, grid_frames=grids_line_set_frame_list, bb_frames=bounding_boxes_frames)


# -------------------------------
# ----- Call Main Function ------
# -------------------------------
if __name__ == '__main__':
    main()
