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
    clustered_pc_frame_list, dbscan_labels_frame_list = pcp.get_clusters_dbscan(outlier_pc_frame_list, eps=0.9, min_points=15)

    # TODO: Get coordinate origin based on edge or centroid of clusters
    # TODO: Get grid for each cluster with the determined coordinate origin
    # TODO: First everything is only frame based but in the second step there needs to be a history/tracking for each grid
    #centroid_list = pcp.get_centroids(clustered_pc_frame_list, dbscan_labels)

    #bounding_boxes_frame_list = pcp.get_bounding_boxes(clustered_pc_frame_list, dbscan_labels_frame_list, min_points=50, max_points=20000, max_x_size=20, max_y_size=5, max_z_size=5)
    pc_frames = [ground_pc_frame_list, clustered_pc_frame_list]
    pcv.LidarViewer(pc_frames, raw_pc_trace.num_frames, raw_pc_trace.num_max_frames) #, bounding_boxes_frame_list)


# -------------------------------
# ----- Call Main Function ------
# -------------------------------
if __name__ == '__main__':
    main()
