# -------------------------------
# ----------- Import ------------
# -------------------------------
import PointCloud as pc
import PointCloudVisu as pcv
import PointCloudProcessing as pcp

from SegmentedPointCloud import segment_ground_plane
from ClusteredPointCloud import panoptic_segmentation
from ClusterTracking import get_cluster_tracks


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NUM_MAX_FRAMES = 5
PCD_FOLDER = r"C:\Users\Q554273\OneDrive - BMW Group\Selbststudium\_Master\LokalisierungBewegungsplanungFusion\Fallstudie\Data\2011_09_26_drive_0052_extract\2011_09_26\2011_09_26_drive_0052_extract\velodyne_points\data"
TRIM_X_AXIS = [1.5, 15]
TRIM_Y_AXIS = [-1.5, 5.6]
TRIM_Z_AXIS = [-2.5, 3]                                                                                                 # Height of LiDAR is 1.73m


# -------------------------------
# -------- Main Function --------
# -------------------------------
def main():
    pc_trace = pc.PointCloudTrace(PCD_FOLDER, NUM_MAX_FRAMES)
    pc_trace.trim_pc_fov(TRIM_X_AXIS, TRIM_Y_AXIS, TRIM_Z_AXIS)
    segmented_pc_trace = segment_ground_plane(pc_trace, distance_threshold=0.15, ransac_n=5, num_iterations=100)         # distance_threshold is a trade-off between Ground FP and Small Obstacle FN
    clustered_pc_trace = panoptic_segmentation(segmented_pc_trace, eps=0.9, min_points=100)
    tracked_pc_trace = get_cluster_tracks(clustered_pc_trace, max_dist=0.5, plot_kalman_results=False)

    # TODO: Adapt to new structure START
    clusters_frame_list = []
    pc_frame_list = []
    for frame in tracked_pc_trace.pc_frame_list:
        clusters_frame_list.append(frame.entity_cluster_list)
        pc_frame_list.append(frame.pcdXYZ)
    # TODO: Adapt to new structure END

    grids_frame_list = pcp.get_entity_grids(clusters_frame_list, pc_frame_list)
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


    #bounding_boxes_frames = [[cluster.bounding_box for cluster in cluster_frame if cluster.bounding_box] for cluster_frame in clusters_frame_list]
    pcv.LidarViewer(tracked_pc_trace, centroid_frames=centroid_cross_frame_list, grid_frames=voxel_cell_visu_frames_list) #, bb_frames=bounding_boxes_frames)


# -------------------------------
# ----- Call Main Function ------
# -------------------------------
if __name__ == '__main__':
    main()
