# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d
from PointCloud import PointCloudFrame


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
GROUND_COLOR = [0.8, 0.8, 0.8]
OUTLIER_COLOR = [1.0, 0, 0]


# -------------------------------
# ------ Get Ground Plane -------
# -------------------------------
def segment_ground_plane(pc_trace, distance_threshold=0.1, ransac_n=5, num_iterations=100):
    print(f"Starting Ground Segmentation with RANSAC")
    for frame_idx, pc_frame in enumerate(pc_trace.pc_frame_list):
        pc_trace.pc_frame_list[frame_idx] = SegmentedPointCloudFrame(pc_frame, distance_threshold, ransac_n, num_iterations)

    return pc_trace


# -------------------------------
# ---- Segmented Point Cloud ----
# -------------------------------
class SegmentedPointCloudFrame(PointCloudFrame):
    """
    Segmented Point Cloud Frame.
    """
    def __init__(self, pc_frame, distance_threshold, ransac_n, num_iterations):
        # --- Complete Point Cloud
        self.frame_idx = pc_frame.frame_idx
        self.pcdXYZ = pc_frame.pcdXYZ
        self.octree = pc_frame.octree
        self.num_points = pc_frame.num_points
        self.refl = pc_frame.refl

        # --- Run RANSAC
        self.pcdXYZ_ground, self.pcdXYZ_outlier, inliers, self.plane_model = self.get_ground_plane_ransac(distance_threshold, ransac_n, num_iterations)

        # --- Ground Point Cloud
        self.octree_ground = o3d.geometry.Octree(max_depth=self.octree_max_depth)
        self.octree_ground.convert_from_point_cloud(self.pcdXYZ_ground)
        self.point_array_ground = np.asarray(self.pcdXYZ_ground.points)
        self.num_points_ground = len(self.point_array_ground)
        self.refl_ground = self.refl[inliers]

        # --- Outlier Point Cloud
        self.octree_outlier = o3d.geometry.Octree(max_depth=self.octree_max_depth)
        self.octree_outlier.convert_from_point_cloud(self.pcdXYZ_outlier)
        self.point_array_outlier = np.asarray(self.pcdXYZ_outlier.points)
        self.num_points_outlier = len(self.point_array_outlier)
        inverse_mask = np.ones(len(self.refl), np.bool)
        inverse_mask[inliers] = 0
        self.refl_outlier = self.refl[inverse_mask]

        # --- Update complete point cloud with new colors
        ground_color_array = np.asarray(self.pcdXYZ_ground.colors)
        outlier_color_array = np.asarray(self.pcdXYZ_outlier.colors)
        merged_point_clouds = np.concatenate([self.point_array_ground, self.point_array_outlier], axis=0)               # Required to have correct indices of points
        merged_color_map = np.concatenate([ground_color_array, outlier_color_array], axis=0)
        self.pcdXYZ.points = o3d.utility.Vector3dVector(merged_point_clouds)
        self.pcdXYZ.colors = o3d.utility.Vector3dVector(merged_color_map)
        self.octree.convert_from_point_cloud(self.pcdXYZ)
        self.point_array = np.asarray(self.pcdXYZ.points)
        self.refl = np.concatenate([self.refl_ground, self.refl_outlier], axis=0)

    def get_ground_plane_ransac(self, distance_threshold, ransac_n, num_iterations):
        plane_model, inliers = self.pcdXYZ.segment_plane(distance_threshold=distance_threshold,
                                                         ransac_n=ransac_n, num_iterations=num_iterations)
        pcdXYZ_ground = self.pcdXYZ.select_by_index(inliers)
        pcdXYZ_outlier = self.pcdXYZ.select_by_index(inliers, invert=True)
        pcdXYZ_ground.paint_uniform_color(GROUND_COLOR)                                                                 # Segment Color
        pcdXYZ_outlier.paint_uniform_color(OUTLIER_COLOR)

        return pcdXYZ_ground, pcdXYZ_outlier, inliers, plane_model

    def get_ground_height(self, x):
        z = -(self.plane_model[0] * x + self.plane_model[3]) / self.plane_model[2]
        return z
