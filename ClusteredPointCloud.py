# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d
from enum import Enum
from sklearn.cluster import DBSCAN
from PointCloud import PointCloudFrame


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NOISE_COLOR = [0.5, 0.5, 0.5]
BOUNDING_BOX_COLOR = [0, 1, 0]


# -------------------------------
# ------------ Enums ------------
# -------------------------------
class AnchorType(Enum):
    CENTROID = 0
    CENTERED_FLOOR = 1
    LEFT_EDGE = 2


# -------------------------------
# ------ Get Ground Plane -------
# -------------------------------
def panoptic_segmentation(pc_trace, eps=0.9, min_points=15, anchor_type=AnchorType.CENTROID):
    print(f"Starting Clustering with DBSCAN")
    for frame_idx, pc_frame in enumerate(pc_trace.pc_frame_list):
        pc_trace.pc_frame_list[frame_idx] = PanopticPointCloudFrame(pc_frame, eps, min_points, anchor_type)

    return pc_trace


# -------------------------------
# ---- Clustered Point Cloud ----
# -------------------------------
class PanopticPointCloudFrame(PointCloudFrame):
    """
    Clustered and Segmented Point Cloud, i.e. Panoptic Segmented Point Cloud Frame.
    """
    def __init__(self, segmented_pc_frame, eps, min_points, anchor_type):
        # --- Complete Point Cloud
        self.frame_idx = segmented_pc_frame.frame_idx
        self.pcdXYZ = segmented_pc_frame.pcdXYZ
        self.octree = segmented_pc_frame.octree
        self.point_array = segmented_pc_frame.point_array
        self.num_points = segmented_pc_frame.num_points
        self.refl = segmented_pc_frame.refl

        # --- Ground Point Cloud
        self.pcdXYZ_ground = segmented_pc_frame.pcdXYZ_ground
        self.octree_ground = segmented_pc_frame.octree_ground
        self.point_array_ground = segmented_pc_frame.point_array_ground
        self.num_points_ground = segmented_pc_frame.num_points_ground
        self.refl_ground = segmented_pc_frame.refl_ground

        # --- Run DBSCAN
        self.entity_cluster_list, self.num_clusters, noise_indices = self.get_entity_clusters(segmented_pc_frame, eps, min_points, anchor_type)
        print(f"Frame {self.frame_idx} has {self.num_clusters} Clusters")

        # --- Noise Points
        if noise_indices is None:
            self.point_array_noise = None
            self.pcdXYZ_noise = None
            self.octree_noise = None
            self.num_points = 0
            self.refl_noise = None
        else:
            self.point_array_noise = self.point_array[noise_indices]
            self.pcdXYZ_noise = self.pcdXYZ.select_by_index(noise_indices)
            self.octree_noise = o3d.geometry.Octree(max_depth=self.octree_max_depth)
            self.octree_noise.convert_from_point_cloud(self.pcdXYZ_noise)
            self.num_points_noise = len(self.point_array_noise)
            self.refl_noise = self.refl[noise_indices]
            self.pcdXYZ_noise.paint_uniform_color(NOISE_COLOR)

        # -- Merge Clusters Noise and Ground to one Point Cloud
        self.merge_point_clouds()

    def get_entity_clusters(self, segmented_pc_frame, eps, min_points, anchor_type):
        dbscan = DBSCAN(eps=eps, min_samples=min_points)
        dbscan_labels = dbscan.fit_predict(segmented_pc_frame.point_array_outlier)
        num_clusters = dbscan_labels.max() + 1

        # --- Get Entity Clusters for each DBSCAN Label
        cluster_list = []
        noise_indices = None
        for label in np.unique(dbscan_labels):
            if label != -1:                                                                                             # Don't add noise clusters
                cluster_indices = np.where(dbscan_labels == label)[0]
                cluster_points = segmented_pc_frame.point_array_outlier[cluster_indices]
                entity_cluster = EntityCluster(point_array=cluster_points, dbscan_label=label, anchor_type=anchor_type,
                                               segmented_pc_frame=segmented_pc_frame)                                   # TODO: Also pass refl values?
                entity_cluster.tracker_id = entity_cluster.dbscan_label                                                 # Add DBSCAN-Label as Tracker ID
                cluster_list.append(entity_cluster)
            else:
                noise_indices = np.where(dbscan_labels == label)[0]
        return cluster_list, num_clusters, noise_indices

    def merge_point_clouds(self):
        merged_list = [cluster.point_array for cluster in self.entity_cluster_list]
        merged_list.extend([self.point_array_noise, self.point_array_ground])
        merged_cluster_point_array = np.concatenate(merged_list, axis=0)
        noise_color_array = np.asarray(self.pcdXYZ_noise.colors)
        ground_color_array = np.asarray(self.pcdXYZ_ground.colors)
        merged_list = [cluster.color_map for cluster in self.entity_cluster_list]
        merged_list.extend([noise_color_array, ground_color_array])
        merged_cluster_color_map = np.concatenate(merged_list, axis=0)
        self.pcdXYZ.points = o3d.utility.Vector3dVector(merged_cluster_point_array)
        self.pcdXYZ.colors = o3d.utility.Vector3dVector(merged_cluster_color_map)
        self.octree.convert_from_point_cloud(self.pcdXYZ)
        self.point_array = np.asarray(self.pcdXYZ.points)
        self.refl = None                                                                                                # TODO: Can be added, if refl is added to EntityCluster Class


class EntityCluster:
    noise_color = NOISE_COLOR
    bounding_box_color = BOUNDING_BOX_COLOR

    def __repr__(self):
        return f"EntityCluster({str(self.min_coords), str(self.max_coords)}, NumPoints:{self.number_points}, ID:{self.tracker_id}, Age:{self.tracker_age})"

    def __str__(self):
        return f"Entity Cluster with Tracker ID {str(self.tracker_id)} and {self.number_points} Points."

    def __init__(self, point_array, dbscan_label, anchor_type, segmented_pc_frame, bb_max_size=None, bb_point_limit=None):
        self.tracker_id = None
        self.tracker_age = 0
        self.dbscan_label = dbscan_label
        self.point_array = point_array
        self.number_points = len(point_array)
        self.centroid = self.get_centroid()
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(point_array)
        self.color, self.color_map = self.get_color_map()
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.color_map)
        self.min_coords, self.max_coords = self.get_min_max_coords()
        self.bounding_box = self.get_bounding_box(bb_max_size, bb_point_limit)
        self.anchor_point = self.get_anchor_point(anchor_type, segmented_pc_frame)

    def get_color_map(self, copy_color=None):
        if copy_color:
            color = copy_color
        else:
            if self.dbscan_label == -1:
                color = self.noise_color
            else:
                color = [round(np.random.choice(range(256))/255, 2) for rgb_value in range(3)]
        color_map = [color for point in self.point_array]
        return color, color_map

    def get_min_max_coords(self):
        if self.dbscan_label != -1:
            x_point_array = self.point_array[:, 0]
            y_point_array = self.point_array[:, 1]
            z_point_array = self.point_array[:, 2]
            x_min = np.min(x_point_array)
            x_max = np.max(x_point_array)
            y_min = np.min(y_point_array)
            y_max = np.max(y_point_array)
            z_min = np.min(z_point_array)
            z_max = np.max(z_point_array)
            min_coords = (x_min, y_min, z_min)
            max_coords = (x_max, y_max, z_max)
        else:
            min_coords = None
            max_coords = None
        return min_coords, max_coords

    def get_bounding_box(self, bb_max_size, bb_point_limit):
        if self.dbscan_label == -1:
            return None
        if bb_max_size:
            if self.max_coords[0] - self.min_coords[0] > bb_max_size[0] and \
               self.max_coords[1] - self.min_coords[1] > bb_max_size[1] and \
               self.max_coords[2] - self.min_coords[2] > bb_max_size[2]:                                            # Check if BB size logical
                return None
        if bb_point_limit:
            if self.number_points < bb_point_limit[0] or self.number_points > bb_point_limit[1]:
                return None

        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.min_coords, self.max_coords)
        bounding_box.color = self.bounding_box_color
        return bounding_box

    def get_centroid(self):
        centroid = np.mean(self.point_array, axis=0)
        return centroid

    def euclidean_distance(self, second_anchor_point):
        return np.sqrt((second_anchor_point[0] - self.anchor_point[0])**2 + (second_anchor_point[1] - self.anchor_point[1])**2 + (second_anchor_point[2] - self.anchor_point[2])**2)

    def copy_color(self, second_centroid):
        self.color, self.color_map = self.get_color_map(second_centroid.color)
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.color_map)

    def get_anchor_point(self, anchor_type, segmented_pc_frame):
        if anchor_type == AnchorType.CENTROID:
            return self.centroid
        if anchor_type == AnchorType.CENTERED_FLOOR:
            x_min = self.min_coords[0]
            z_min = segmented_pc_frame.get_ground_height(x_min)
            y_centroid = self.centroid[1]
            return np.array([x_min, y_centroid, z_min])
        if anchor_type == AnchorType.LEFT_EDGE:
            z_min = self.min_coords[2]
            y_min = self.min_coords[1]
            x_min = self.min_coords[0]
            return np.array([x_min, y_min, z_min])
