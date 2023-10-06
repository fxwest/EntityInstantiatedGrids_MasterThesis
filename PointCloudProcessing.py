# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
GROUND_COLOR = [0.8, 0.8, 0.8]
OUTLIER_COLOR = [1.0, 0, 0]
NOISE_COLOR = [0.5, 0.5, 0.5]
BOUNDING_BOX_COLOR = [0, 1, 0]


# -------------------------------
# ---------- TRIM FOV -----------
# -------------------------------
def trim_fov(pc_frames, x_axis, y_axis, z_axis):
    print(f"Starting FOV Trimming")
    trimmed_pc_frames = []
    for frame in pc_frames:
        trimmed_pc = []
        pc_array = np.asarray(frame.pcdXYZ.points)
        for point in pc_array:
            if point[0] < x_axis[1] and point[0] > x_axis[0]:             # Filter X
                if point[1] < y_axis[1] and point[1] > y_axis[0]:         # Filter Y
                    if point[2] < z_axis[1] and point[2] > z_axis[0]:     # Filter Z
                        trimmed_pc.append(point)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(trimmed_pc)
        trimmed_pc_frames.append(pc)
    return trimmed_pc_frames


# -------------------------------
# ----------- RANSAC ------------
# -------------------------------
def get_ground_plane_ransac(pc_frames, distance_threshold=0.01, ransac_n=5, num_iterations=100):
    ground_pc_frame_list = []
    outlier_pc_frame_list = []
    print(f"Starting Ground Segmentation with RANSAC")
    for frame in pc_frames:
        # --- Run RANSAC
        plane_model, inliers = frame.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        ground_pc = frame.select_by_index(inliers)
        outlier_pc = frame.select_by_index(inliers, invert=True)

        # --- Segment Color
        ground_pc.paint_uniform_color(GROUND_COLOR)
        outlier_pc.paint_uniform_color(OUTLIER_COLOR)

        ground_pc_frame_list.append(ground_pc)
        outlier_pc_frame_list.append(outlier_pc)

    return ground_pc_frame_list, outlier_pc_frame_list


# -------------------------------
# ---------- CLUSTERS -----------
# -------------------------------
def get_entity_cluster(pc_frames, eps=0.1, min_points=10):
    print(f"Starting Clustering with DBSCAN")
    labels_frame_list = []
    clusters_frame_list = []
    clustered_pc_frames = []
    for idx, frame in enumerate(pc_frames):
        # --- Determine clusters with DBSCAN
        pc_array = np.asarray(frame.points)
        dbscan = DBSCAN(eps=eps, min_samples=min_points)
        dbscan_labels = dbscan.fit_predict(pc_array)

        # --- Determine number of clusters
        max_label = dbscan_labels.max()
        print(f"Frame {idx +1} has {max_label + 1} Clusters")

        # --- Get clusters
        cluster_list = []
        for label in np.unique(dbscan_labels):
            cluster_indices = np.where(dbscan_labels == label)[0]
            cluster_points = pc_array[cluster_indices]
            entity_cluster = EntityCluster(point_array=cluster_points, dbscan_label=label)
            cluster_list.append(entity_cluster)

        # -- Merge Clusters to one Point Cloud
        merged_cluster_point_array = np.concatenate([cluster.point_array for cluster in cluster_list], axis=0)
        merged_cluster_color_map = np.concatenate([cluster.color_map for cluster in cluster_list], axis=0)
        merged_cluster_point_cloud = o3d.geometry.PointCloud()
        merged_cluster_point_cloud.points = o3d.utility.Vector3dVector(merged_cluster_point_array)
        merged_cluster_point_cloud.colors = o3d.utility.Vector3dVector(merged_cluster_color_map)

        labels_frame_list.append(dbscan_labels)
        clusters_frame_list.append(cluster_list)
        clustered_pc_frames.append(merged_cluster_point_cloud)

    return clustered_pc_frames, labels_frame_list, clusters_frame_list


class EntityCluster:
    noise_color = NOISE_COLOR
    bounding_box_color = BOUNDING_BOX_COLOR

    def __init__(self, point_array, dbscan_label, bb_max_size=None, bb_point_limit=None):
        self.dbscan_label = dbscan_label
        self.point_array = point_array
        self.number_points = len(point_array)
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(point_array)
        self.color_map = self.get_color_map()
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.color_map)
        self.min_coords, self.max_coords = self.get_min_max_coords()
        self.bounding_box = self.get_bounding_box(bb_max_size, bb_point_limit)

    def get_color_map(self):
        if self.dbscan_label == -1:
            color = self.noise_color
        else:
            color = [round(np.random.choice(range(256))/255, 2) for rgb_value in range(3)]
        color_map = [color for point in self.point_array]
        return color_map

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


# -------------------------------
# -------- Entity Grids ---------
# -------------------------------
def get_entity_grids(clusters_frame_list):
    print("Starting Entity Grid determination...")
    grids_frame_list = []
    for cluster_frame in clusters_frame_list:
        grid_list = []
        for cluster in cluster_frame:
            if cluster.dbscan_label != -1:
                entity_grid = EntityGrid(cluster)
                grid_list.append(entity_grid)
            else:
                grid_list.append(None)
        grids_frame_list.append(grid_list)

    return grids_frame_list


class EntityGrid:
    coord_cross_size = 1
    grid_color = [1.00, 0.41, 0.71]

    def __init__(self, entity_cluster, grid_cell_size=0.4, grid_offset=0.2):
        self.entity_cluster = entity_cluster
        self.cluster_centroid, self.centroid_coord_cross = self.get_centroid()
        self.grid_line_set = self.get_entity_grid(grid_cell_size, grid_offset)                                          # TODO: Also return voxel, not only line set for visu

    def get_centroid(self):
        centroid = np.mean(self.entity_cluster.point_array, axis=0)
        centroid_coord_cross = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.coord_cross_size, origin=centroid)
        return centroid, centroid_coord_cross

    def get_entity_grid(self, grid_cell_size, grid_offset):
        x_min, x_max = self.entity_cluster.min_coords[0] - grid_offset, self.entity_cluster.max_coords[0] + grid_offset
        y_min, y_max = self.entity_cluster.min_coords[1] - grid_offset, self.entity_cluster.max_coords[1] + grid_offset
        z_min, z_max = self.entity_cluster.min_coords[2] - grid_offset, self.entity_cluster.max_coords[2] + grid_offset

        # --- Create grid points in 3D grid
        grid_points = []
        for x in np.arange(x_min, x_max + grid_cell_size, grid_cell_size):
            for y in np.arange(y_min, y_max + grid_cell_size, grid_cell_size):
                for z in np.arange(z_min, z_max + grid_cell_size, grid_cell_size):
                    grid_points.append([x, y, z])

        # --- Create lines between grid points to create a mesh
        lines = []
        for i in range(len(grid_points)):
            for j in range(i + 1, len(grid_points)):
                if np.linalg.norm(np.array(grid_points[i]) - np.array(grid_points[j])) <= (grid_cell_size + 0.005):
                    lines.append([i, j])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(grid_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(self.grid_color)
        return line_set
