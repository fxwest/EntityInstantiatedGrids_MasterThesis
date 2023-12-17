# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d
from enum import Enum
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
def get_entity_cluster(pc_frames, eps=0.1, min_points=10, max_dist=0.5):
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
            if entity_cluster.dbscan_label != -1:                                                                       # Don't add noise clusters
                if idx == 0:
                    entity_cluster.tracker_id = entity_cluster.dbscan_label                                             # Add DBSCAN-Label as Tracker ID in first frame
                else:                                                                                                   # TODO: Use KALMAN-Filter for Tracking, separate in other Function
                    prev_cluster_list = clusters_frame_list[idx-1]
                    for prev_cluster in prev_cluster_list:
                        dist = entity_cluster.euclidean_distance(prev_cluster.centroid)
                        if dist <= max_dist:
                            entity_cluster.tracker_id = prev_cluster.tracker_id
                            entity_cluster.copy_color(prev_cluster)
                            continue
                    if not entity_cluster.tracker_id:
                        entity_cluster.tracker_id = entity_cluster.dbscan_label                                         # If no matching prev cluster was found -> new track # TODO: Must be unique over all frames
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

    def __repr__(self):
        return f"EntityCluster({str(self.min_coords), str(self.max_coords)}, {self.number_points}, {self.tracker_id})"

    def __str__(self):
        return f"Entity Cluster with Tracker ID {str(self.tracker_id)} and {self.number_points} Points."

    def __init__(self, point_array, dbscan_label, bb_max_size=None, bb_point_limit=None):
        self.tracker_id = None
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

    def euclidean_distance(self, second_centroid):
        return np.sqrt((second_centroid[0] - self.centroid[0])**2 + (second_centroid[1] - self.centroid[1])**2 + (second_centroid[2] - self.centroid[2])**2)

    def copy_color(self, second_centroid):
        self.color, self.color_map = self.get_color_map(second_centroid.color)
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.color_map)


# -------------------------------
# -------- Entity Grids ---------
# -------------------------------
def get_entity_grids(clusters_frame_list, trimmed_pc_frame_list):
    print("Starting Entity Grid determination...")
    grids_frame_list = []
    for frame_idx, cluster_frame in enumerate(clusters_frame_list):
        grid_list = []
        for cluster in cluster_frame:
            if cluster.dbscan_label != -1:
                point_cloud_all_points = np.asarray(trimmed_pc_frame_list[frame_idx].points)
                entity_grid = EntityGrid(cluster, point_cloud_all_points)
                grid_list.append(entity_grid)
            else:
                grid_list.append(None)
        grids_frame_list.append(grid_list)

    return grids_frame_list


class EntityGrid:
    coord_cross_size = 1
    grid_color = [1.00, 0.41, 0.71]

    def __init__(self, entity_cluster, point_cloud_all_points, grid_offset=0.4):
        self.entity_cluster = entity_cluster
        self.centroid_coord_cross = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.coord_cross_size, origin=entity_cluster.centroid)
        self.voxel_grid = self.get_entity_grid(grid_offset, point_cloud_all_points)

    def get_entity_grid(self, grid_offset, point_cloud_all_points):
        x_min, x_max = self.entity_cluster.min_coords[0] - grid_offset, self.entity_cluster.max_coords[0] + grid_offset
        y_min, y_max = self.entity_cluster.min_coords[1] - grid_offset, self.entity_cluster.max_coords[1] + grid_offset
        z_min, z_max = self.entity_cluster.min_coords[2] - grid_offset, self.entity_cluster.max_coords[2] + grid_offset

        voxel_grid = VoxelGrid(start_pos_skosy=(x_min, y_min, z_min), end_pos_skosy=(x_max, y_max, z_max),
                               point_cloud_all_points=point_cloud_all_points, point_color=self.entity_cluster.color)
        return voxel_grid


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


class VoxelGrid:
    def __init__(self, start_pos_skosy, end_pos_skosy, point_cloud_all_points, point_color, cell_size=0.4):
        self.start_pos_skosy = start_pos_skosy                                                                          # Front Down Left (x, y, z)
        self.end_pos_skosy = end_pos_skosy                                                                              # Rear Up Right (x, y, z)
        self.x_cells_pos = np.arange(self.start_pos_skosy[0], self.end_pos_skosy[0] + cell_size, cell_size)             # Including last point
        self.y_cells_pos = np.arange(self.start_pos_skosy[1], self.end_pos_skosy[1] + cell_size, cell_size)
        self.z_cells_pos = np.arange(self.start_pos_skosy[2], self.end_pos_skosy[2] + cell_size, cell_size)
        self.grid_dim = (len(self.x_cells_pos)-1, len(self.y_cells_pos)-1, len(self.z_cells_pos)-1)                     # Grid dimensions (number of cells per axis)
        self.grid_array = np.empty(shape=self.grid_dim, dtype=object)

        for x_idx, x_axis_row in enumerate(self.grid_array):
            for y_idx, y_axis_row in enumerate(x_axis_row):
                for z_idx, voxel_cell in enumerate(y_axis_row):
                    voxel_pos = (x_idx, y_idx, z_idx)
                    voxel_start_pos_skosy = (self.x_cells_pos[x_idx], self.y_cells_pos[y_idx], self.z_cells_pos[z_idx])
                    voxel_end_pos_skosy = (self.x_cells_pos[x_idx+1], self.y_cells_pos[y_idx+1], self.z_cells_pos[z_idx+1])
                    voxel_point_indices = np.where((point_cloud_all_points[:, 0] > voxel_start_pos_skosy[0])
                                                   & (point_cloud_all_points[:, 0] < voxel_end_pos_skosy[0])
                                                   & (point_cloud_all_points[:, 1] > voxel_start_pos_skosy[1])
                                                   & (point_cloud_all_points[:, 1] < voxel_end_pos_skosy[1])
                                                   & (point_cloud_all_points[:, 2] > voxel_start_pos_skosy[2])
                                                   & (point_cloud_all_points[:, 2] < voxel_end_pos_skosy[2]))
                    voxel_point_array = point_cloud_all_points[voxel_point_indices]
                    voxel_cell = VoxelCell(voxel_pos=voxel_pos, start_pos_skosy=voxel_start_pos_skosy,
                                           end_pos_skosy=voxel_end_pos_skosy, point_array=voxel_point_array,
                                           point_color=point_color)
                    self.grid_array[x_idx, y_idx, z_idx] = voxel_cell


class CellStatus(Enum):
    FREE = 0
    OCCUPIED = 1
    OCCLUDED = 2
    NOISE = 3


class VoxelCell:
    visu_border_color_filled = [1.00, 0.41, 0.71]
    visu_border_color_empty = [0.20, 0.58, 1.00]
    visu_border_color_noise = NOISE_COLOR

    def __repr__(self):
        return f"VoxelCell({str(self.voxel_pos)}, {self.num_points})"

    def __str__(self):
        return f"Voxel Cell at grid position {str(self.voxel_pos)} with {self.num_points} Points."

    def __init__(self, voxel_pos, start_pos_skosy, end_pos_skosy, point_array, point_color):
        self.voxel_pos = voxel_pos
        self.start_pos_skosy = start_pos_skosy                                                                          # Front Down Left (x, y, z)
        self.end_pos_skosy = end_pos_skosy                                                                              # Rear Up Right (x, y, z)
        self.point_array = point_array
        self.num_points = len(point_array)
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(point_array)
        self.point_cloud.colors = o3d.utility.Vector3dVector([point_color for point in self.point_array])
        #self.rel_pos_coord

        self.visu_cell = o3d.geometry.AxisAlignedBoundingBox(self.start_pos_skosy, self.end_pos_skosy)
        self.visu_cell.color = self.visu_border_color_empty
        self.cell_status = CellStatus.FREE
        if self.num_points > 0:                                                                                         # TODO: Status der Zelle (belegt, Noise, ML, etc.)
            self.voxel_centroid = np.mean(self.point_array, axis=0)
            if self.is_geometrically_ordered(self.point_array, self.voxel_centroid) and self.num_points > 1:
                if self.has_vertical_extend(self.point_array):
                    self.visu_cell.color = self.visu_border_color_filled
                    self.cell_status = CellStatus.OCCUPIED
            else:
                self.visu_cell.color = self.visu_border_color_noise
                self.cell_status = CellStatus.NOISE

    @staticmethod
    def is_geometrically_ordered(points, voxel_center, variance_threshold=0.01):
        distances = np.linalg.norm(points - voxel_center, axis=1)
        variance = np.var(distances)
        if variance <= variance_threshold:
            return True
        else:
            return False

    @staticmethod
    def has_vertical_extend(points, variance_threshold=0.001):
        variance = np.var(points[:, 2])
        if variance <= variance_threshold:
            return False
        else:
            return True
