# -------------------------------
# ----------- Import ------------
# -------------------------------
import copy
import math
import numpy as np
import open3d as o3d
from enum import Enum
from sklearn.cluster import DBSCAN


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NOISE_COLOR = [0.5, 0.5, 0.5]
BOUNDING_BOX_COLOR = [0, 1, 0]
PKW_SIZE = [2.50, 4.00, 2.50]                                                                                           # width, length, height
LKW_SIZE = [3.00, 18.75, 4.00]


# -------------------------------
# ---------- CLUSTERS -----------
# -------------------------------
def get_entity_cluster(pc_frames, eps=0.1, min_points=10, max_dist=0.5):
    print(f"Starting Clustering with DBSCAN")
    labels_frame_list = []
    clusters_frame_list = []
    clustered_pc_frames = []
    used_tracker_ids = []
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
                    used_tracker_ids.append(entity_cluster.tracker_id)                                                  # Add Traker ID to list to avoid multiple usage of same IDs
                else:
                    prev_cluster_list = clusters_frame_list[idx-1]
                    for prev_cluster in prev_cluster_list:
                        dist = entity_cluster.euclidean_distance(prev_cluster.centroid)
                        if dist <= max_dist:
                            entity_cluster.tracker_id = prev_cluster.tracker_id
                            entity_cluster.copy_color(prev_cluster)
                            entity_cluster.tracker_age = prev_cluster.tracker_age + 1
                            continue
                    if not entity_cluster.tracker_id:
                        entity_cluster.tracker_id = used_tracker_ids[-1] + 1                                            # If no matching prev cluster was found -> new track
                        entity_cluster.tracker_age = 0
                        used_tracker_ids.append(entity_cluster.tracker_id)
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

    return clustered_pc_frames, labels_frame_list, clusters_frame_list, used_tracker_ids


class EntityCluster:
    noise_color = NOISE_COLOR
    bounding_box_color = BOUNDING_BOX_COLOR

    def __repr__(self):
        return f"EntityCluster({str(self.min_coords), str(self.max_coords)}, NumPoints:{self.number_points}, ID:{self.tracker_id}, Age:{self.tracker_age})"

    def __str__(self):
        return f"Entity Cluster with Tracker ID {str(self.tracker_id)} and {self.number_points} Points."

    def __init__(self, point_array, dbscan_label, bb_max_size=None, bb_point_limit=None):
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
        point_cloud_all_points = np.asarray(trimmed_pc_frame_list[frame_idx].points)
        for cluster in cluster_frame:
            if cluster.dbscan_label != -1:
                if cluster.tracker_age > 0:
                    for grid in grids_frame_list[frame_idx - 1]:
                        if grid.tracker_id == cluster.tracker_id:
                            grid.update_entity_grid(cluster, point_cloud_all_points)
                            grid_list.append(copy.deepcopy(grid))
                else:
                    entity_grid = EntityGrid(cluster, point_cloud_all_points)
                    grid_list.append(entity_grid)
        grids_frame_list.append(grid_list)

    return grids_frame_list


class VehicleType(Enum):
    UNKNOWN = 0
    PKW = 1
    LKW = 2


class EntityGrid:
    coord_cross_size = 1
    grid_color = [1.00, 0.41, 0.71]
    width_tol = 0.2
    length_tol = 1.0
    height_tol = 0.5

    def __repr__(self):
        return f"EntityGrid(ID:{str(self.tracker_id)}, Age:{str(self.tracker_age)})"

    def __str__(self):
        return f"Entity Grid with Tracker ID {str(self.tracker_id)} and Tracker Age {str(self.tracker_age)}."

    def __init__(self, entity_cluster, point_cloud_all_points, grid_offset=0.4):
        self.voxel_grid_history = []
        self.vehicle_type = VehicleType.UNKNOWN
        self.grid_offset = grid_offset
        self.tracker_id = entity_cluster.tracker_id
        self.entity_cluster = entity_cluster
        self.tracker_age = entity_cluster.tracker_age
        self.centroid_coord_cross = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.coord_cross_size, origin=entity_cluster.centroid)
        grid_borders = self.get_vehicle_model(grid_offset)
        self.voxel_grid = self.get_entity_grid(grid_borders, point_cloud_all_points, self.voxel_grid_history)

    def get_vehicle_model(self, grid_offset):
        cluster_width = self.entity_cluster.max_coords[0] - self.entity_cluster.min_coords[0]
        cluster_length = self.entity_cluster.max_coords[0] - self.entity_cluster.min_coords[0]
        cluster_height = self.entity_cluster.max_coords[0] - self.entity_cluster.min_coords[0]

        pkw_count = 0
        lkw_count = 0
        if math.isclose(cluster_width, PKW_SIZE[0], abs_tol=self.width_tol):
            pkw_count += 1
        elif math.isclose(cluster_width, LKW_SIZE[0], abs_tol=self.width_tol):
            lkw_count += 1
        if math.isclose(cluster_length, PKW_SIZE[1], abs_tol=self.length_tol):
            pkw_count += 1
        elif math.isclose(cluster_length, LKW_SIZE[1], abs_tol=self.length_tol):
            lkw_count += 1
        if math.isclose(cluster_height, PKW_SIZE[2], abs_tol=self.height_tol):
            pkw_count += 1
        elif math.isclose(cluster_height, LKW_SIZE[2], abs_tol=self.height_tol):
            lkw_count += 1

        if pkw_count > 0 or lkw_count > 0:
            if pkw_count > lkw_count:
                self.vehicle_type = VehicleType.PKW
                x_min, x_max = self.entity_cluster.min_coords[0] - PKW_SIZE[0]/2 - grid_offset, self.entity_cluster.max_coords[0] + PKW_SIZE[0]/2 + grid_offset
                y_min, y_max = self.entity_cluster.min_coords[1] - grid_offset, self.entity_cluster.min_coords[1] + PKW_SIZE[1] + grid_offset
                z_min, z_max = self.entity_cluster.min_coords[2] - grid_offset, self.entity_cluster.max_coords[2] + PKW_SIZE[2] + grid_offset
            else:
                self.vehicle_type = VehicleType.LKW
                x_min, x_max = self.entity_cluster.min_coords[0] - LKW_SIZE[0] / 2 - grid_offset, self.entity_cluster.max_coords[0] + LKW_SIZE[0] / 2 + grid_offset
                y_min, y_max = self.entity_cluster.min_coords[1] - grid_offset, self.entity_cluster.min_coords[1] + LKW_SIZE[1] + grid_offset
                z_min, z_max = self.entity_cluster.min_coords[2] - grid_offset, self.entity_cluster.max_coords[2] + LKW_SIZE[2] + grid_offset
        else:
            x_min, x_max = self.entity_cluster.min_coords[0] - grid_offset, self.entity_cluster.max_coords[0] + grid_offset
            y_min, y_max = self.entity_cluster.min_coords[1] - grid_offset, self.entity_cluster.max_coords[1] + grid_offset
            z_min, z_max = self.entity_cluster.min_coords[2] - grid_offset, self.entity_cluster.max_coords[2] + grid_offset

        print(f"Assumed Vehicle Model: {self.vehicle_type}")
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def get_entity_grid(self, grid_borders, point_cloud_all_points, voxel_grid_history):
        voxel_grid = VoxelGrid(start_pos_skosy=(grid_borders[0], grid_borders[2], grid_borders[4]), end_pos_skosy=(grid_borders[1], grid_borders[3], grid_borders[5]),
                               point_cloud_all_points=point_cloud_all_points, point_color=self.entity_cluster.color,
                               history=voxel_grid_history)
        return voxel_grid

    def update_entity_grid(self, entity_cluster, point_cloud_all_points):
        self.entity_cluster = entity_cluster
        self.tracker_age = entity_cluster.tracker_age
        self.centroid_coord_cross = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.coord_cross_size, origin=entity_cluster.centroid)

        # TODO: Grenzen nicht nur vom Cluster abhängig machen, möglichst stabil halten (relativer Abstand zum Ankerpunkt sollte konstant bleiben)
        grid_borders = self.get_vehicle_model(self.grid_offset)

        self.voxel_grid = self.get_entity_grid(grid_borders, point_cloud_all_points, self.voxel_grid_history)
        self.voxel_grid_history.append(self.voxel_grid)


class VoxelGrid:
    def __init__(self, start_pos_skosy, end_pos_skosy, point_cloud_all_points, point_color, history, cell_size=0.4):

        if len(history) > 0:
            self.grid_dim = history[0].grid_dim
            self.start_pos_skosy = start_pos_skosy                                                                      # Front Down Left (x, y, z)
            self.end_pos_skosy = end_pos_skosy                                                                          # Rear Up Right (x, y, z)
            # --- Take dim from first grid
            self.x_cells_pos = np.array([self.start_pos_skosy[0] + (cell_size * cell_count) for cell_count in range(1, self.grid_dim[0] + 2)])
            self.y_cells_pos = np.array([self.start_pos_skosy[1] + (cell_size * cell_count) for cell_count in range(1, self.grid_dim[1] + 2)])
            self.z_cells_pos = np.array([self.start_pos_skosy[2] + (cell_size * cell_count) for cell_count in range(1, self.grid_dim[2] + 2)])
        else:
            self.start_pos_skosy = start_pos_skosy                                                                      # Front Down Left (x, y, z)
            self.end_pos_skosy = end_pos_skosy                                                                          # Rear Up Right (x, y, z)
            self.x_cells_pos = np.arange(self.start_pos_skosy[0], self.end_pos_skosy[0] + cell_size, cell_size)         # Including last point
            self.y_cells_pos = np.arange(self.start_pos_skosy[1], self.end_pos_skosy[1] + cell_size, cell_size)
            self.z_cells_pos = np.arange(self.start_pos_skosy[2], self.end_pos_skosy[2] + cell_size, cell_size)
            self.grid_dim = (len(self.x_cells_pos) - 1, len(self.y_cells_pos) - 1, len(self.z_cells_pos) - 1)           # Grid dimensions (number of cells per axis) TODO: Erster Frame entscheided Grid größe
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
                    if len(history) > 0:
                        cell_history = [frame.grid_array[x_idx, y_idx, z_idx] for frame in history]
                    else:
                        cell_history = None
                    voxel_cell = VoxelCell(voxel_pos=voxel_pos, start_pos_skosy=voxel_start_pos_skosy,
                                           end_pos_skosy=voxel_end_pos_skosy, point_array=voxel_point_array,
                                           point_color=point_color,
                                           cell_history=cell_history)
                    self.grid_array[x_idx, y_idx, z_idx] = voxel_cell


class CellStatus(Enum):
    FREE = 0
    OCCUPIED = 1
    OCCLUDED = 2
    NOISE = 3
    CONFIRMED_OCCUPIED = 4


class VoxelCell:
    visu_border_color_filled = [1.00, 0.41, 0.71]
    visu_border_color_empty = [0.20, 0.58, 1.00]
    visu_border_color_confirmed = [1.00, 0.00, 0.00]
    visu_border_color_noise = NOISE_COLOR

    def __repr__(self):
        return f"VoxelCell({str(self.voxel_pos)}, {self.num_points})"

    def __str__(self):
        return f"Voxel Cell at grid position {str(self.voxel_pos)} with {self.num_points} Points."

    def __init__(self, voxel_pos, start_pos_skosy, end_pos_skosy, point_array, point_color, cell_history):
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

        # --- 3 out of 5 confirmation for Confirmed Occupied (aggregation of cells status)
        if cell_history is not None:
            if len(cell_history) >= 4:
                occupied_count = 0
                for hist in cell_history[-4:]:                                                                           # Loop through last 4 cells of history
                    if hist.cell_status == CellStatus.OCCUPIED:
                        occupied_count += 1
                if self.cell_status == CellStatus.OCCUPIED:
                    occupied_count += 1
                if occupied_count >= 3:
                    self.cell_status = CellStatus.CONFIRMED_OCCUPIED
                    self.visu_cell.color = self.visu_border_color_confirmed

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
