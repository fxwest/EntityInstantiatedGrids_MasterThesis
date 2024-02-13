# -------------------------------
# ----------- Import ------------
# -------------------------------
import copy
import math
import numpy as np
import open3d as o3d
from enum import Enum


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
NOISE_COLOR = [0.5, 0.5, 0.5]
PKW_SIZE = [2.50, 4.00, 2.50]                                                                                           # width, length, height
LKW_SIZE = [3.00, 18.75, 4.00]


# -------------------------------
# -------- Entity Grids ---------
# -------------------------------
def get_entity_grids(tracked_pc_trace):
    print("Starting Entity Grid determination...")
    grids_frame_list = []
    for frame_idx, cluster_frame in enumerate(tracked_pc_trace.pc_frame_list):
        grid_list = []
        point_cloud_all_points = np.asarray(cluster_frame.pcdXYZ.points)
        for cluster in cluster_frame.entity_cluster_list:
            if cluster.dbscan_label != -1:
                if cluster.tracker_age > 0:
                    for grid in grids_frame_list[frame_idx - 1]:
                        if grid.tracker_id == cluster.tracker_id:
                            grid_copy = copy.deepcopy(grid)
                            grid_copy.update_entity_grid(cluster, point_cloud_all_points)
                            grid_list.append(copy.deepcopy(grid_copy))
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
                if self.has_vertical_extend(self.point_array) or self.voxel_pos[2] > 1:                                 # Consider height to not classify the even roof as ground/noise
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
