# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
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


def get_clusters_dbscan(pc_frames, eps=0.1, min_points=10):
    print(f"Starting Clustering with DBSCAN")
    labels_frame_list = []
    clustered_pc_frames = pc_frames
    for idx, frame in enumerate(clustered_pc_frames):
        pc_array = np.asarray(frame.points)
        dbscan = DBSCAN(eps=eps, min_samples=min_points)
        labels = dbscan.fit_predict(pc_array)

        max_label = labels.max()
        print(f"Frame {idx +1} has {max_label + 1} Clusters")
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        filtered_colors = colors.copy()
        for idx, point in enumerate(colors[labels < 0]):                                                                # Noise is labeled with -1
            point[:3] = NOISE_COLOR
            filtered_colors[idx] = point
        frame.colors = o3d.utility.Vector3dVector(filtered_colors[:, :3])
        labels_frame_list.append(labels)

    return clustered_pc_frames, labels_frame_list


def get_bounding_boxes(pc_frames, dbscan_labels_frame_list, min_points=10, max_points=100, max_x_size=20, max_y_size=5, max_z_size=10):
    bb_frames = []
    for idx, frame in enumerate(pc_frames):
        bounding_boxes = []
        pc_array = np.asarray(frame.points)

        dbscan_labels = dbscan_labels_frame_list[idx]
        for label in np.unique(dbscan_labels):
            if label == -1:
                continue                                                                                                # Skip Noise

            cluster_indices = np.where(dbscan_labels == label)[0]
            cluster_points = pc_array[cluster_indices]

            if len(cluster_points) < min_points or len(cluster_points) > max_points:                                    # Check min and max points inside BB
                continue

            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)

            if max_coords[0] - min_coords[0] <= max_x_size and \
               max_coords[1] - min_coords[1] <= max_y_size and \
               max_coords[2] - min_coords[2] <= max_z_size:                                                             # Check if BB size logical
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_coords, max_coords)
                bounding_box.color = BOUNDING_BOX_COLOR
                bounding_boxes.append(bounding_box)

        bb_frames.append(bounding_boxes)

    return bb_frames


def get_centroids(clustered_pc_frames, dbscan_labels_frame_list):
    centroids_frame_list = []
    centroid_cross_frames = []

    for idx, frame in enumerate(clustered_pc_frames):
        centroids = []
        centroid_cross_list = []
        dbscan_labels = dbscan_labels_frame_list[idx]
        unique_clusters, counts = np.unique(dbscan_labels, return_counts=True)                                          # Calculate centroid of clusters
        pc_array = np.asarray(frame.points)
        for cluster_id in unique_clusters:
            if cluster_id == -1:                                                                                        # -1 is noise
                continue
            cluster_point_idx = np.where(dbscan_labels == cluster_id)
            cluster_points = [pc_array[point_idx] for point_idx in cluster_point_idx][0]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            centroid_cross_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=centroid))
        centroids_frame_list.append(centroids)
        centroid_cross_frames.append(centroid_cross_list)

    return centroids_frame_list, centroid_cross_frames


def get_grids(clustered_pc_frames, centroids_frame_list, grid_size, grid_cell_size):
    grids_frame_list = []
    grids_geometries_frames = []

    # --- Create a Open3D-PointCloud-Object for each centroid
    for idx, frame in enumerate(clustered_pc_frames):
        centroids = centroids_frame_list[idx]

        # --- Create a 3D-Grid around each centroid
        centroid_grid_point_lists = []
        centroid_lines = []
        for centroid in centroids:
            x_min, x_max = centroid[0] - (grid_size*2) / 2, centroid[0] + (grid_size*2) / 2                             # TODO: Grid Size = Cluster Size + Offset
            y_min, y_max = centroid[1] - grid_size / 2, centroid[1] + grid_size / 2
            z_min, z_max = centroid[2] - grid_size / 2, centroid[2] + grid_size / 2

            # --- Create grid points in 3D grid
            grid_points = []
            for x in np.arange(x_min, x_max + grid_cell_size, grid_cell_size):
                for y in np.arange(y_min, y_max + grid_cell_size, grid_cell_size):
                    for z in np.arange(z_min, z_max + grid_cell_size, grid_cell_size):
                        grid_points.append([x, y, z])
            centroid_grid_point_lists.append(grid_points)

            # --- Create lines between grid points to create a mesh
            lines = []
            for i in range(len(grid_points)):
                for j in range(i + 1, len(grid_points)):
                    if np.linalg.norm(np.array(grid_points[i]) - np.array(grid_points[j])) <= (grid_cell_size + 0.005):
                        lines.append([i, j])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(grid_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([255 / 255, 105 / 255, 180 / 255])
            centroid_lines.append(line_set)

        grids_frame_list.append(centroid_grid_point_lists)
        grids_geometries_frames.append(centroid_lines)

    return grids_frame_list, grids_geometries_frames
