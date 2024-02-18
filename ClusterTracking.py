# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# -------- Hyperparameter -------
# -------------------------------
FRAME_RATE = 10
P0 = np.eye(6) * 1000                                                                                                   # Starting covariance (high uncertainty)
Q = np.eye(6) * 0.01                                                                                                    # Process noise covariance
R = np.eye(3) * 0.1                                                                                                     # Measurement noise covariance


# -------------------------------
# ------------ Models -----------
# -------------------------------
dt = 1 / FRAME_RATE                                                                                                     # Time interval between two frames
A = np.array(                                                                                                           # System matrix
    [
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]
)
H = np.array(                                                                                                           # Measurement matrix
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ]
)


# -------------------------------
# -------- KALMAN Filter --------
# -------------------------------
def get_cluster_tracks(clustered_pc_trace, max_dist=0.5, plot_kalman_results=False):
    used_tracker_ids = []
    entity_cluster_tracks = []
    print("Starting Cluster Tracking with KALMAN-Filter")
    for frame_idx, frame_pc in enumerate(clustered_pc_trace.pc_frame_list):
        if frame_idx == 0:
            for entity_cluster in frame_pc.entity_cluster_list:
                used_tracker_ids.append(entity_cluster.tracker_id)                                                      # Keep DBSCAN-Label as Tracker ID in first frame
                entity_cluster_tracks.append(ClusterTrack(entity_cluster.tracker_id, entity_cluster.anchor_point))      # Init new tracks for each cluster in first frame
        else:
            prev_clusters_list = [entity_cluster for entity_cluster in clustered_pc_trace.pc_frame_list[frame_idx-1].entity_cluster_list]
            for entity_cluster in frame_pc.entity_cluster_list:
                found_track = False
                for cluster_track in entity_cluster_tracks:
                    dist = entity_cluster.euclidean_distance(cluster_track.predicted_anchor_point)                      # Distance between measurement and prediction
                    if dist <= max_dist:
                        entity_cluster.tracker_id = cluster_track.tracker_id
                        for prev_cluster in prev_clusters_list:
                            if prev_cluster.tracker_id == cluster_track.tracker_id:
                                entity_cluster.copy_color(prev_cluster)
                                entity_cluster.tracker_age = prev_cluster.tracker_age + 1
                                cluster_track.update_track(entity_cluster.anchor_point)
                                entity_cluster.anchor_point = cluster_track.estimated_anchor_point
                                found_track = True
                                break
                        if found_track:
                            break
                if found_track:
                    continue
                else:
                    entity_cluster.tracker_id = used_tracker_ids[-1] + 1                                                # If no matching prev cluster was found -> new track
                    entity_cluster.tracker_age = 0
                    used_tracker_ids.append(entity_cluster.tracker_id)
                    entity_cluster_tracks.append(ClusterTrack(entity_cluster.tracker_id, entity_cluster.anchor_point))
        frame_pc.merge_point_clouds()

    if plot_kalman_results:
        for cluster_track in entity_cluster_tracks:
            cluster_track.plot_filter_result()

    return clustered_pc_trace


class ClusterTrack:
    def __init__(self, tracker_id, measured_anchor_point):
        self.predicted_anchor_point = None
        self.estimated_anchor_point = None
        self.measured_anchor_points = []
        self.estimated_anchor_points = []
        x0 = np.array([measured_anchor_point[0], measured_anchor_point[1], measured_anchor_point[2], 0, 0, 0])          # Starting coordinates and velocity
        self.x = x0
        self.P = P0
        self.xs = []
        self.Ps = []
        self.tracker_id = tracker_id
        self.update_track(measured_anchor_point)

    def update_track(self, measured_anchor_point):
        self.measured_anchor_points.append(measured_anchor_point)
        self.estimated_anchor_point = self.x[:4 -1]
        self.estimated_anchor_points.append(self.estimated_anchor_point)

        # --- Prediction Step
        self.x = A @ self.x                                                                                             # State estimation
        self.P = A @ self.P @ A.T + Q                                                                                   # Covariance estimation

        # --- Update Step
        z = measured_anchor_point
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)                                                                             # KALMAN-Gain
        self.x = self.x + K @ y                                                                                         # State update
        self.P = (np.eye(6) - K @ H) @ self.P                                                                           # Covariance update
        self.xs.append(self.x)
        self.Ps.append(self.P)

        self.predicted_anchor_point = self.xs[-1][:3]

    def plot_filter_result(self):
        xs = np.array(self.xs)
        self.estimated_anchor_points = np.array(self.estimated_anchor_points)
        self.measured_anchor_points = np.array(self.measured_anchor_points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.estimated_anchor_points[:, 0], self.estimated_anchor_points[:, 1], self.estimated_anchor_points[:, 2], c='y',
                   marker='v', label='Estimated Position')
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c='r', marker='x', label='Predicted Position')
        ax.scatter(self.measured_anchor_points[:, 0], self.measured_anchor_points[:, 1], self.measured_anchor_points[:, 2], c='g', label='Measured Position')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()
