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
def get_cluster_tracks(clusters_frame_list, used_tracker_ids):
    # --- Get Tracker/Cluster_ID x Frame_ID Matrix
    cluster_id_frames = [[None for frame in clusters_frame_list] for tracker_id in used_tracker_ids]
    for frame_idx, frame in enumerate(clusters_frame_list):
        for tracker_id in used_tracker_ids:
            for cluster in frame:
                if cluster.tracker_id == tracker_id:
                    cluster_id_frames[tracker_id][frame_idx] = cluster

    # --- Update Centroid Coordinates with estimated coordinates from KALMAN-Filter
    for cluster_id, cluster_frames in enumerate(cluster_id_frames):
        cluster_frames_filtered = [i for i in cluster_frames if i is not None]
        updated_cluster_frames = ClusterTrack(cluster_frames_filtered)
        for frame_idx, frame in enumerate(clusters_frame_list):
            for cluster in frame:
                if cluster.tracker_id == cluster_id:
                    cluster.centroid = updated_cluster_frames.estimated_centroids[0]                                    # Take first estimated coordinate
                    updated_cluster_frames.estimated_centroids.pop(0)                                                   # Drop first estimated coordinate to handle frames before tracker_id

    return clusters_frame_list


class ClusterTrack:
    def __init__(self, cluster_frames):
        self.measured_centroids = np.array([frame.centroid for frame in cluster_frames])
        x0 = np.array([self.measured_centroids[0, 0], self.measured_centroids[0, 1], self.measured_centroids[0, 2],
                       0, 0, 0])                                                                                        # Starting coordinates and velocity
        self.x = x0
        self.P = P0
        self.xs = []
        self.Ps = []
        self.track_cluster(cluster_frames)
        self.plot_filter_result()
        self.estimated_centroids = [[xs[0], xs[1], xs[2]] for xs in self.xs]
        self.update_centroids(cluster_frames)

    def track_cluster(self, cluster_frames):
        for i in range(len(cluster_frames)):
            # --- Prediction Step
            self.x = A @ self.x                                                                                         # State estimation
            self.P = A @ self.P @ A.T + Q                                                                               # Covariance estimation

            # --- Update Step
            z = cluster_frames[i].centroid                                                                              # Cluster measurement of first frame
            y = z - H @ self.x
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)                                                                         # KALMAN-Gain
            self.x = self.x + K @ y                                                                                     # State update
            self.P = (np.eye(6) - K @ H) @ self.P                                                                       # Covariance update

            # --- Save update
            self.xs.append(self.x)
            self.Ps.append(self.P)

        self.xs = np.array(self.xs)                                                                                     # Save as array
        self.Ps = np.array(self.Ps)

    def update_centroids(self, cluster_frames):
        for frame_idx, cluster in enumerate(cluster_frames):
            cluster.centroid = self.estimated_centroids[frame_idx]
        return cluster_frames

    def plot_filter_result(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xs[:, 0], self.xs[:, 1], self.xs[:, 2], c='r', label='Estimated Position')
        ax.scatter(self.measured_centroids[:, 0], self.measured_centroids[:, 1], self.measured_centroids[:, 2], c='g',
                   label='Measured Position')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()
