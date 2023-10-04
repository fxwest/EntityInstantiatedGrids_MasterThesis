# -------------------------------
# ----------- Import ------------
# -------------------------------
import os
import numpy as np
import open3d as o3d


# -------------------------------
# -------- Point Cloud ----------
# -------------------------------
class PointCloudTrace:
    """
    Multiple Point Cloud Frames are defined as a Point Cloud Trace.
    """
    def __init__(self, folder_path, num_max_frames=0):
        self.folder_path = folder_path
        self.num_max_frames = num_max_frames
        self.num_frames, self.raw_pc_frame_list = self.load_raw_point_cloud()

    def load_raw_point_cloud(self):
        """
        Load the raw point cloud frame by frame by iterating through the PCD files.
        """
        raw_pc_frame_list = []
        print(f"Loading Point Cloud from {self.folder_path}")
        print(f"Max Frames Limit active, value: {self.num_max_frames }")
        pc_files = os.listdir(self.folder_path)
        num_frames = len(pc_files)
        print(f'Number of Frames: {num_frames}')

        for frame_idx, pc_file in enumerate(pc_files):
            file_path = self.folder_path + r"\\" + pc_file
            data_frame = np.fromfile(file_path, sep=' ')
            data_frame = np.reshape(data_frame, (-1, 4))
            print(f'Frame {frame_idx + 1}/{num_frames}')
            raw_pc_frame = RawPointCloudFrame(data_frame, frame_idx)
            raw_pc_frame_list.append(raw_pc_frame)
            if self.num_max_frames > 0 and frame_idx == (self.num_max_frames - 1):
                break

        return num_frames, raw_pc_frame_list


class PointCloudFrame:
    """
    Base Class for all Point Cloud Frames
    """
    def __init__(self, data_frame, frame_idx):
        self.frame_idx = frame_idx
        self.num_points = len(data_frame)
        self.pcdXYZ = o3d.geometry.PointCloud()
        print(f'Number of Points: {self.num_points}')


class RawPointCloudFrame(PointCloudFrame):
    """
    Raw Point Cloud loaded from the raw PCD files.
    """
    def __init__(self, data_frame, frame_idx):
        super().__init__(data_frame, frame_idx)
        raw_pc = (data_frame[:, 0:3])
        self.refl = data_frame[:, 3]
        self.pcdXYZ.points = o3d.utility.Vector3dVector(raw_pc)
