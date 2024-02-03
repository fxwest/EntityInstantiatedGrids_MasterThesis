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
        self.num_total_frames, self.pc_frame_list = self.load_raw_point_cloud()

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

    def trim_pc_fov(self, x_axis, y_axis, z_axis):
        """
        Trims the Field of View of all Point Cloud Frames.
        """
        print(f"Starting FOV Trimming")
        for frame_idx, pc_frame in enumerate(self.pc_frame_list):
            trimmed_pc = []
            trimmed_refl = []
            for point_idx, point in enumerate(pc_frame.point_array):
                if x_axis[1] > point[0] > x_axis[0]:                                                                    # Filter X
                    if y_axis[1] > point[1] > y_axis[0]:                                                                # Filter Y
                        if z_axis[1] > point[2] > z_axis[0]:                                                            # Filter Z
                            trimmed_pc.append(point)
                            trimmed_refl.append(pc_frame.refl[point_idx])
            self.pc_frame_list[frame_idx] = TrimmedPointCloudFrame(pc_frame, trimmed_pc, trimmed_refl)


class PointCloudFrame:
    """
    Base Class for all Point Cloud Frames.
    """
    octree_max_depth = 8

    def __init__(self, data_frame, frame_idx):
        self.frame_idx = frame_idx
        self.num_points = len(data_frame)
        self.pcdXYZ = o3d.geometry.PointCloud()
        self.octree = o3d.geometry.Octree(max_depth=self.octree_max_depth)
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
        self.octree.convert_from_point_cloud(self.pcdXYZ)
        self.point_array = np.asarray(self.pcdXYZ.points)


class TrimmedPointCloudFrame(PointCloudFrame):
    """
    Trimmed Point Cloud Frame.
    """
    def __init__(self, raw_pc_frame, trimmed_pc, trimmed_refl):
        self.frame_idx = raw_pc_frame.frame_idx
        self.pcdXYZ = raw_pc_frame.pcdXYZ
        self.pcdXYZ.points = o3d.utility.Vector3dVector(trimmed_pc)
        self.octree = raw_pc_frame.octree
        self.octree.convert_from_point_cloud(self.pcdXYZ)
        self.refl = np.array(trimmed_refl)
        self.point_array = np.asarray(self.pcdXYZ.points)
        self.num_points = len(self.point_array)

