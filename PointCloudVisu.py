# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d


# -------------------------------
# -------- LiDAR Viewer ---------
# -------------------------------
class LidarViewer:
    def __init__(self, pc_frames, num_frames, num_max_frames, bb_frames=None):
        self.pc_frames = pc_frames
        self.bb_frames = bb_frames
        self.num_frames = num_frames
        self.num_max_frames = num_max_frames
        self.curr_frame = 0

        # --- Define the visualization window and add the point cloud
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1280, height=720)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        for point_cloud in self.pc_frames:
            self.vis.add_geometry(point_cloud[self.curr_frame])
        if self.bb_frames:
            for bounding_box in self.bb_frames[self.curr_frame]:
                self.vis.add_geometry(bounding_box)

        # --- Add the keyboard and camera callback functions to the visualizer
        self.vis.register_key_callback(256, self.exit_viewer)                                                            # Escape
        self.vis.register_key_callback(262, self.next_frame)                                                            # Right arrow key
        self.vis.register_key_callback(263, self.prev_frame)                                                            # Left arrow key

        self.vis.poll_events()
        self.vis.run()
        self.vis.destroy_window()

    def exit_viewer(self, vis=None):
        if vis:
            self.vis = vis
        print('Exit LiDAR Viewer...')
        self.vis.destroy_window()

    def next_frame(self, vis=None):
        if vis:
            self.vis = vis
        if self.curr_frame < self.num_frames:
            if self.curr_frame >= self.num_max_frames - 1 and self.num_max_frames > 0:
                return
            self.curr_frame += 1
            self.update_point_cloud()

    def prev_frame(self, vis=None):
        if vis:
            self.vis = vis
        self.vis = vis
        if self.curr_frame >= 1:
            self.curr_frame -= 1
            self.update_point_cloud()

    def update_point_cloud(self):
        self.vis.clear_geometries()
        for frame in self.pc_frames:
            self.vis.add_geometry(frame[self.curr_frame])
        if self.bb_frames:
            for bounding_box in self.bb_frames[self.curr_frame]:
                self.vis.add_geometry(bounding_box)
        self.vis.poll_events()
        self.vis.update_renderer()
