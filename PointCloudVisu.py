# -------------------------------
# ----------- Import ------------
# -------------------------------
import numpy as np
import open3d as o3d


# -------------------------------
# -------- LiDAR Viewer ---------
# -------------------------------
class LidarViewer:
    def __init__(self, pc_trace, bb_frames=None, centroid_frames=None, grid_frames=None):
        self.pc_frame_list = pc_trace.pc_frame_list
        self.bb_frames = bb_frames
        self.centroid_frames = centroid_frames
        self.grid_frames = grid_frames
        self.num_frames = pc_trace.num_total_frames
        self.num_max_frames = pc_trace.num_max_frames
        self.curr_frame = 0

        # --- Define the visualization window and add the point cloud
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1280, height=720)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        self.vis.add_geometry(self.pc_frame_list[self.curr_frame].pcdXYZ)
        if self.bb_frames:
            for bounding_box in self.bb_frames[self.curr_frame]:
                self.vis.add_geometry(bounding_box)
        if self.centroid_frames:
            for centroid_cross in self.centroid_frames[self.curr_frame]:
                self.vis.add_geometry(centroid_cross)
        if self.grid_frames:
            for grid in self.grid_frames[self.curr_frame]:
                self.vis.add_geometry(grid)

        # --- Add the keyboard and camera callback functions to the visualizer
        self.vis.register_key_callback(256, self.exit_viewer)                                                           # Escape
        self.vis.register_key_callback(262, self.next_frame)                                                            # Right arrow key
        self.vis.register_key_callback(263, self.prev_frame)                                                            # Left arrow key

        self.vis.run()

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
        self.vis.add_geometry(self.pc_frame_list[self.curr_frame].pcdXYZ, reset_bounding_box=False)
        if self.bb_frames:
            for bounding_box in self.bb_frames[self.curr_frame]:
                self.vis.add_geometry(bounding_box, reset_bounding_box=False)
        if self.centroid_frames:
            for centroid_cross in self.centroid_frames[self.curr_frame]:
                pass
                self.vis.add_geometry(centroid_cross, reset_bounding_box=False)
        if self.grid_frames:
            for grid in self.grid_frames[self.curr_frame]:
                pass
                self.vis.add_geometry(grid, reset_bounding_box=False)
        self.vis.poll_events()
        self.vis.update_renderer()
