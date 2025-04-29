import os
import numpy as np
import cv2 as cv
import open3d as o3d
from scipy.spatial.transform import Rotation
from typing import Union
from macros import Color


class RackModelInfo:
    def __init__(self, tag_id: int, size: np.ndarray, pipes_pos=None, extra_info=None):
        self.tag_id = tag_id

        self.size = size

        if pipes_pos is None:
            pipes_pos = []
        if extra_info is None:
            extra_info = {}
        self.pipes_pos = pipes_pos
        self.extra_info = extra_info


class RackModelManager:
    def __init__(self, config):
        self.config = config
        self.rack_config = config['rack']

        self.dict_rack_info = {}
        for color in self.rack_config.keys():
            info = self.rack_config[color]
            tag_id = info['tag_id']
            if 'file' in info:
                size, pipes_pos = self._get_rack_model_size_pipes(info['file'])
            else:
                size = info['size']
                pipes_pos = []
            if 'extra_info' in info:
                extra_info = info['extra_info']
            else:
                extra_info = None
            self.dict_rack_info[color] = RackModelInfo(tag_id, size, pipes_pos, extra_info)

    def get_info(self, color: Union[str, Color]):
        if isinstance(color, Color):
            color = color.name
        return self.dict_rack_info[color]

    def _get_rack_model_size_pipes(self, filename) -> tuple:
        path = os.path.join(self.config['folder'], filename)
        mesh = o3d.io.read_triangle_mesh(path)

        # Get its bounding box for its size
        box = mesh.get_axis_aligned_bounding_box()
        box_size = box.get_extent() * .001
        size = box_size[[0, 2]]

        # Crop the mesh to only its top surface
        surface_box_min = box.get_min_bound()
        surface_box_max = box.get_max_bound()
        surface_box_min[1] = surface_box_max[1] - 1  # the box only has a height of 1 mm
        surface_box = o3d.geometry.AxisAlignedBoundingBox(surface_box_min, surface_box_max)
        surface_mesh = mesh.crop(surface_box)
        surface_mesh = surface_mesh.translate(-surface_mesh.get_axis_aligned_bounding_box().get_center())
        surface_mesh = surface_mesh.rotate(Rotation.from_euler('x', 90, degrees=True).as_matrix())

        # Take a picture of the surface
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(surface_mesh)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()

        # Format the picture to grayscale of OpenCv
        img = np.mean(img, axis=2)
        img = np.array(img < 1., dtype=np.uint8) * 255

        # Detect circles
        img = cv.blur(img, (5, 5))
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, img.shape[0] / 20, param2=50, minRadius=20, maxRadius=100)
        circles = circles.reshape(-1, 3)

        pipes = []

        for circle in circles:
            # Transform from image coordinates to world coordinates
            coor = np.concatenate((circle[:2], [1]))
            coor *= cam_param.extrinsic[2, 3]
            coor = np.linalg.inv(cam_param.intrinsic.intrinsic_matrix) @ coor
            coor = np.concatenate((coor, [1.]))
            pipes.append(coor[:2] * .001)

        return size, pipes
