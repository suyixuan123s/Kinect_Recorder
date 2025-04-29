import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from .rack_detector import RackDetection
from .utils import get_intrinsic_mat
from .macros import Color


class PipeDetection:
    def __init__(self, color: Color, pos: np.ndarray, orn: Rotation):
        self.color = color
        self.pos = pos
        self.orn = orn

    def __repr__(self) -> str:
        return 'Pipe color {0} pos {1} orn (zyx) {2}'.format(self.color, str(self.pos),
                                                             str(self.orn.as_euler('zyx', degrees=True)))


def _get_line_pts(endpts):
    pt_x = np.arange(np.floor(endpts[0, 0]), np.ceil(endpts[1, 0]), dtype=int)
    pt_y = np.linspace(*endpts[:, 1], pt_x.shape[0], endpoint=True, dtype=float)
    pt_y1, pt_y2 = map(np.int16, (np.floor(pt_y), np.ceil(pt_y)))
    return np.concatenate((np.stack((pt_x, pt_y1), axis=1),
                           np.stack((pt_x, pt_y2), axis=1)), dtype=int)


class PipeDetectorBase:
    def __init__(self, config):
        self.config = config
        self.intr_mat = get_intrinsic_mat(config['intrinsic'])
        self.distortion = np.array(config['intrinsic']['distortion'])
        self.extr_mat = np.array(config['extrinsic'])
        self.extr_mat = np.concatenate((self.extr_mat, np.eye(4)[[3]]))
        self.rvec = Rotation.from_matrix(self.extr_mat[:3, :3]).as_rotvec()
        self.tvec = self.extr_mat[:3, 3]

        self.mask = None
        self.lb = None
        self.rb = None

    def _get_mask(self, color_img: np.ndarray):
        hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
        if self.lb[0] > self.rb[0]:
            hsv_img = np.uint16(hsv_img)
            t_lb, t_rb = self.lb.copy(), self.rb.copy()
            t_lb[0], t_rb[0] = 0, 180
            mask1 = cv.inRange(hsv_img, self.lb, t_rb)
            mask2 = cv.inRange(hsv_img, t_lb, self.rb)
            mask = mask1 + mask2
        else:
            mask = cv.inRange(hsv_img, self.lb, self.rb)
        ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ele)

    def obj_proj_to_img(self, obj_pts):
        return cv.projectPoints(obj_pts, self.rvec, self.tvec, self.intr_mat, self.distortion)[0]

    def process(self, color_img: np.ndarray, rack_detections: list[RackDetection] = []):
        raise NotImplementedError


class GreenPipeDetector(PipeDetectorBase):
    def __init__(self, config):
        super().__init__(config)

        config_pipe = config['pipe_detector']['GREEN']
        self.lb = np.uint16(config_pipe['lower_bound'])
        self.rb = np.uint16(config_pipe['upper_bound'])
        self.height = config_pipe['height']

    def _test_pipe_exists(self, pipe_pos, debug_img=None):
        pipe = np.array([pipe_pos, pipe_pos])
        pipe[:, 2] += self.height

        # Get the image coordinates of this line
        img_pts = self.obj_proj_to_img(pipe).reshape(2, 2)
        if img_pts[0, 0] > img_pts[1, 0]:
            img_pts[[0, 1]] = img_pts[[1, 0]]
        img_pts[1, 0] += 1

        # Get all points on this line
        pt_xy = _get_line_pts(img_pts)

        if debug_img is not None:
            debug_img[pt_xy[:, 1], pt_xy[:, 0]] = [0, 0, 255]

        return self.mask[pt_xy[:, 1], pt_xy[:, 0]].any()

    def process(self, color_img: np.ndarray, rack_detections: list[RackDetection] = []):
        self._get_mask(color_img)
        results = []

        for rack in rack_detections:
            for pipe_pos in rack.model_info.pipes_pos:
                pipe_pos = np.concatenate((pipe_pos, [0.]))
                pipe_pos += rack.pos
                if self._test_pipe_exists(pipe_pos):
                    results.append(PipeDetection(Color.GREEN, pipe_pos, rack.orn))
        return results


class RedPipeDetector(PipeDetectorBase):
    def __init__(self, config):
        super().__init__(config)

        config_pipe = config['pipe_detector']['RED']
        self.lb = np.uint16(config_pipe['lower_bound'])
        self.rb = np.uint16(config_pipe['upper_bound'])
        self.radius = config_pipe['radius']
        self.rect_size = config_pipe['rect_size']

    def _test_pipe_exists(self, pipe_pos, debug_img=None):
        pipe_pos = np.array(pipe_pos, dtype=float)
        pipe_pos[0] += self.radius
        img_pt = self.obj_proj_to_img(pipe_pos).flatten()
        lb = np.int16(np.floor(img_pt - self.rect_size * .5))
        lb[lb <= 0] = 0
        rb = np.int16(np.ceil(img_pt + self.rect_size * .5)) + 1
        return self.mask[lb[1]:rb[1], lb[0]:rb[0]].any()

    def process(self, color_img: np.ndarray, rack_detections: list[RackDetection] = []):
        self._get_mask(color_img)
        results = []

        for rack in rack_detections:
            for pipe_pos in rack.model_info.pipes_pos:
                pipe_pos = np.concatenate((pipe_pos, [0.]))
                pipe_pos += rack.pos
                if self._test_pipe_exists(pipe_pos):
                    results.append(PipeDetection(Color.RED, pipe_pos, rack.orn))
        return results


class PipeDetector:
    def __init__(self, config):
        self.config = config
        self.detectors = [GreenPipeDetector(config), RedPipeDetector(config)]

    def obj_proj_to_img(self, obj_pts):
        return self.detectors[0].obj_proj_to_img(obj_pts)

    def process(self, color_img: np.ndarray, rack_detections: list[RackDetection] = []):
        results = []
        for detector in self.detectors:
            results += detector.process(color_img, rack_detections)
        return results
