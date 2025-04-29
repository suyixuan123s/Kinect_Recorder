from rack_detector import RackDetector
from pipe_detector import PipeDetector


class RackPipeDetector:
    def __init__(self, config):
        self.config = config
        self.rack_detector = RackDetector(config)
        self.pipe_detector = PipeDetector(config)

    def obj_proj_to_img(self, obj_pts):
        return self.pipe_detector.obj_proj_to_img(obj_pts)

    def detect(self, color_img, depth_img=None):
        racks = self.rack_detector.process(color_img, depth_img)
        pipes = self.pipe_detector.process(color_img, racks)
        return racks, pipes
