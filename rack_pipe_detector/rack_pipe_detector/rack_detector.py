import numpy as np
from scipy.spatial.transform import Rotation

from rack_marker_detector import RackMarkerDetector
from rack_model import RackModelInfo, RackModelManager
from utils import arccos, calc_angle, calc_avg_rot
from macros import Color


class RackDetection:
    def __init__(self, color: Color, pos: np.ndarray, orn: Rotation, model_info: RackModelInfo):
        self.color = color
        self.pos, self.orn = pos, orn
        self.model_info = model_info

    def __repr__(self) -> str:
        return 'Rack color {0} pos {1} orn (zyx) {2}'.format(self.color, str(self.pos),
                                                             str(self.orn.as_euler('zyx', degrees=True)))


class RackDetector:
    def __init__(self, config, angle_thresh=15, angle_thresh2=40):
        self.config = config
        self.rack_model = RackModelManager(config)
        self.angle_thresh = angle_thresh
        self.angle_thresh2 = angle_thresh2

        self.detector = RackMarkerDetector(config)

        self.detections = None
        self.rackets = None

    def del_detections(self, i, j):
        del self.detections[i]
        if j > i:
            j -= 1
        del self.detections[j]

    def find_white_rack(self) -> bool:
        rack_model_info = self.rack_model.get_info(Color.WHITE)

        # First, find the reversed AprilTag with specific ID
        for i in range(len(self.detections)):
            detect1 = self.detections[i]
            if detect1.tag_id != rack_model_info.tag_id:
                continue
            detect1_y = detect1.trans.to_matrix()[:3, 1]
            if detect1_y[1] > 0 and False:
                # This tag is not reversed
                continue

            # Then, find the closest corresponding AprilTag
            tup = (np.infty, -1)
            for j in range(len(self.detections)):
                if i == j:
                    continue
                detect2 = self.detections[j]
                if detect2.tag_id != rack_model_info.tag_id:
                    continue

                # Check that their y axis are aligned
                detect2_y = detect2.trans.to_matrix()[:3, 1]
                dif_ang = calc_angle(-detect1_y, detect2_y)
                if dif_ang > self.angle_thresh:
                    continue
                # Check the angle between tag 1 y axis and the vector of position difference
                delta_pos = detect2.trans.pos - detect1.trans.pos
                dif_ang = calc_angle(-detect1_y, delta_pos)
                if dif_ang > self.angle_thresh:
                    continue

                # Get the distance of their center
                dis = np.linalg.norm(delta_pos)
                new_tup = (dis, j)
                if new_tup < tup:
                    tup = new_tup

            j = tup[1]
            if j == -1:
                # Cannot find the corresponding marker
                continue

            # Find a match
            detect2 = self.detections[j]

            color = Color.WHITE
            pos = .5 * (detect1.trans.pos + detect2.trans.pos)
            detect1.trans.inverse_xy()
            orn1 = detect1.trans.orn
            orn2 = detect2.trans.orn
            orn = calc_avg_rot([orn1, orn2])

            if orn.as_matrix()[1, 1] < 0:
                # Inverse x and y axis since it's symmetrical
                orn *= Rotation.from_euler('z', 180, degrees=True)
            # Rotate the x and y axis
            orn *= Rotation.from_euler('z', -90, degrees=True)

            self.rackets.append(RackDetection(color, pos, orn, rack_model_info))

            self.del_detections(i, j)
            return True

        return False

    def find_blue_or_green_rack(self, color: Color) -> bool:
        rack_model_info = self.rack_model.get_info(color)

        # First, find the left AprilTag with specific ID
        for i in range(len(self.detections)):
            detect1 = self.detections[i]
            if detect1.tag_id != rack_model_info.tag_id:
                continue
            detect1_x = detect1.trans.to_matrix()[:3, 0]

            # Then, find the closest corresponding AprilTag
            tup = (np.infty, -1)
            for j in range(len(self.detections)):
                if i == j:
                    continue
                detect2 = self.detections[j]
                if detect2.tag_id != rack_model_info.tag_id:
                    continue

                # Check that their x axis are aligned
                detect2_x = detect2.trans.to_matrix()[:3, 0]
                dif_ang = calc_angle(detect1_x, detect2_x)
                if dif_ang > self.angle_thresh:
                    continue
                # Check the angle between x axis and the vector of position difference
                delta_pos = detect2.trans.pos - detect1.trans.pos
                dif_ang = calc_angle(detect1_x, delta_pos)
                if dif_ang > self.angle_thresh2:
                    continue

                # Get the distance of their center
                dis = np.linalg.norm(delta_pos)
                new_tup = (dis, j)
                if new_tup < tup:
                    tup = new_tup

            j = tup[1]
            if j == -1:
                # Cannot find the corresponding marker
                continue

            # Find a match
            detect2 = self.detections[j]

            pos = .5 * (detect1.trans.pos + detect2.trans.pos)
            orn1 = detect1.trans.orn
            orn2 = detect2.trans.orn
            orn = calc_avg_rot([orn1, orn2])

            # Rotate the y and z axis
            orn *= Rotation.from_euler('x', -90, degrees=True)
            # Rotate the x and y axis
            orn *= Rotation.from_euler('z', -90, degrees=True)

            # Move pos to the center
            move_dis = rack_model_info.size[0] * .5
            pos -= orn.as_matrix()[0] * move_dis
            # Move pos to the upper surface
            move_dis = rack_model_info.extra_info['delta_z']
            pos += orn.as_matrix()[2] * move_dis

            self.rackets.append(RackDetection(color, pos, orn, rack_model_info))

            self.del_detections(i, j)
            return True

        return False

    def process(self, color_img: np.ndarray, depth_img: np.ndarray | None = None):
        self.detections = self.detector.process(color_img, depth_img)
        self.rackets = []

        while self.find_white_rack():
            continue
        while self.find_blue_or_green_rack(Color.BLUE):
            continue
        while self.find_blue_or_green_rack(Color.GREEN):
            continue

        return self.rackets
