# This is only for calibrating the extrinsic matrix of the camera
# Needs prior information about its intrinsic parameters

import os
import numpy as np
import cv2 as cv

from apriltag_detector import AprilTagDetector
from utils import read_config, write_config, get_intrinsic_mat, calc_angle


def calc_polygon_area(pts: np.ndarray):
    area = 0
    for i in range(pts.shape[0]):
        area += np.cross(pts[i], pts[(i + 1) % pts.shape[0]])
    return np.abs(area) / 2


def main(folder='data'):
    config = read_config(folder)
    marker_size = config['marker_size']['large']
    intr_mat = get_intrinsic_mat(config['intrinsic'])
    distortion = np.array(config['intrinsic']['distortion'])

    detector = AprilTagDetector(marker_size, intr_mat, distortion)

    img = cv.imread(os.path.join(folder, 'calibration.png'))
    detections = detector.detect(img)

    max_area, max_area_detection = -1, None
    for detection in detections:
        if not detection.list_trans: continue
        area = calc_polygon_area(detection.img_pts)
        if area > max_area:
            max_area, max_area_detection = area, detection

    max_area_detection.draw_detection(img)
    cv.imshow('Result', img)
    cv.waitKey(0)

    trans = max_area_detection.list_trans[0]
    extr_mat = trans.to_matrix()[:3, :]

    config['extrinsic'] = extr_mat.tolist()
    write_config(config, folder)

    print('Extrinsic matrix:')
    print(extr_mat)
    angle = 180 - calc_angle(extr_mat[:, 2], np.array([0, 0, 1]))
    print('Angle of the camera (in degrees):', angle)


if __name__ == '__main__':
    folder = 'data'
    main(folder)
