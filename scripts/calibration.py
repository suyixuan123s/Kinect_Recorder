import os
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from rack_pipe_detector import AprilTagDetection, AprilTagDetector
from rack_pipe_detector.utils import read_config, get_intrinsic_mat, write_config


def calc_poly_size(pts):
    area = 0.
    for i in range(1, pts.shape[0]):
        area += np.cross(pts[i], pts[(i + 1) % pts.shape[0]])
    return abs(area * .5)


def main(data_folder: str):
    config = read_config(data_folder)
    intr_mat = get_intrinsic_mat(config['intrinsic'])
    distortion = np.array(config['intrinsic']['distortion'])

    marker_size = config['marker_size']['large']

    detector = AprilTagDetector(marker_size, intr_mat, distortion)

    img_path = os.path.join(data_folder, 'calibration.png')
    if not os.path.exists(img_path):
        print('Image', img_path, 'does not exist')
        return

    img = cv.imread(img_path)

    detections = detector.detect(img)

    # Find largest AprilTag
    tup = (-1, None)
    for detection in detections:
        area = calc_poly_size(detection.img_pts)
        if tup < (area, detection):
            tup = (area, detection)

    detection = tup[1]
    assert isinstance(detection, AprilTagDetection)

    corners = np.int16(np.round(detection.img_pts))
    for i in range(4):
        cv.line(img, corners[i], corners[(i + 1) % 4], (0, 0, 255), 2)
    cv.imshow('base_marker', img)
    cv.waitKey()

    trans = detection.list_trans[0]

    extr_mat = np.zeros((3, 4), dtype=float)
    extr_mat[:3, 3] = trans.pos
    extr_mat[:3, :3] = trans.orn.as_matrix()
    print(extr_mat)

    config['extrinsic'] = extr_mat.tolist()
    write_config(config, data_folder)


if __name__ == '__main__':
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    main(data_folder)
