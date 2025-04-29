import os
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rack_pipe_detector.rack_detector import RackDetector
from rack_pipe_detector.utils import read_config, get_intrinsic_mat
from test_utils import get_default_data_folder

filename = 'example5'


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    detector = RackDetector(config)

    color_img = cv.imread(os.path.join(data_folder, filename + '_color.png'))
    depth_img = Image.open(os.path.join(data_folder, filename + '_depth.png'))
    depth_img = np.array(depth_img, dtype=float) / 1000

    hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)

    cam_intrinsic = get_intrinsic_mat(config['intrinsic'])
    cam_extrinsic = np.array(config['extrinsic'])
    distortion = np.array(config['intrinsic']['distortion'])
    rvec = Rotation.from_matrix(cam_extrinsic[:3, :3]).as_rotvec()
    tvec = cam_extrinsic[:3, 3]

    detections = detector.process(color_img, depth_img)
    print('All detections:')
    for detect in detections:
        print(detect)
        for pipe in detect.model_info.pipes_pos:
            pipes = np.zeros((2, 3), dtype=float)
            pipes[:, :2] = pipe
            pipes[:, 2] = [.00, .02]
            pipes += detect.pos[np.newaxis, :]
            img_pts, _ = cv.projectPoints(pipes, rvec, tvec, cam_intrinsic, distortion)
            img_pts = np.int16(img_pts.reshape(2, -1))
            cv.line(color_img, img_pts[0], img_pts[1], (0, 0, 255), 2)
            img_pts = np.int16(np.mean(img_pts, axis=0))
            if np.sum(img_pts >= 0) == 2:
                print(img_pts, hsv_img[img_pts[1], img_pts[0]])
                if hsv_img[img_pts[1], img_pts[0], 1] >= 220:
                    cv.circle(color_img, img_pts, 2, (0, 255, 0), -1)
        img_pts, _ = cv.projectPoints(detect.pos, rvec, tvec, cam_intrinsic, distortion)
        cv.circle(color_img, np.uint16(img_pts.flatten()), 2, (0, 0, 255), -1)

    cv.imshow('img', color_img)
    cv.waitKey()


if __name__ == '__main__':
    main(filename)
