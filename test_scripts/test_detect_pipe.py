import os
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rack_pipe_detector.rack_pipe_detector.rack_detector import RackDetector, RackDetection
from rack_pipe_detector.rack_pipe_detector.utils import read_config, get_intrinsic_mat
from test_utils import get_default_data_folder

filename = 'example1'


def show_img_with_mask(color_img, mask, title='hsv_img'):
    hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    masked_vue = hsv_img[:, :, 2] * (mask // 255)
    unmasked_vue = np.float32(hsv_img[:, :, 2] * (1 - mask // 255))
    alpha = .05
    hsv_img[:, :, 2] = np.uint8(masked_vue + unmasked_vue * alpha)
    cv.imshow('hsv_img', cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR))


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    detector = RackDetector(config)

    color_img = cv.imread(os.path.join(data_folder, filename + '_color.png'))
    depth_img = Image.open(os.path.join(data_folder, filename + '_depth.png'))
    depth_img = np.array(depth_img, dtype=float) / 1000
    # depth_img = None

    hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(hsv_img, (80, 150, 100), (95, 255, 255))
    mask1 = cv.inRange(hsv_img, (0, 100, 0), (20, 255, 150))
    mask2 = cv.inRange(hsv_img, (160, 100, 0), (180, 255, 150))
    mask = mask1 + mask2
    ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ele)
    mask = cv.dilate(mask, ele)

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
            pipes[:, 2] = [.00, .015]
            pipes += detect.pos[np.newaxis, :]

            pipes[:, 0] += 0.007
            pipes[:, 1] += 0.005
            img_pts, _ = cv.projectPoints(pipes, rvec, tvec, cam_intrinsic, distortion)
            img_pts = np.int16(img_pts.reshape(2, -1))
            cv.line(color_img, img_pts[0], img_pts[1], (0, 0, 255), 2)
            # img_pts = np.int16(np.mean(img_pts, axis=0))
            img_pts = np.int16(img_pts[0])
            if np.sum(img_pts >= 0) == 2:
                print(img_pts, hsv_img[img_pts[1], img_pts[0]])
                if hsv_img[img_pts[1], img_pts[0], 0] <= 50 or hsv_img[img_pts[1], img_pts[0], 1] <= 220:
                    cv.circle(color_img, img_pts, 2, (0, 255, 0), -1)
        img_pts, _ = cv.projectPoints(detect.pos, rvec, tvec, cam_intrinsic, distortion)
        cv.circle(color_img, np.uint16(img_pts.flatten()), 2, (0, 0, 255), -1)

    show_img_with_mask(color_img, mask)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    example_ids = [1, 5]
    for i in example_ids:
        main('example%d' % i)
