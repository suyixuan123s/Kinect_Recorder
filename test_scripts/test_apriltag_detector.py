import os
import cv2 as cv
import numpy as np
from PIL import Image

from rack_pipe_detector.rack_pipe_detector.apriltag_detector import AprilTagDetector, PnPTransform
from rack_pipe_detector.rack_pipe_detector.utils import read_config, get_intrinsic_mat, calc_intersect_pt, \
    calc_center_depth
from test_utils import get_default_data_folder

filename = 'example8'


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    marker_size = config['marker_size']['small']
    intr_mat = get_intrinsic_mat(config['intrinsic'])
    distortion = np.array(config['intrinsic']['distortion'])
    extr_mat = np.array(config['extrinsic'])
    extr_mat = np.concatenate((extr_mat, np.eye(4, dtype=float)[[3]]))

    img = cv.imread(os.path.join(data_folder, filename + '_color.png'))
    depth_img = Image.open(os.path.join(data_folder, filename + '_depth.png'))
    depth_img = np.array(depth_img, dtype=float) / 1000

    res_img = img.copy()

    detector = AprilTagDetector(marker_size, intr_mat, distortion)
    detections = detector.detect(img)

    with open(os.path.join(output_folder, 'marker_result.csv'), 'w') as f:
        for detection in detections:
            tag_id, transforms = detection.tag_id, detection.list_trans
            if not transforms:
                continue

            detection.draw_detection(res_img)

            img_pts = detection.img_pts
            center = calc_intersect_pt(img_pts[[0, 2]], img_pts[[1, 3]])
            depth = calc_center_depth(depth_img, center)

            print('Tag id:', tag_id)
            print(tag_id, end=',', file=f)

            tup = (np.infty, None)
            for trans in transforms:
                print('Difference of depth:', abs(trans.pos[2] - depth))
                trans.pos[2] = depth

                new_trans = trans.get_trans_to_base(extr_mat)

                angle_z = np.arccos(new_trans.to_matrix()[2, 2]) / np.pi * 180
                new_tup = (min(angle_z, abs(angle_z - 90)), new_trans)
                if new_tup < tup:
                    tup = new_tup

            trans = tup[1]
            assert isinstance(trans, PnPTransform)
            # Get translation
            print('Translation:', trans.pos)
            print(*trans.pos, sep=',', end=',', file=f)

            # Get angle between their z axes
            angle_z = np.arccos(trans.to_matrix()[2, 2]) / np.pi * 180
            print('Angle:', angle_z)
            print(angle_z, end=',', file=f)

            print('Re-projection error:', trans.error)
            print('All errors:', [trans.error for trans in transforms])
            print(trans.error, file=f)
            print()

    cv.imshow('detection_results', res_img)
    cv.waitKey()
    cv.imwrite(os.path.join(output_folder, 'marker_detection.png'), res_img)


if __name__ == '__main__':
    main(filename)
