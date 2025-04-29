import os
import time
import cv2 as cv
import numpy as np
from PIL import Image

from rack_pipe_detector.rack_detector import RackDetector
from rack_pipe_detector.pipe_detector import PipeDetector
from rack_pipe_detector.utils import read_config
from test_utils import get_default_data_folder

filename = 'example1'


def show_img_with_mask(color_img, mask, title='hsv_img'):
    hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] // 2 + hsv_img[:, :, 2] * mask // 2
    cv.imshow(title, cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR))


def draw_point(img, pos, col, radius=3):
    cv.circle(img, np.int16(pos.flatten()), radius, col, -1)


def draw_text(img, pos, txt, col, fontsize=2):
    cv.putText(img, txt, np.int16(pos.flatten()), cv.FONT_HERSHEY_PLAIN, fontsize, col)


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    rack_detector = RackDetector(config)
    pipe_detector = PipeDetector(config)

    color_img = cv.imread(os.path.join(data_folder, filename + '_color.png'))
    depth_img = Image.open(os.path.join(data_folder, filename + '_depth.png'))
    depth_img = np.array(depth_img, dtype=float) / 1000
    depth_img = None

    st_tme = time.time()
    racks = rack_detector.process(color_img, depth_img)
    pipes = pipe_detector.process(color_img, racks)
    print('Overall detection time (seconds):', time.time() - st_tme)

    print(racks)
    print(pipes)

    for rack in racks:
        img_pts = pipe_detector.obj_proj_to_img(rack.pos)
        draw_point(color_img, img_pts, (255, 0, 0))
        draw_text(color_img, img_pts, str(rack.color)[6], (255, 0, 0))

    for pipe in pipes:
        img_pts = pipe_detector.obj_proj_to_img(pipe.pos + [0, 0, .01])
        draw_point(color_img, img_pts, (0, 0, 255))
        draw_text(color_img, img_pts, str(pipe.color)[6], (0, 0, 255))

    cv.imshow('img', color_img)
    cv.waitKey()
    cv.destroyAllWindows()
    cv.imwrite(os.path.join(output_folder, '%s_result.png' % filename), color_img)

    print()


if __name__ == '__main__':
    example_ids = range(1, 9)
    for i in example_ids:
        main('example%d' % i)
