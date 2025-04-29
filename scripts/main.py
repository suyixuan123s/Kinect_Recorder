import os
import time
import argparse
import cv2 as cv
import numpy as np
from PIL import Image

from rack_pipe_detector.rack_pipe_detector import RackPipeDetector, read_config


def parse_args():
    default_data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=default_data_folder)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--id', type=str, default='')
    return parser.parse_args()


def draw_point(img, pos, col, radius=3):
    cv.circle(img, np.int16(pos.flatten()), radius, col, -1)


def draw_text(img, pos, txt, col, fontsize=2):
    cv.putText(img, txt, np.int16(pos.flatten()), cv.FONT_HERSHEY_PLAIN, fontsize, col)


config = None
detector = None


def main(filename: str, data_folder: str):
    global config, detector
    if config is None:
        config = read_config(data_folder)
    if detector is None:
        st_tme = time.time()
        detector = RackPipeDetector(config)
        print('Time for initializing detector (seconds):', time.time() - st_tme)

    color_path = os.path.join(data_folder, filename + '_color.png')
    if not os.path.exists(color_path):
        print('Color image', color_path, 'does not exist')
        return
    color_img = cv.imread(os.path.join(data_folder, filename + '_color.png'))

    depth_path = os.path.join(data_folder, filename + '_depth.png')
    if os.path.exists(depth_path):
        depth_img = Image.open(depth_path)
        depth_img = np.array(depth_img, dtype=float) / 1000

    # Discard depth information
    depth_img = None

    st_tme = time.time()
    racks, pipes = detector.detect(color_img, depth_img)
    print('Overall detection time (seconds):', time.time() - st_tme)

    print(racks)
    print(pipes)

    for rack in racks:
        img_pts = detector.obj_proj_to_img(rack.pos)
        draw_point(color_img, img_pts, (255, 0, 0))
        draw_text(color_img, img_pts, str(rack.color)[6], (255, 0, 0))

    for pipe in pipes:
        img_pts = detector.obj_proj_to_img(pipe.pos + [0, 0, .01])
        draw_point(color_img, img_pts, (0, 0, 255))
        draw_text(color_img, img_pts, str(pipe.color)[6], (0, 0, 255))

    cv.imshow('img', color_img)
    cv.waitKey()
    cv.destroyAllWindows()

    print()


if __name__ == '__main__':
    args = parse_args()
    data_folder = args.data

    if args.id:
        files = ['example' + args.id]
    elif args.name:
        files = [args.name]
    else:
        files = []
        for file in os.listdir(data_folder):
            if len(file) > 10 and file[-10:] == '_color.png':
                files.append(file[:-10])

    for filename in files:
        main(filename, data_folder)
