import os
import cv2 as cv
import numpy as np
from PIL import Image

from rack_pipe_detector.rack_detector import RackDetector
from rack_pipe_detector.utils import read_config
from test_utils import get_default_data_folder

filename = 'example5'


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    detector = RackDetector(config)

    color_img = cv.imread(os.path.join(data_folder, filename + '_color.png'))
    depth_img = Image.open(os.path.join(data_folder, filename + '_depth.png'))
    depth_img = np.array(depth_img, dtype=float) / 1000

    detections = detector.process(color_img, depth_img)
    print('All detections:')
    for detect in detections:
        print(detect)


if __name__ == '__main__':
    main(filename)
