import os
import json
import numpy as np
import cv2 as cv
import pyrealsense2 as rs


class L515_dirver:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Get color profile
        p_color = profile.get_stream(rs.stream.color)
        self.intr_color = p_color.as_video_stream_profile().get_intrinsics()
        print('intrinsic parameters:', self.intr_color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        depth_sensor.set_option(rs.option.visual_preset, 3)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def grab(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale * 1000
        depth_image = depth_image.astype('uint16')
        color_image = np.asanyarray(color_frame.get_data())
        color_image[:, :, [0, 2]] = color_image[:, :, [2, 0]]

        return color_image, depth_image


def update_intrinsic(intr, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['intrinsic'] = {
        'fx': intr.fx,
        'fy': intr.fy,
        'cx': intr.ppx,
        'cy': intr.ppy,
        'distortion': intr.coeffs
    }

    with open(config_path, 'w') as f:
        json.dump(config, f)


def get_image_id(dirname):
    i = 1
    while os.path.exists(os.path.join(dirname, 'example%d_color.png' % i)):
        i += 1
    return i


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    driver = L515_dirver()

    # Add/update intrinsic parameters into config.json
    config_path = os.path.join(folder, 'config.json')
    update_intrinsic(driver.intr_color, config_path)

    # Get save image path
    i = get_image_id(folder)
    color_path = os.path.join(folder, 'example%d_color.png' % i)
    depth_path = os.path.join(folder, 'example%d_depth.png' % i)

    # Grab and save images
    color_img, depth_img = driver.grab()
    print('Saving to', color_path, 'and', depth_path)
    cv.imwrite(color_path, color_img)
    cv.imwrite(depth_path, depth_img)
