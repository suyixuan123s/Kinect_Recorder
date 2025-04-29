import os
import cv2 as cv
import numpy as np
import open3d as o3d

from rack_pipe_detector.utils import read_config, get_intrinsic_mat
from test_utils import get_default_data_folder


def main(filename, data_folder=get_default_data_folder(), output_folder='output'):
    config = read_config(data_folder)
    intr_mat = get_intrinsic_mat(config['intrinsic'])
    distortion = np.array(config['intrinsic']['distortion'])
    extr_mat = np.array(config['extrinsic'])
    extr_mat = np.concatenate((extr_mat, np.eye(4)[[3]]))

    cam_intr = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080,
                                                 fx=intr_mat[0, 0],
                                                 fy=intr_mat[1, 1],
                                                 cx=intr_mat[0, 2],
                                                 cy=intr_mat[0, 2])

    rgb_file = os.path.join(data_folder, '%s_color.png' % filename)
    dep_file = os.path.join(data_folder, '%s_depth.png' % filename)

    rgb_img = cv.imread(rgb_file)
    # rgb_img = cv.undistort(rgb_img, intr_mat, distortion)
    rgb_img[..., [0, 2]] = rgb_img[..., [2, 0]]

    rgb_img = o3d.geometry.Image(rgb_img)
    dep_img = o3d.io.read_image(dep_file)

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, dep_img,
                                                                  convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, intrinsic=cam_intr, extrinsic=extr_mat)

    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    example_ids = [1, 2, 3, 4, 5]
    example_ids = [5]
    for i in example_ids:
        main('example%d' % i)
