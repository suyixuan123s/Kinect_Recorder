import os
import numpy as np
import open3d as o3d
import cv2 as cv
from scipy.spatial.transform import Rotation

color = 'BLUE'

mesh = o3d.io.read_triangle_mesh(os.path.join('..', 'data', 'rack_%s.STL' % color))

bounding_box = mesh.get_axis_aligned_bounding_box()

surface_box_min = bounding_box.get_min_bound()
surface_box_max = bounding_box.get_max_bound()
surface_box_min[1] = surface_box_max[1] - 1
surface_box = o3d.geometry.AxisAlignedBoundingBox(surface_box_min, surface_box_max)
surface_mesh = mesh.crop(surface_box)
surface_mesh = surface_mesh.translate(-surface_mesh.get_axis_aligned_bounding_box().get_center())
surface_mesh = surface_mesh.rotate(Rotation.from_euler('x', 90, degrees=True).as_matrix())

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(surface_mesh)
ctr = vis.get_view_control()
cam_param = ctr.convert_to_pinhole_camera_parameters()

img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
vis.destroy_window()

img = np.mean(img, axis=2)
img = np.array(img < 1., dtype=np.uint8) * 255

img = cv.blur(img, (5, 5))
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, img.shape[0] / 20, param2=50, minRadius=20, maxRadius=100)
circles = circles.reshape(-1, 3)
print(circles)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for circle in circles:
    circle = np.uint16(np.round(circle))
    cv.circle(img, circle[:2], circle[2], (0, 0, 255), 2)


def transform_from_image_to_world(img_coor, depth):
    coor = np.concatenate((img_coor, [1.]))
    coor *= depth
    coor = np.linalg.inv(cam_param.intrinsic.intrinsic_matrix) @ coor
    coor = np.concatenate((coor, [1.]))
    world_coor = np.linalg.inv(cam_param.extrinsic) @ coor
    return world_coor[:3]


depth = cam_param.extrinsic[2, 3]

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(surface_mesh)
for circle in circles:
    center, radius = circle[:2], circle[2]

    pt1 = transform_from_image_to_world(center, depth)
    pt2 = transform_from_image_to_world(center + [radius, 0], depth)
    hole_center = pt1
    hole_radius = np.linalg.norm(pt2 - pt1)
    print(hole_center, hole_radius)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(hole_radius, 0.5)
    cylinder = cylinder.translate(hole_center)
    cylinder.paint_uniform_color([1, 0, 0])
    vis.add_geometry(cylinder)

vis.run()
vis.capture_screen_image('output/rack_%s_circles.png' % color)
vis.destroy_window()