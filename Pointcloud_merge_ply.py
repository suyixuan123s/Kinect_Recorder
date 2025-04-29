import numpy as np
import datetime
import math
import csv
import open3d as o3d
import copy


class pointcloud_merging:
    def __init__(self):
        # self.filename = 'AzureKinect_1200/CapturePose_Tool0_Base0.csv'
        self.filename = '3D Reconstruction/CalibrationData.csv'
        # self.tool2c_rt = np.array([[-0.00214674, 0.999991, 0.00354434, -0.0762475],
        #                            [-0.999848, -0.00208502, -0.017326, -0.0353502],
        #                            [-0.0173185, -0.00358099, 0.999844, 0.141391],
        #                            [0.0, 0.0, 0.0, 1.0]])
        # self.tool2c_rt = np.array([[-0.008334505, 1.000008401, 0.016834489, -0.07957697121],
        #                            [-1.000143425, -0.008271811, -0.003791021, -0.02191276866],
        #                            [-0.003651127, -0.016865383, 1.000035946, 0.1429821046],
        #                            [0.0, 0.0, 0.0, 1.0]])
        # self.tool2c_rt = np.array([[-0.0123, 0.999823, 0.009527, -0.0761881],
        #                            [-0.99986, -0.01234, -0.003916, -0.0317218],
        #                            [0.004033, -0.00948, 0.999891, 0.112199],
        #                            [0.0, 0.0, 0.0, 1.0]])
        self.tool2c_rt = np.array([[-0.010502, 0.999823, 0.012148, -0.0780857*1000],
                                   [-0.99985, -0.01056,0.005337, -0.031727699*1000],
                                   [0.0054654, -0.012091, 0.9998, 0.11451644*1000],
                                   [0.0, 0.0, 0.0, 1.0]])
        self.tool2c_r = np.array([[-0.00214674, 0.999991, 0.00354434],
                                  [-0.999848, -0.00208502, -0.017326],
                                  [-0.0173185, -0.00358099, 0.999844]])
        self.tool2c_t = np.array([[-0.0762475],
                                  [-0.0353502],
                                  [0.141391]])
        self.c2tool_r, self.c2tool_t = self.invert_rt(self.tool2c_r, self.tool2c_t)
        self.c2tool_rt = self.form_rt(self.c2tool_r, self.c2tool_t)
        # print(self.c2tool_rt)
        self.test_trans = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0.0, 0.0, 0.0, 1.0]])
        self.test_trans_r = np.array([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]
                                      ])

        self.test_trans_t = np.array([[0],
                                      [0],
                                      [1]])
        self.start_frame = 1
        self.end_frame = 2
        self.voxel_size = 0.1  # m
        self.output_convex_hull = False
        self.output_RT_matrix = True
        if self.output_RT_matrix:
            self.fw = open("RT_matrix.txt", "w")

    def form_rt(self, r_input, t_input):
        rt = np.hstack((r_input, t_input))
        rt = np.vstack((rt, np.array([[0, 0, 0, 1]])))
        return rt

    def invert_rt(self, r_input, t_input):
        inv_r = np.linalg.inv(r_input)
        inv_r = r_input.transpose()
        inv_t = -np.dot(inv_r, t_input)
        # inv_t = -t_input
        # print(np.dot(r_input.transpose(), r_input))
        # print(inv_t)
        return inv_r, inv_t

    def angle2radian(self, x):
        rx = float(x) / 180 * math.pi
        return rx

    def run(self):
        vis = o3d.visualization.Visualizer()
        with open(self.filename, 'rt', newline='', encoding='utf-8', errors='ignore') as f:
            f_csv = csv.reader(f)
            i = 1
            for lines in f_csv:
                # x, y, z, rx, ry, rz = lines
                # depth_file = '2022-09-23-13-39-05/depth/' + str(i - 1).zfill(5) + '.png'
                # color_file = '2022-09-23-13-39-05/color/' + str(i - 1).zfill(5) + '.jpg'
                pcd_file = '3D Reconstruction/PointCloud/PointCloud_'+ str(i).zfill(4) +'.ply'
                intrinsic = o3d.io.read_pinhole_camera_intrinsic('intrinsic.json')
                # print(ply_file)
                # print(lines)
                if i == self.start_frame:
                    # depth_image = o3d.io.read_image(depth_file)
                    # color_image = o3d.io.read_image(color_file)
                    pcd = o3d.io.read_point_cloud(pcd_file)
                    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                    #                                                           depth_image,
                    #                                                           convert_rgb_to_intensity=False)
                    # image_16 = np.asarray(image, dtype='uint16')
                    # source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                    source = pcd
                    # o3d.visualization.draw_geometries([source])
                    # source.voxel_down_sample(self.voxel_size)
                    s_x, s_y, s_z, s_rw, s_rx, s_ry, s_rz = lines
                    # S_R, S_T = self.Euler_to_Matrix(s_x, s_y, s_z, s_rx, s_ry, s_rz)
                    # S_R = o3d.geometry.PointCloud.get_rotation_matrix_from_zyx((self.angle2radian(s_rz),
                    #                                                             self.angle2radian(s_ry),
                    #                                                             self.angle2radian(s_rx)))
                    S_R = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion((s_rw, s_rx, s_ry, s_rz))
                    S_T = np.array([[float(s_x.strip("\ufeff")) / 1], [float(s_y) / 1], [float(s_z) / 1]])
                    # S_R, S_T = self.invert_rt(S_R, S_T)
                    # S_T = np.array([[0],
                    #                 [0],
                    #                 [0]])
                    s2base_rt = self.form_rt(S_R, S_T)
                    source_t = copy.deepcopy(source)
                    # vis.create_window('recorder', 1280, 720)
                    # vis.add_geometry(source)
                    if self.output_RT_matrix:
                        trans = np.dot(s2base_rt, self.tool2c_rt)
                        print(trans)
                        #print(np.resize(trans, [1, 16]))
                        k = str(np.resize(trans, [1, 16]))
                        k = k.replace('\n', '')
                        k = k.replace('[', '')
                        k = k.replace(']', '')
                        self.fw.writelines(k + '\n')
                    source_t.transform(self.tool2c_rt)
                    source_t.transform(s2base_rt)
                    #source_t.transform(trans)
                    # print(source_t.get_center())

                elif i > self.start_frame:
                    timer = datetime.datetime.now().timestamp()
                    # depth_image = o3d.io.read_image(depth_file)
                    # color_image = o3d.io.read_image(color_file)
                    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                    #                                                           depth_image,
                    #                                                           convert_rgb_to_intensity=False)
                    # image_16 = np.asarray(image, dtype='uint16')
                    # image_16 = np.array(image, dtype='uint16')
                    pcd = o3d.io.read_point_cloud(pcd_file)
                    # target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                    target = pcd
                    # print('file reading time:')
                    # print(datetime.datetime.now().timestamp() - timer)
                    # target.voxel_down_sample(self.voxel_size)
                    t_x, t_y, t_z, t_rw, t_rx, t_ry, t_rz = lines
                    # T_R = o3d.geometry.PointCloud.get_rotation_matrix_from_zyx((self.angle2radian(t_rz),
                    #                                                             self.angle2radian(t_ry),
                    #                                                             self.angle2radian(t_rx)))
                    T_R = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion((t_rw, t_rx, t_ry, t_rz))
                    T_T = np.array([[float(t_x) / 1], [float(t_y) / 1], [float(t_z) / 1]])
                    t2base_rt = self.form_rt(T_R, T_T)
                    target_t = copy.deepcopy(target)
                    trans = np.dot(t2base_rt, self.tool2c_rt)
                    if self.output_RT_matrix:
                        print(trans)
                        # print(np.resize(trans, [1, 16]))
                        k = str(np.resize(trans, [1, 16]))
                        k = k.replace('\n', '')
                        k = k.replace('[', '')
                        k = k.replace(']', '')
                        self.fw.writelines(k + '\n')
                    target_t.transform(self.tool2c_rt)
                    target_t.transform(t2base_rt)
                    #target_t.transform(trans)
                    timer = datetime.datetime.now().timestamp()
                    source_t = source_t + target_t
                    # vis.update_geometry(source_t)
                    # vis.poll_events()
                    # vis.update_renderer()
                    # print('pointcloud merging time:')
                    # print(datetime.datetime.now().timestamp() - timer)
                    # source_t.remove_duplicated_points()
                    # source_t.uniform_down_sample(2)
                    source_t.voxel_down_sample(self.voxel_size)
                    # o3d.visualization.draw_geometries([source_t])

                if i > self.end_frame - 1:
                    break

                i = i + 1
            if self.output_RT_matrix:
                self.fw.close()
            o3d.visualization.draw_geometries([source_t])
            o3d.io.write_point_cloud('multi_pose.ply',source_t)
            # o3d.visualization.draw_geometries([source + target])
            if self.output_convex_hull:
                timer = datetime.datetime.now().timestamp()
                convex, list = source_t.compute_convex_hull(True)
                # print(datetime.datetime.now().timestamp() - timer)
                # print(list)
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex)
                hull_ls.paint_uniform_color((1, 0, 0))
                # o3d.visualization.draw_geometries([source_t, hull_ls])


if __name__ == '__main__':
    r = pointcloud_merging()
    r.run()
