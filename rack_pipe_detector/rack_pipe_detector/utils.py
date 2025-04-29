import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List


def read_config(folder='data'):
    config_path = os.path.join(folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['folder'] = folder
    return config


def write_config(config, folder='data'):
    config_path = os.path.join(folder, 'config.json')
    if 'folder' in config:
        del config['folder']
    with open(config_path, 'w') as f:
        json.dump(config, f)


def get_intrinsic_mat(config_intr):
    mat = np.zeros((3, 3), dtype=float)
    mat[0, 0] = config_intr['fx']
    mat[0, 2] = config_intr['cx']
    mat[1, 1] = config_intr['fy']
    mat[1, 2] = config_intr['cy']
    mat[2, 2] = 1
    return mat


def arccos(x: float, degrees=True, eps=1e-5):
    if 1 < np.abs(x) < 1 + eps:
        x = np.sign(x)
    angle = np.arccos(x)
    if degrees:
        angle *= 180 / np.pi
    return angle


def calc_angle(vec1: np.ndarray, vec2: np.ndarray, degrees=True):
    cos_val = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return arccos(cos_val, degrees)


def calc_intersect_pt(line1, line2):
    # Write line1 as: x1 = ax1 * t1 + bx1 and y1 = ay1 * t1 + by1
    ax1, bx1 = line1[1][0] - line1[0][0], line1[0][0]
    ay1, by1 = line1[1][1] - line1[0][1], line1[0][1]
    # Write line2 as: x2 = ax2 * t2 + bx2 and y2 = ay2 * t2 + by2
    ax2, bx2 = line2[1][0] - line2[0][0], line2[0][0]
    ay2, by2 = line2[1][1] - line2[0][1], line2[0][1]

    # For x1 = x2, we have ax1 * t1 + bx1 = ax2 * t2 + bx2
    # Thus ax1 * t1 - ax2 * t2 = bx2 - bx1
    # The same, ay1 * t1 - ay2 * t2 = by2 - by1
    # We write them in form of At = b
    A = np.array([[ax1, -ax2], [ay1, -ay2]])
    b = np.array([[bx2 - bx1], [by2 - by1]])
    t = np.linalg.inv(A) @ b

    # Get their intersection point
    return line1[0] + t[0, 0] * (line1[1] - line1[0])


def calc_center_depth(depth_img, center):
    # Get only the part of depth image
    pix, piy = map(int, center)
    img_part = depth_img[piy:piy + 2, pix:pix + 2]

    # Get their weights
    # Difference between x and floor(x), y and floor(y)
    dxy = center - np.array(center, dtype=int)
    w_x = np.array([1 - dxy[0], dxy[0]])[np.newaxis, :]
    w_y = np.array([1 - dxy[1], dxy[1]])[:, np.newaxis]
    w_arr = w_x * w_y
    return np.sum(w_arr * img_part)


def calc_avg_rot(list_rot: List[Rotation], list_weight=None):
    # References: Markley, Landis & Cheng, Yang & Crassidis, John & Oshman, Yaakov. (2007). Averaging Quaternions.
    # Journal of Guidance, Control, and Dynamics. 30. 1193-1196. 10.2514/1.28949.
    if list_weight is None:
        list_weight = np.ones(len(list_rot), dtype=float) / len(list_rot)
    else:
        list_weight = np.array(list_weight)
    assert list_weight.shape == (len(list_rot),)

    mat_quat = np.array([rot.as_quat() for rot in list_rot])
    tensor_quat = np.einsum('ij,ik->ijk', mat_quat, mat_quat)
    mat_sum = np.einsum('i,ijk->jk', list_weight, tensor_quat)

    eigenvalues, eigenvectors = np.linalg.eig(mat_sum)
    avg_quat = eigenvectors[:, np.argmax(np.real(eigenvalues))]
    assert np.sum(np.abs(np.imag(avg_quat))) < 4e-5
    avg_quat = np.real(avg_quat)
    return Rotation.from_quat(avg_quat)
