import time
from typing import Union
import cv2 as cv
import numpy as np
from pyapriltags import Detector
from scipy.spatial.transform import Rotation


class PnPTransform:
    def __init__(self, pos: np.ndarray, orn: Union[np.ndarray, Rotation], error):
        self.pos = pos.flatten()
        if isinstance(orn, Rotation):
            self.orn = orn
        else:
            self.orn = Rotation.from_rotvec(orn.flatten())
        if isinstance(error, np.ndarray):
            self.error = error.item()
        else:
            self.error = error

    def to_matrix(self) -> np.ndarray:
        mat = np.zeros((4, 4), dtype=float)
        mat[:3, :3] = self.orn.as_matrix()
        mat[:3, 3] = self.pos
        mat[3, 3] = 1
        return mat

    def inverse_yz(self):
        self.orn = self.orn * Rotation.from_euler('x', 180, degrees=True)

    def inverse_xy(self):
        self.orn = self.orn * Rotation.from_euler('z', 180, degrees=True)

    def get_trans_to_base(self, extr_mat: np.ndarray):
        new_mat = np.linalg.inv(extr_mat) @ self.to_matrix()
        return PnPTransform.from_matrix(new_mat, self.error)

    def __repr__(self) -> str:
        return 'pos {0} orn (zyx) {1} error {2}'.format(str(self.pos), str(self.orn.as_euler('zyx', degrees=True)),
                                                        str(self.error))

    @staticmethod
    def from_matrix(mat: np.ndarray, error):
        assert mat.shape == (4, 4) and mat[3, 3] == 1
        orn = Rotation.from_matrix(mat[:3, :3])
        pos = mat[:3, 3]
        return PnPTransform(pos, orn, error)


class AprilTagDetection:
    def __init__(self, tag_id, img_pts, obj_pts=None, intr_mat=None, distortion=None, err_thres=0.1):
        self.tag_id = tag_id
        self.img_pts = img_pts.copy()

        self.list_trans = []
        if obj_pts is not None and intr_mat is not None and distortion is not None:
            self.calc_pnp(obj_pts, intr_mat, distortion, err_thres)

    def calc_pnp(self, obj_pts, intr_mat, distortion, err_thres=0.1):
        # _, rvecs, tvecs, errors = cv.solvePnPGeneric(obj_pts, self.img_pts, intr_mat, distortion)
        # Have to use IPPE Square for better estimations
        _, rvecs, tvecs, errors = cv.solvePnPGeneric(obj_pts, self.img_pts, intr_mat, distortion,
                                                     flags=cv.SOLVEPNP_IPPE_SQUARE)

        self.list_trans = []
        for orn, pos, error in zip(rvecs, tvecs, errors):
            if error > err_thres: continue
            trans = PnPTransform(pos, orn, error)
            trans.inverse_yz()
            self.list_trans.append(trans)

    def draw_detection(self, img, scale=1, offset=(0, 0)):
        # Transform image points
        pts = self.img_pts.copy()
        pts -= offset
        pts *= scale

        center = np.array(np.mean(pts, axis=0), dtype=int)
        cv.putText(img, str(self.tag_id), center, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        pts = np.array(pts, dtype=int)

        for i in range(4):
            cv.line(img, pts[i], pts[(i + 1) % 4], (0, 0, 255), 1)


class AprilTagDetector:
    def __init__(self, marker_size, intr_mat, distortion,
                 reproj_err_thres=0.1, tag_families='tag16h5', decode_sharpening=0.52):
        self.intr_mat = intr_mat.copy()
        self.distortion = distortion.copy()
        self.reproj_err_thres = reproj_err_thres

        self.detector = Detector(searchpath=['apriltags'],
                                 families=tag_families,
                                 nthreads=1,
                                 quad_decimate=1,
                                 quad_sigma=0,
                                 decode_sharpening=decode_sharpening,
                                 debug=0)

        # Create object points for a square on the surface
        self.obj_pts = np.zeros((4, 3), dtype=float)
        self.obj_pts[:2, 1] = 1
        self.obj_pts[1:3, 0] = 1
        self.obj_pts[:, :2] -= 0.5
        self.obj_pts *= marker_size

    def detect(self, img, roi=None):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if roi is not None:
            # Apply masking
            si, ei, sj, ej = roi
            mask = np.zeros(gray_img.shape, dtype=gray_img.dtype)
            mask[si:ei, sj:ej] = 1
            gray_img = gray_img * mask

        # Detect
        st_tme = time.time()
        detections = self.detector.detect(gray_img)
        print('AprilTag detection time (seconds):', time.time() - st_tme)
        results = []
        for detection in detections:
            results.append(
                AprilTagDetection(detection.tag_id, detection.corners, self.obj_pts, self.intr_mat, self.distortion,
                                  self.reproj_err_thres))
        return results
