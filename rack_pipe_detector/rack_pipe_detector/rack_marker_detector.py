import numpy as np

from apriltag_detector import AprilTagDetector, PnPTransform
from utils import get_intrinsic_mat, calc_intersect_pt, calc_center_depth, calc_angle


class RackMarkerDetection:
    def __init__(self, tag_id: int, trans: PnPTransform):
        # all errors for re-projection errors of all PnP solutions
        self.tag_id = tag_id
        self.trans = trans

    def __repr__(self) -> str:
        return 'RackMarkerDetection id {0} transform {1}'.format(self.tag_id, repr(self.trans))


class RackMarkerDetector:
    def __init__(self, config):
        self.config = config
        self.marker_size = config['marker_size']['small']
        self.intr_mat = get_intrinsic_mat(config['intrinsic'])
        self.distortion = np.array(config['intrinsic']['distortion'])
        self.extr_mat = np.array(config['extrinsic'])
        self.extr_mat = np.concatenate((self.extr_mat, np.eye(4)[[3]]))

        self.apriltag_detector = AprilTagDetector(self.marker_size, self.intr_mat, self.distortion)

    def process(self, color_img: np.ndarray, depth_img: np.ndarray | None = None):
        detections = self.apriltag_detector.detect(color_img)
        results = []
        for detection in detections:
            tag_id, transforms = detection.tag_id, detection.list_trans
            if not transforms:
                continue

            if depth_img is None:
                depth = None
            else:
                img_pts = detection.img_pts
                center = calc_intersect_pt(img_pts[[0, 2]], img_pts[[1, 3]])
                depth = calc_center_depth(depth_img, center)

            tup = (np.infty, None)
            for trans in transforms:
                if depth is not None:
                    trans.pos[2] = depth
                new_trans = trans.get_trans_to_base(self.extr_mat)
                angle_z = calc_angle(new_trans.to_matrix()[:3, 2], np.array([0, 0, 1]))
                new_tup = (min(angle_z, abs(angle_z - 90)), new_trans)
                if new_tup < tup:
                    tup = new_tup

            trans = tup[1]
            assert isinstance(trans, PnPTransform)
            results.append(RackMarkerDetection(tag_id, trans))

        return results
