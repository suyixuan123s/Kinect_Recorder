import numpy as np
from scipy.spatial.transform import Rotation

rot = Rotation.from_euler('zyx', np.random.random(3))
print(rot.as_matrix())

rot1 = Rotation.from_euler('Z', 90, degrees=True) * rot
print(rot1.as_matrix())

rot2 = Rotation.from_euler('z', 90, degrees=True) * rot
print(rot2.as_matrix())

rot3 = rot * Rotation.from_euler('Z', 90, degrees=True)
print(rot3.as_matrix())

rot4 = rot * Rotation.from_euler('z', 90, degrees=True)
print(rot4.as_matrix())
