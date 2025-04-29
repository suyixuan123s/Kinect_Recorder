import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

arr1 = a[0][:, np.newaxis] @ a[0][np.newaxis, :]
arr2 = a[1][:, np.newaxis] @ a[1][np.newaxis, :]
res = np.einsum('ij,ik->ijk', a, a)
print(res.shape)
print(res - np.array([arr1, arr2]))