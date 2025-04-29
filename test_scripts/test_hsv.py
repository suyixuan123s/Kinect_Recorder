import numpy as np
import cv2 as cv

img = np.zeros((180, 256, 3), dtype=np.uint8)
img[:, :, 0] = np.arange(img.shape[0])[:, np.newaxis]
img[:, :, 1] = np.arange(img.shape[1])[np.newaxis, :]
img[:, :, 2] = 255

img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
# cv.line(img, (0, 80), (255, 80), (0, 0, 255), 1)
# cv.line(img, (0, 100), (255, 100), (0, 0, 255), 1)

cv.imshow('hsv_img', img)
cv.waitKey()
cv.imwrite('output/hsv_color_space.png', img)