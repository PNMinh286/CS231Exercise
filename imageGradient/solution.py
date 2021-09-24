import numpy as np
import cv2

img_rgb = cv2.imread("resources/img.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

Fx = cv2.resize(img_gray[1:, :] - img_gray[:-1,:], (img_gray.shape[1], img_gray.shape[0]))
Fy = cv2.resize(img_gray[:, 1:] - img_gray[:, :-1], (img_gray.shape[1], img_gray.shape[0]))
G = np.array(np.sqrt(np.square(abs(Fx))) + np.sqrt(abs(Fy)), dtype=np.uint8)

cv2.imshow("Anh sau khi lay image gradient", G)
cv2.waitKey(0)
print("Ma tran cua anh sau khi lay image gradient\n",G)