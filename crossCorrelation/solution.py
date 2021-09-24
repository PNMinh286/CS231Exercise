import numpy as np
import cv2

img_rgb = cv2.imread("resources/img.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
norm_1 = np.linalg.norm(img_gray)
norm_img_gray = img_gray/norm_1
ker = cv2.imread("resources/ker.jpg",0)
norm_2 = np.linalg.norm(ker)
norm_ker = ker/norm_2

def corr2d(X, K):
    h = K.shape[0]
    w = K.shape[1]
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    Y2 = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = ((X[i: i + h, j: j + w] * K).sum())
            Y2[i, j] = pow((pow(X[i: i + h, j: j + w], 2).sum() * pow(K, 2).sum()), 0.5)
            Y[i, j] = Y[i, j]/Y2[i, j]
    return Y

out_img = corr2d(norm_img_gray,norm_ker)
threshold = 0.995
cnt = 0
loc = np.where( out_img >= threshold)
w, h = ker.shape[::-1]
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    cnt = cnt+1
cv2.imshow('Danh dau cac mau doi tuong',img_rgb)
cv2.waitKey(0)
print('So luong mau doi tuong:', cnt)