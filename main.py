import cv2, numpy as np, glob
from module import *

### Camera matrix, Homography - CAM0
mtx_cam0 = np.array([[351.169058, 0.000000, 330.784880],
                     [0.000000, 351.657223, 239.471923],
                     [0.000000, 0.000000, 1.000000]])

dist_cam0 = np.array([-0.325330, 0.091562, 0.000301, 0.000526, 0.000000])

H_cam0 = np.array([[-1.74197765e-01, -1.15197305e+00,  3.18445397e+02],
                   [-4.14922723e-02, -1.10185034e+00,  3.09749033e+02],
                   [-1.69361013e-04, -4.14349013e-03,  1.00000000e+00]])
cam0_matrix = [mtx_cam0, dist_cam0, H_cam0]

### camera matrix, Homography - CAM1
mtx_cam1 = np.array([[522.047976, 0.000000, 312.409819],
                     [0.000000, 522.667306, 263.691174],
                     [0.000000, 0.000000, 1.000000]])

dist_cam1 = np.array([0.052595, -0.175610, 0.002809, -0.001501, 0.000000])
H_cam1 = np.array([[-1.61928347e+00,  1.74440705e+00,  3.04707228e+02],
                   [-1.44258333e+00 , 1.53464376e+00, 8.23741751e+01],
                   [-6.20092519e-03,  5.45304794e-03,  1.00000000e+00]])
cam1_matrix = [mtx_cam1, dist_cam1, H_cam1]


# Image Read
path_cam0 = "cam0"
path_cam1 = "cam1"
cam0_list = sorted(glob.glob("cam0/*.png"))
cam1_list = sorted(glob.glob("cam1/*.png"))

# Mask
img0, img1 = BEV(cam0_list[0], cam1_list[0], cam0_matrix, cam1_matrix)
mask = get_mask(img0, img1)

for cam0, cam1 in zip(cam0_list, cam1_list):
    dst_cam0, dst_cam1 = BEV(cam0, cam1, cam0_matrix, cam1_matrix)

    dst_cam0 = cv2.copyTo(dst_cam0, ~mask)
    dst_all = dst_cam0+dst_cam1
    cv2.imshow("dst", dst_all); cv2.waitKey(0)
