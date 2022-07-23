import cv2, numpy as np

def BEV(img0_path, img1_path, cam0_matrix, cam1_matrix):
    mtx_cam0, dist_cam0, H_cam0 = cam0_matrix
    mtx_cam1, dist_cam1, H_cam1 = cam1_matrix
    
    cam0_img = cv2.imread(img0_path, cv2.IMREAD_COLOR)
    und_cam0 = cv2.undistort(cam0_img, mtx_cam0, dist_cam0, None)
    dst_cam0 = cv2.warpPerspective(und_cam0, H_cam0, (540,540), None, cv2.INTER_CUBIC)  

    cam1_img = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    und_cam1 = cv2.undistort(cam1_img, mtx_cam1, dist_cam1, None)
    rot_mat = cv2.getRotationMatrix2D((320,240), 315,1)
    rot_cam1 = cv2.warpAffine(und_cam1,rot_mat,(640,480))
    dst_cam1 = cv2.warpPerspective(rot_cam1, H_cam1, (540,540), None, cv2.INTER_CUBIC)

    dst_cam0[250:,:] = 0
    dst_cam1[250:,:] = 0
    return dst_cam0, dst_cam1

def get_mask(img0, img1):
    img0_gray= cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img1_gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img0_gray)

    for y in range(img0_gray.shape[0]):
        for x in range(img0_gray.shape[1]):
            if img0_gray[y,x] and img1_gray[y,x]:
                mask[y,x] = 255
    
    return mask