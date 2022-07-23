import cv2, numpy as np, glob

src = cv2.imread("car.png", cv2.IMREAD_ANYCOLOR)
car = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(car, 0, 255, cv2.THRESH_BINARY)
mask[:33,:] = 0
mask[:,:215] = 0
mask[:,585:] = 0
mask[767:,:] = 0

resize_factor = 0.25
re_src = cv2.resize(src, (0, 0), None, resize_factor, resize_factor, cv2.INTER_AREA)
re_mask = cv2.resize(mask, (0, 0), None, resize_factor, resize_factor, cv2.INTER_AREA)
wh = re_src.shape[0]//2

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('svm.avi', fourcc, 30.0, (540, 540))

svm_path = sorted(glob.glob("img/*.png"))
for img in svm_path:
    svm = cv2.imread(img, cv2.IMREAD_ANYCOLOR)
    svm_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    padding = np.zeros_like(svm, np.uint8)
    padding_mask = np.zeros_like(svm_gray, np.uint8)

    padding[270-wh:270+wh, 270-wh:270+wh] = re_src 
    padding_mask[270-wh:270+wh, 270-wh:270+wh] = re_mask 
    svm[padding_mask > 0] = padding[padding_mask > 0]
    out.write(svm)
    # cv2.imshow("svm", svm); cv2.waitKey(0)

# out = cv2.VideoWriter("svm.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 10, (540, 540))
out.release()
