import cv2
import imutils
import numpy as np
import imutils

# # (shift_width, shift_height): how far away the birdview looks outside
# # of the calibration pattern in horizontal and vertical directions
# shift_w = 300
# shift_h = 300
#
# # size of the gap between the calibration pattern and the car
# # in horizontal and vertical directions
# inn_shift_w = 20
# inn_shift_h = 50
#
# # total width/height of the stitched image
# total_w = 600 + 2 * shift_w
# total_h = 1000 + 2 * shift_h
#
# # four corners of the rectangular region occupied by the car
# # top-left (x_left, y_top), bottom-right (x_right, y_bottom)
# xl = shift_w + 180 + inn_shift_w
# xr = total_w - xl
# yt = shift_h + 200 + inn_shift_h
# yb = total_h - yt
#
#
# def FI(front_image):
#     return front_image[:, :xl]
#
#
# def FII(front_image):
#     return front_image[:, xr:]
#
#
# def FM(front_image):
#     return front_image[:, xl:xr]
#
#
# def BIII(back_image):
#     return back_image[:, :xl]
#
#
# def BIV(back_image):
#     return back_image[:, xr:]
#
#
# def BM(back_image):
#     return back_image[:, xl:xr]
#
#
# def LI(left_image):
#     return left_image[:yt, :]
#
#
# def LIII(left_image):
#     return left_image[yb:, :]
#
#
# def LM(left_image):
#     return left_image[yt:yb, :]
#
#
# def RII(right_image):
#     return right_image[:yt, :]
#
#
# def RIV(right_image):
#     return right_image[yb:, :]
#
#
# def RM(right_image):
#     return right_image[yt:yb, :]
#
#
# def adjust_luminance(gray, factor):
#     """
#     Adjust the luminance of a grayscale image by a factor.
#     """
#     return np.minimum((gray * factor), 255).astype(np.uint8)
#
#
# def get_mean_statistisc(gray, mask):
#     """
#     Get the total values of a gray image in a region defined by a mask matrix.
#     The mask matrix must have values either 0 or 1.
#     """
#     return np.sum(gray * mask)
#
#
# def mean_luminance_ratio(grayA, grayB, mask):
#     return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)
#
#
# weights = None
# masks = None
# frames = None
#
# def make_luminance_balance(frame):
#     def tune(x):
#         if x >= 1:
#             return x * np.exp((1 - x) * 0.5)
#         else:
#             return x * np.exp((1 - x) * 0.8)
#
#     front, back, left, right = (frame, frame, frame, frame)
#     m1, m2, m3, m4 = masks
#     Fb, Fg, Fr = cv2.split(front)
#     Bb, Bg, Br = cv2.split(back)
#     Lb, Lg, Lr = cv2.split(left)
#     Rb, Rg, Rr = cv2.split(right)
#
#     a1 = mean_luminance_ratio(RII(Rb), FII(Fb), m2)
#     a2 = mean_luminance_ratio(RII(Rg), FII(Fg), m2)
#     a3 = mean_luminance_ratio(RII(Rr), FII(Fr), m2)
#
#     b1 = mean_luminance_ratio(BIV(Bb), RIV(Rb), m4)
#     b2 = mean_luminance_ratio(BIV(Bg), RIV(Rg), m4)
#     b3 = mean_luminance_ratio(BIV(Br), RIV(Rr), m4)
#
#     c1 = mean_luminance_ratio(LIII(Lb), BIII(Bb), m3)
#     c2 = mean_luminance_ratio(LIII(Lg), BIII(Bg), m3)
#     c3 = mean_luminance_ratio(LIII(Lr), BIII(Br), m3)
#
#     d1 = mean_luminance_ratio(FI(Fb), LI(Lb), m1)
#     d2 = mean_luminance_ratio(FI(Fg), LI(Lg), m1)
#     d3 = mean_luminance_ratio(FI(Fr), LI(Lr), m1)
#
#     t1 = (a1 * b1 * c1 * d1) ** 0.25
#     t2 = (a2 * b2 * c2 * d2) ** 0.25
#     t3 = (a3 * b3 * c3 * d3) ** 0.25
#
#     x1 = t1 / (d1 / a1) ** 0.5
#     x2 = t2 / (d2 / a2) ** 0.5
#     x3 = t3 / (d3 / a3) ** 0.5
#
#     x1 = tune(x1)
#     x2 = tune(x2)
#     x3 = tune(x3)
#
#     Fb = adjust_luminance(Fb, x1)
#     Fg = adjust_luminance(Fg, x2)
#     Fr = adjust_luminance(Fr, x3)
#
#     y1 = t1 / (b1 / c1) ** 0.5
#     y2 = t2 / (b2 / c2) ** 0.5
#     y3 = t3 / (b3 / c3) ** 0.5
#
#     y1 = tune(y1)
#     y2 = tune(y2)
#     y3 = tune(y3)
#
#     Bb = adjust_luminance(Bb, y1)
#     Bg = adjust_luminance(Bg, y2)
#     Br = adjust_luminance(Br, y3)
#
#     z1 = t1 / (c1 / d1) ** 0.5
#     z2 = t2 / (c2 / d2) ** 0.5
#     z3 = t3 / (c3 / d3) ** 0.5
#
#     z1 = tune(z1)
#     z2 = tune(z2)
#     z3 = tune(z3)
#
#     Lb = adjust_luminance(Lb, z1)
#     Lg = adjust_luminance(Lg, z2)
#     Lr = adjust_luminance(Lr, z3)
#
#     w1 = t1 / (a1 / b1) ** 0.5
#     w2 = t2 / (a2 / b2) ** 0.5
#     w3 = t3 / (a3 / b3) ** 0.5
#
#     w1 = tune(w1)
#     w2 = tune(w2)
#     w3 = tune(w3)
#
#     Rb = adjust_luminance(Rb, w1)
#     Rg = adjust_luminance(Rg, w2)
#     Rr = adjust_luminance(Rr, w3)
#
#     frames = [cv2.merge((Fb, Fg, Fr)),
#               cv2.merge((Bb, Bg, Br)),
#               cv2.merge((Lb, Lg, Lr)),
#               cv2.merge((Rb, Rg, Rr))]
#     return frames
#

h = 720
w = 1280
# SIZE = (7, 7)
SIZE = (4, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 90, 0.001)

# camMatrix = np.load("/home/msi-user/Рабочий стол/test/pyqt5/programm/arrays/homography/10/matrixCamera10.npy")
# camDist = np.load("/home/msi-user/Рабочий стол/test/pyqt5/programm/arrays/homography/10/distCamera10.npy")

# camMatrix = np.load("/home/msi-user/Рабочий стол/test/pyqt5/programm/arrays/homography/12(333)/matrixCamera12.npy")
# camDist = np.load("/home/msi-user/Рабочий стол/test/pyqt5/programm/arrays/homography/12(333)/distCamera12.npy")

# filename = '11(board_center)_cleanup.jpg'
filename = '/home/msi-user/Рабочий стол/IntrinsicCalibration/All_images/CAM_2_13.12.22/0.jpg'

# name = "allImages/img2/img_dst1"
name = "nAsA"
# map1x, map1y = cv2.initUndistortRectifyMap(camMatrix, camDist, None, camMatrix, (w, h),
#                                            cv2.CV_16SC2)
img = cv2.imread(filename)

# img = cv2.remap(img, map1x, map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
# img = imutils.rotate(img, angle=1)
#
# cv2.namedWindow("remap", cv2.WINDOW_NORMAL)
# cv2.imshow("remap", img)
# cv2.imwrite(f"{name}", img)
# img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
#for i in make_luminance_balance(f)
#img = make_luminance_balance(img)
# cv2.imshow("make_white_balance", img)
# cv2.waitKey(0)
print(img)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
thresh1 = 150


def imgThreshold(val):
    global thresh1
    ret, thresh = cv2.threshold(imgray, val, 255, 0)
    thresh1 = thresh
    cv2.imshow("thresh", thresh)


cv2.createTrackbar('Threshold: ', "thresh", thresh1, 255 - 1, imgThreshold)
imgThreshold(150)

cv2.waitKey(0)

cnts = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        # screenCnt = approx
        break
print(approx)
cv2.polylines(img, [approx], True, (255, 0, 100), 4)
cv2.namedWindow("polylines", cv2.WINDOW_NORMAL)
cv2.imshow("polylines", img)
cv2.waitKey(0)
fill_color = [0, 0, 0]

stencil = np.zeros(img.shape[:-1]).astype(np.uint8)
cv2.fillPoly(stencil, [approx], 255)

sel = stencil != 255
img[sel] = fill_color

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img)
#cv2.imwrite("stitch/all/img/result_12(1).jpg", img)
# ------------------------------------------------------------------------#
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.imshow("gray", gray2)

cv2.namedWindow("threshed_gray", cv2.WINDOW_NORMAL)
thresh2 = 150


def imgThreshold2(val):
    global thresh2
    ret2, thresh22 = cv2.threshold(gray2, val, 255, 0)
    thresh2 = thresh22
    cv2.imshow("threshed_gray", thresh22)


cv2.createTrackbar('Threshold2: ', "threshed_gray", thresh2, 255 - 1, imgThreshold2)
imgThreshold2(thresh2)

cv2.waitKey(0)

kernel2 = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel2)

cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
cv2.imshow("closing", closing)

convertToColor = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

#cv2.imwrite(f"{name}.jpg", closing)

# ret, corners = cv2.findChessboardCorners(closing, SIZE,
#                                          cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
# objp = np.zeros((24 * 17, 3), np.float32)
# objp[:, :2] = np.mgrid[0:24, 0:17].T.reshape(-1, 2)
# axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
# axisBoxes = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
#                         [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
#
# if ret:
#     corners2 = cv2.cornerSubPix(closing, corners, (11, 11), (-1, -1), criteria)
#
#     new_img = cv2.drawChessboardCorners(objp, SIZE, corners2, ret)
#     new_closing_img = cv2.drawChessboardCorners(convertToColor, SIZE, corners2, ret)
#
#     cv2.namedWindow("new_img", cv2.WINDOW_NORMAL)
#     cv2.imshow("new_img", new_img)
#
#     cv2.namedWindow("new_closing_img", cv2.WINDOW_NORMAL)
#     cv2.imshow("new_closing_img", new_closing_img)
#
#     ret, rvecs, tvecs = cv2.solvePnP(img, corners2, camMatrix, camDist)
#
#     imgpts, jac = cv2.projectPoints(axisBoxes, rvecs, tvecs, camMatrix, camDist)
#
#
# else:
#     print("Corners not found")

while (1):
    if cv2.waitKey(1) == 27:
        break
