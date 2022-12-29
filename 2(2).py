import cv2
import imutils
import numpy as np
import imutils

filename = '/home/msi-user/Рабочий стол/IntrinsicCalibration/undishort_cameera_2(center)_20.12.22.jpg'

img = cv2.imread(filename)
median_image = cv2.blur(img, (10, 10))
img_grey = cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY)
thresh = 150
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(img.shape)

# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

cv2.imshow('contours', img_contours)  # выводим итоговое изображение в окно

cv2.waitKey(0)


def nothing(val):
    pass


def nothing2(val):
    pass


cv2.namedWindow('controls')
# cv2.createTrackbar('blur', "controls", 150, 255 - 1, nothing)
cv2.createTrackbar('threshold', "controls", 150, 255 - 1, nothing)
# cv2.createTrackbar('threshold2', "controls", 150, 255 - 1, nothing2)
nothing(150)
# nothing2(150)

# global imgray
i = 0

while True:
    threshold1 = int(cv2.getTrackbarPos('threshold', 'controls'))
    # threshold2 = int(cv2.getTrackbarPos('threshold2', 'controls'))
    ret, thresh = cv2.threshold(img_contours, threshold1, 255, 0)
    # ret2, thresh2 = cv2.threshold(thresh, threshold2, 255, 0)
    cv2.imshow("thresh", thresh)

    # cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     if len(approx) == 4:
    #         # screenCnt = approx
    #         break
    #
    # img3 = img.copy()
    # cv2.polylines(img3, [approx], True, (255, 0, 100), 4)
    # cv2.namedWindow("polylines", cv2.WINDOW_NORMAL)
    # cv2.imshow("polylines", img3)

    if cv2.waitKey(1) == 27:
        break
# cv2.imshow("polylines2", img)
# cv2.waitKey(0)
