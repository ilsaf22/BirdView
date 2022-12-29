import cv2
import numpy as np

h = 720
w = 1280
sh = w + h + h
hs = h + h + w + w

img2 = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/BirdView/birdView_cam2_14.12.22.jpg')
img1 = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/BirdView/birdView_cam3_14.12.22.jpg')
# dst = cv2.addWeighted(img1, 0.5, img2, 0.7, 0)

src = np.float32([[0, 0],
                  [0, h],
                  [w, h],
                  [w, 0]])

dst1 = np.float32([[0, 0],
                   [0, h],
                   [w, h],
                   [w, 0]])
M1 = cv2.getPerspectiveTransform(src, dst1)

k = 1
while k < 3:
    dst2 = np.float32([[int(w / k), 0],
                       [int(w / k), h],
                       [int(w + w / k), h],
                       [int(w + w / k), 0]])
    M2 = cv2.getPerspectiveTransform(src, dst2)

    img22 = cv2.warpPerspective(img2, M2, (int(w * 2), int(h * 1.5)))
    img11 = cv2.warpPerspective(img1, M1, (int(w * 2), int(h * 1.5)))
    out1 = cv2.addWeighted(img11, 0.5, img22, 1, 0)

    # img_arr = np.hstack((img1, img2))

    # cv2.namedWindow('Input Images', cv2.WINDOW_NORMAL)
    # cv2.imshow('Input Images', img_arr)

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.imshow('out', out1)

    if cv2.waitKey(0) == 27:
        k -= 0.01
    else:
        k += 0.01
# cv2.imshow('Blended Image', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
