# import opencv and numpy
import cv2
import numpy as np


# trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
    pass


# create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
# create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('r', 'controls', 15, 255, nothing)

# create a while loop act as refresh for the view
while (1):

    # create a black image
    img = np.zeros((512, 512, 3), np.uint8)
    # calculate center of image
    img_center_y = img.shape[0] // 2
    img_center_x = img.shape[1] // 2

    # returns current position/value of trackbar
    radius = int(cv2.getTrackbarPos('r', 'controls'))
    # draw a red circle in the center of the image with radius set by trackbar position
    cv2.circle(img, (img_center_y, img_center_x), radius, (0, 0, 255), -1)
    # show the image window
    cv2.imshow('img', img)

    # waitfor the user to press escape and break the while loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# destroys all window
cv2.destroyAllWindows()