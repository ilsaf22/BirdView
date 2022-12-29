import cv2
import numpy as np

my_photo = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/undishort_cameera_2(center)_20.12.22.jpg')
img = cv2.cvtColor(my_photo, cv2.COLOR_BGR2HSV)
h_channel = my_photo[:, :, 0]
v_channel = my_photo[:, :, 2]
bin_img = np.zeros(my_photo.shape)
bin_img[(h_channel < 70) * (h_channel > 20) * (v_channel > 100)] = [0, 0, 255]
cv2.imshow('v_channel', v_channel)
cv2.imshow('result', bin_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
