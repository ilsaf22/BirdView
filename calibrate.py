import cv2
import numpy as np

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
SIZE_SCALE = 1
FOCAL_SCALE = 0.5

camera_mat = np.load("/home/msi-user/Рабочий стол/IntrinsicCalibration/allArrays/cam_3_data_12.12.22/camera_1_K.npy")
dist_coeff = np.load("/home/msi-user/Рабочий стол/IntrinsicCalibration/allArrays/cam_3_data_12.12.22/camera_1_D.npy")

image = cv2.imread("/home/msi-user/Рабочий стол/IntrinsicCalibration/All_images/CAM3_13.12.22/3.jpg")  # изображение


def _get_camera_mat_dst(camera_mat):
    camera_mat_dst = camera_mat.copy()
    camera_mat_dst[0][0] *= FOCAL_SCALE
    camera_mat_dst[1][1] *= FOCAL_SCALE
    camera_mat_dst[0][2] = FRAME_WIDTH / 2 * SIZE_SCALE
    camera_mat_dst[1][2] = FRAME_HEIGHT / 2 * SIZE_SCALE
    return camera_mat_dst

def _get_undistort_maps():
    camera_mat_dst = _get_camera_mat_dst(camera_mat)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_mat, dist_coeff, np.eye(3, 3), camera_mat_dst,
        (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv2.CV_16SC2)

    return map1, map2

def calibrator(img, map1, map2):
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def main():
    map1, map2 = _get_undistort_maps()
    undishort = calibrator(image, map1, map2)
    cv2.namedWindow("undishort", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("undishort", undishort)
    cv2.waitKey(0)
    cv2.imwrite("undishort_cameera_3(center)_20.12.22.jpg", undishort)


if __name__ == '__main__':
    main()
