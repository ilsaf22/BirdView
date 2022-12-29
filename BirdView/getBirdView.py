import cv2
import ffmpeg
import numpy as np
from surround_view import PointSelector
import surround_view.param_settings as settings

h = 720
w = 1280

# -----------------------------------------------------------------------------------------------------------------------------------------------#


img = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/All_images/CAM3_13.12.22/3.jpg')
img2 = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/All_images/CAM3_13.12.22/0.jpg')

# img = cv2.imread('/home/msi-user/PycharmProjects/170Camera/31(4)/6.jpg')
# img2 = cv2.imread("/home/msi-user/PycharmProjects/170Camera/31(4)/0.jpg")
# img = cv2.imread('/home/msi-user/PycharmProjects/170Camera/31(3)/0.jpg')
# img2 = cv2.imread("/home/msi-user/PycharmProjects/170Camera/31(3)/6.jpg")


# -----------------------------------------------------------------------------------------------------------------------------------------------#

camera_mat = np.load("/home/msi-user/Рабочий стол/IntrinsicCalibration/allArrays/cam_3_data_12.12.22/camera_1_K.npy")
dist_coeff = np.load("/home/msi-user/Рабочий стол/IntrinsicCalibration/allArrays/cam_3_data_12.12.22/camera_1_D.npy")

# -----------------------------------------------------------------------------------------------------------------------------------------------#

# DIM = (1280, 720)
# # balance = 0
#
#
# img_dim = img.shape[:2][::-1]
# print(img_dim)
# scaled_K = camMatix * img_dim[0] / DIM[0]
# #scaled_K[2][2] = 0.0
#
# img_dim2 = img2.shape[:2][::-1]
# scaled_K2 = camMatix * img_dim2[0] / DIM[0]
# #scaled_K2[2][2] = 0.0
#
# new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, dist,
#                                                                img_dim, np.eye(3), balance=0.8)
# map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, dist, np.eye(3),
#                                                  new_K, img_dim, cv2.CV_16SC2)
#
# undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
#                          borderMode=cv2.BORDER_CONSTANT)
# #print(undist_image)
#
# # ------------------------------------------------------------------------------------------
# new_K2 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K2, dist,
#                                                                 img_dim2, np.eye(3), balance=0.8)
# map12, map22 = cv2.fisheye.initUndistortRectifyMap(scaled_K2, dist, np.eye(3),
#                                                    new_K2, img_dim2, cv2.CV_16SC2)
#
# undist_image2 = cv2.remap(img2, map12, map22, interpolation=cv2.INTER_LINEAR,
#                           borderMode=cv2.BORDER_CONSTANT)

# -----------------------------------------------------------------------------------------------------------------------------------------------#

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
SIZE_SCALE = 1
FOCAL_SCALE = 0.5

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
map1, map2 = _get_undistort_maps()

# -----------------------------------------------------------------------------------------------------------------------------------------------#

undist_image = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
undist_image2 = cv2.remap(img2, map1, map2, cv2.INTER_LINEAR)
name = "front"
gui = PointSelector(undist_image, title=name)
dst_points = settings.project_keypoints[name]

choice = gui.loop()
if choice > 0:
    src = np.float32(gui.keypoints)
    dst = np.float32(dst_points)
#print(src.tolist())
#src = np.float32([[484.0, 307.0], [716.0, 321.0], [222.0, 523.0], [836.0, 495.0]])
project_matrix = cv2.getPerspectiveTransform(src, dst)
print(project_matrix.tolist())

result = cv2.warpPerspective(undist_image, project_matrix, (w, h))
result2 = cv2.warpPerspective(undist_image2, project_matrix, (w, h))

cv2.imshow('CAMERA1', result)

# -----------------------------------------------------------------------------------------------------------------------------------------------#

cv2.imwrite("birdView_cam3_14.12.22.jpg", result2)
#np.save("calibrated_image_cam2", project_matrix)

# -----------------------------------------------------------------------------------------------------------------------------------------------#

cv2.imshow('CAMERA2', result2)
while (True):
    if (cv2.waitKey(0) == 27):
        break
