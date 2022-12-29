import cv2
import ffmpeg
import numpy as np
import os
import threading

#----------------------------------------------------------------------#
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
SIZE_SCALE = 1
FOCAL_SCALE = 0.5
#----------------------------------------------------------------------#


img = cv2.imread('images/no_signal.jpg')
img2 = cv2.imread('/home/msi-user/Рабочий стол/IntrinsicCalibration/All_images/CAM_2_13.12.22/0.jpg')
camera_mat = np.load("../allArrays/cam_2_data_12.12.22/camera_1_K.npy")
dist_coeff = np.load("../allArrays/cam_2_data_12.12.22/camera_1_D.npy")
project_matrix = np.load("/home/msi-user/Рабочий стол/IntrinsicCalibration/BirdView/calibrated_image_cam2.npy")
#----------------------------------------------------------------------#


img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

#----------------------------------------------------------------------#
RTSP_Url_Cam1 = "rtsp://10.0.0.12:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream"
streamCam1 = (ffmpeg.input(RTSP_Url_Cam1, fflags='nobuffer', flags='low_delay').vflip().hflip()
              .output('pipe:', format='rawvideo', loglevel='quiet', pix_fmt='bgr24')
              .run_async(pipe_stdout=True, pipe_stderr=True))

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

#----------------------------------------------------------------------#

testA = [img2]


def cam1():
    while True:
        out = streamCam1.stdout.read(FRAME_HEIGHT * FRAME_WIDTH * 3)
        testA[0] = np.frombuffer(out, np.uint8).reshape([FRAME_HEIGHT, FRAME_WIDTH, 3])


def allcams():
    map1, map2 = _get_undistort_maps()
    #cv2.imshow("t", testA[0])
    while True:
        frame1 = cv2.remap(testA[0], map1, map2, cv2.INTER_LINEAR)
        #cv2.imshow("t", frame1)

        undist_image = cv2.warpPerspective(frame1, project_matrix, (FRAME_WIDTH, FRAME_HEIGHT))

        cv2.imshow('CAMERA1', undist_image)

        waitKey = cv2.waitKey(1)
        if waitKey == 27:
            break

#threading.Thread(target=cam1).start()

allcams()
