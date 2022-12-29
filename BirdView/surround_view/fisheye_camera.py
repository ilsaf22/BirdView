import os
import numpy as np
import cv2

from . import param_settings as settings


class FisheyeCameraModel(object):
    """
    Fisheye camera model, for undistorting, projecting and flipping camera frames.
    """

    def __init__(self, camera_param_file, camera_name):
        if not os.path.isfile(camera_param_file):
            raise ValueError("Cannot find camera param file")

        if camera_name not in settings.camera_names:
            raise ValueError("Unknown camera name: {}".format(camera_name))

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.scale_xy = (0.0, 0.0)
        self.shift_xy = (0, 0)
        self.undistort_maps = None
        self.project_matrix = None
        self.project_shape = settings.project_shapes[self.camera_name]
        self.load_camera_params()

    def load_camera_params(self):
        #fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        # self.camera_matrix = fs.getNode("camera_matrix").mat()
        # self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        # #print(self.dist_coeffs)
        # self.resolution = [640, 960]
        # self.resolution = fs.getNode("resolution").mat().flatten()

        camMatrix = np.load("/arrays_K_D/K2.npy")
        camDist = np.load("/arrays_K_D/D2.npy")

        self.camera_matrix = camMatrix
        self.dist_coeffs = camDist
        self.resolution = [1280, 720]
        # self.resolution = [1920, 1080]

        print(self.resolution)
        # scale_xy = fs.getNode("scale_xy").mat()
        # if scale_xy is not None:
        #     self.scale_xy = scale_xy
        #
        # shift_xy = fs.getNode("shift_xy").mat()
        # if shift_xy is not None:
        #     self.shift_xy = shift_xy
        #
        # project_matrix = fs.getNode("project_matrix").mat()
        # if project_matrix is not None:
        #     self.project_matrix = project_matrix
        #
        # fs.release()
        self.update_undistort_maps()

    def update_undistort_maps(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        width, height = self.resolution

        # self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
        #     self.camera_matrix,
        #     self.dist_coeffs,
        #     np.eye(3),
        #     self.camera_matrix,
        #     (width, height),
        #     cv2.CV_16SC2
        # )
        #self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, np.eye(3), self.camera_matrix, (width, height), cv2.CV_16SC2)
        return self

    def set_scale_and_shift(self, scale_xy=(0.0, 0.0), shift_xy=(0, 0)):
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_maps()
        return self

    def undistort(self, image):
        #pass
        # = cv2.remap(image, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # result = cv2.remap(image, *self.undistort_maps, interpolation=cv2.INTER_LINEAR,
        #                    borderMode=cv2.BORDER_CONSTANT)
        return image

    def project(self, image):
        print(self.project_shape)
        result = cv2.warpPerspective(image, self.project_matrix, self.project_shape)
        return result

    def flip(self, image):
        if self.camera_name == "front":
            return image.copy()

        elif self.camera_name == "back":
            return image.copy()[::-1, ::-1, :]

        elif self.camera_name == "left":
            return cv2.transpose(image)[::-1]

        else:
            return np.flip(cv2.transpose(image), 1)

    def save_data(self, img2, file_name):
        # print(self.project_matrix)
        # np.save(file_name, self.project_matrix)

        img = cv2.imread(img2)
        remap = self.undistort(img)
        warped = self.project(remap)
        # cv2.imwrite("3.jpg", warped)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)
        # fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE)
        # fs.write("camera_matrix", self.camera_matrix)
        # fs.write("dist_coeffs", self.dist_coeffs)
        # fs.write("resolution", self.resolution)
        # fs.write("project_matrix", self.project_matrix)
        # fs.write("scale_xy", np.float32(self.scale_xy))
        # fs.write("shift_xy", np.float32(self.shift_xy))
        # fs.release()
