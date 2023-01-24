import os
import numpy as np
import cv2
from PIL import Image


# import surround_view.param_settings as settings


class FisheyeCameraModel(object):
    """
    Fisheye camera model, for undistorting, projecting and flipping camera frames.
    """

    def __init__(self, camera_param_file, camera_name):
        if not os.path.isfile(camera_param_file):
            raise ValueError("Cannot find camera param file")

        if camera_name not in camera_names:
            raise ValueError("Unknown camera name: {}".format(camera_name))

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.scale_xy = (1.0, 1.0)
        self.shift_xy = (0, 0)
        self.undistort_maps = None
        self.project_matrix = None
        self.project_shape = project_shapes[self.camera_name]
        self.load_camera_params()

    def load_camera_params(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        # self.camera_matrix = np.float32(
        #     [[653.3316602062021, 0.0, 639.5157201817591], [0.0, 585.6811656471701, 350.2157172327047],
        #      [0.0, 0.0, 1.0]])

        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        # self.dist_coeffs = np.float32(
        #     [[-0.3124117494366039, 0.11418861375485781, 0.0003159985644099134, 5.923436858450262e-05,
        #       -0.0222586789709766]])

        self.resolution = (960, 640)

        scale_xy = fs.getNode("scale_xy").mat()
        if scale_xy is not None:
            self.scale_xy = scale_xy

        shift_xy = fs.getNode("shift_xy").mat()
        if shift_xy is not None:
            self.shift_xy = shift_xy

        project_matrix = fs.getNode("project_matrix").mat()
        if project_matrix is not None:
            self.project_matrix = project_matrix

        fs.release()
        self.update_undistort_maps()

    def update_undistort_maps(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        width, height = self.resolution

        self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3),
            new_matrix,
            (width, height),
            cv2.CV_16SC2
        )
        return self

    def set_scale_and_shift(self, scale_xy=(1.0, 1.0), shift_xy=(0, 0)):
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_maps()
        return self

    def undistort(self, image):
        result = cv2.remap(image, *self.undistort_maps, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return result

    def project(self, image):
        result = cv2.warpPerspective(image, self.project_matrix, self.project_shape)
        return result

    def flip(self, image):
        if self.camera_name == "front":
            # cv2.imwrite("front.jpg", image.copy())
            # cv2.imshow("front", image.copy())

            return image.copy()

        elif self.camera_name == "back":
            # cv2.namedWindow("back", cv2.WINDOW_NORMAL)
            cv2.imshow("back", image.copy()[::-1, ::-1, :])
            return image.copy()[::-1, ::-1, :]

        elif self.camera_name == "left":
            # cv2.imwrite("left.jpg", cv2.transpose(image)[::-1])
            # cv2.imshow("left", cv2.transpose(image)[::-1])
            return cv2.transpose(image)[::-1]

        else:
            # cv2.namedWindow("right", cv2.WINDOW_NORMAL)
            cv2.imshow("right", np.flip(cv2.transpose(image), 1))
            return np.flip(cv2.transpose(image), 1)

    def save_data(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("dist_coeffs", self.dist_coeffs)
        fs.write("resolution", self.resolution)
        fs.write("project_matrix", self.project_matrix)
        fs.write("scale_xy", np.float32(self.scale_xy))
        fs.write("shift_xy", np.float32(self.shift_xy))
        fs.release()


# --------------------------------------------------------------------
# (shift_width, shift_height): на каком расстоянии снаружи выглядит вид птицы.
# калибровочного шаблона в горизонтальном и вертикальном направлениях
shift_w = 300
shift_h = 300

# размер зазора между калибровочным шаблоном и автомобилем
# в горизонтальном и вертикальном направлениях
inn_shift_w = 20
inn_shift_h = 50

# общая ширина/высота сшитого изображения
total_w = 600 + 2 * shift_w
total_h = 1000 + 2 * shift_h

# четыре угла прямоугольной области, занимаемой автомобилем
# верхний-левый (x_left, y_top), нижний-правый (x_right, y_bottom)
xl = shift_w + 180 + inn_shift_w
xr = total_w - xl
yt = shift_h + 200 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------


camera_names = ["front", "back", "left", "right"]

project_shapes = {
    "front": (total_w, yt),
    "back": (total_w, yt),
    "left": (total_h, xl),
    "right": (total_h, xl)
}

images = [os.path.join(os.getcwd(), "images", name + ".png") for name in camera_names]
yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in camera_names]

# images = ['/home/msi-user/Рабочий стол/surround-view-system-introduction-master/images/front.png',
#           '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/images/back.png',
#           '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/images/left.png',
#           '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/images/right.png']
#
# yamls = ['/home/msi-user/Рабочий стол/surround-view-system-introduction-master/yaml/front.yaml',
#          '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/yaml/back.yaml',
#          '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/yaml/left.yaml',
#          '/home/msi-user/Рабочий стол/surround-view-system-introduction-master/yaml/right.yaml']

camera_models = [FisheyeCameraModel(camera_file, camera_name) for camera_file, camera_name in zip(yamls, camera_names)]

images2 = []
for image_file, camera in zip(images, camera_models):
    img = cv2.imread(image_file)
    # cv2.imshow("image_file", img)
    # cv2.waitKey(0)
    print(img.shape)
    img = camera.undistort(img)  # убираем фишай
    img = camera.project(img)  # делаем вид сверху
    img = camera.flip(img)  # переворачиваем картинки
    images2.append(img)

front, back, left, right = images2
cv2.imshow("front", front)
cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.imshow("back", back)


def get_outmost_polygon_boundary(img):
    """
    Дано изображение с маской, маска описывает область перекрытия
    двух изображений, получите крайний контур этой области.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # cv2.waitKey(0)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # get the contour with largest aera
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # polygon approximation
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)
    cv2.polylines(img, [polygon], True, (255, 0, 100), 4)
    cv2.namedWindow("polylines", cv2.WINDOW_NORMAL)
    cv2.imshow("polylines", img)
    return polygon


# Весовая матрица G, которая объединяет два изображения imA, imB
def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    Получите весовую матрицу G, которая объединяет два изображения imA, imB гладко.
    """
    overlap = cv2.bitwise_and(imA, imB)

    gray = cv2.cvtColor(overlap, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    overlapMask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    overlapMaskInv = cv2.bitwise_not(overlapMask)

    cv2.imshow("overlapMaskInv", overlapMaskInv)

    indices = np.where(overlapMask == 255)
    # print(indices)
    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)
    cv2.imshow("imA_diff", imA_diff)
    cv2.imshow("imB_diff", imB_diff)

    gray = cv2.cvtColor(imA, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    G = mask.astype(np.float32) / 255.0

    polyA = get_outmost_polygon_boundary(imA_diff)  # Получаем точки границы
    polyB = get_outmost_polygon_boundary(imB_diff)
    # cv2.imshow("polyA", polyA)
    # cv2.imshow("polyB", polyB)

    # cv2.waitKey(0)
    for y, x in zip(*indices):
        f = int(round(x))
        f2 = int(round(y))
        # print(f)
        # print(f2)
        pt = tuple([f, f2])
        distToB = cv2.pointPolygonTest(polyB, pt, True)
        if distToB < dist_threshold:
            distToA = cv2.pointPolygonTest(polyA, pt, True)
            distToB *= distToB
            distToA *= distToA
            G[y, x] = distToB / (distToA + distToB)
    cv2.imshow("overlapMask", overlapMask)
    return G, overlapMask


G0, M0 = get_weight_mask_matrix(front[:, :xl], left[:yt, :])
G1, M1 = get_weight_mask_matrix(front[:, xr:], right[:yt, :])
G2, M2 = get_weight_mask_matrix(back[:, :xl], left[yb:, :])
G3, M3 = get_weight_mask_matrix(back[:, xr:], right[yb:, :])

weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]
print(masks)
cv2.imshow("stack", np.stack((G0, G1, G2, G3), axis=2))
cv2.imshow("stack2", np.stack((M0, M1, M2, M3), axis=2))
# cv2.waitKey(0)
Gmat = np.stack((G0, G1, G2, G3), axis=2)
Mmat = np.stack((M0, M1, M2, M3), axis=2)


def merge(imA, imB, k):
    G = weights[k]
    return (imA * G + imB * (1 - G)).astype(np.uint8)


image = np.zeros((total_h, total_w, 3), np.uint8)

image[:yt, xl:xr] = front[:, xl:xr]
image[yb:, xl:xr] = back[:, xl:xr]
image[yt:yb, :xl] = left[yt:yb, :]
image[yt:yb, xr:] = right[yt:yb, :]
image[:yt, :xl] = merge(front[:, :xl], left[:yt, :], 0)
image[:yt, xr:] = merge(front[:, xr:], right[:yt, :], 1)
image[yb:, :xl] = merge(back[:, :xl], left[yb:, :], 2)
image[yb:, xr:] = merge(back[:, xr:], right[yb:, :], 3)

cv2.imshow("RESULT", image)
cv2.waitKey(0)
