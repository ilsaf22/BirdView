import os
import cv2


camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): на каком расстоянии снаружи выглядит вид птицы.
# калибровочного шаблона в горизонтальном и вертикальном направлениях ШИРИНА = 713, ВЫСОТА = 235
shift_w = 200
shift_h = 525

# shift_w = 500
# shift_h = 50
# МЕНЯЕМ ЭТО

# размер зазора между калибровочным шаблоном и автомобилем
# в горизонтальном и вертикальном направлениях
inn_shift_w = 50  # МЕНЯЕМ ЭТО БЫЛО 20, СТАЛО 50
inn_shift_h = 50

# общая ширина/высота сшитого изображения
total_w = 320 + 2 * shift_w #320 БЫЛО 600 СТАЛО 320
total_h = 240 + 2 * shift_h #240 БЫЛО 1000 СТАЛО 240

# четыре угла прямоугольной области, занимаемой автомобилем
# верхний-левый (x_left, y_top), нижний-правый (x_right, y_bottom)
xl = shift_w + 180 + inn_shift_w
xr = total_w - xl
yt = shift_h + 200 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# расположение пикселей четырех точек, которые необходимо выбрать.
# Вы должны щелкнуть эти пиксели в том же порядке при выполнении
# сценария get_projection_map.py
project_keypoints = {
    "front": [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "back":  [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "left":  [(shift_h + 280, shift_w),
              (shift_h + 840, shift_w),
              (shift_h + 280, shift_w + 160),
              (shift_h + 840, shift_w + 160)],

    "right": [(shift_h + 160, shift_w),
              (shift_h + 720, shift_w),
              (shift_h + 160, shift_w + 160),
              (shift_h + 720, shift_w + 160)]
}

# car_image = cv2.imread("/home/msi-user/PycharmProjects/170Camera/images/car.png")
# car_image = cv2.resize(car_image, (xr - xl, yb - yt))
