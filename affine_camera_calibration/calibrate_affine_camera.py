import sys

import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calibrate_affine_camera(img_pnts_list):
    # TODO Don't use for loop
    for img_pnts in img_pnts_list:
        x_pnts = img_pnts[:, 0]
        y_pnts = img_pnts[:, 1]
        t_x = sum(x_pnts) / len(x_pnts)
        t_y = sum(y_pnts) / len(y_pnts)
        A = t_x * t_y
        C = t_x**2 - t_y**2
        


def main():
    rot_euler_degrees = [[0, -30, 0], [0, -15, 0], [0, 0, 0], [0, 15, 0], [0, 30, 0]]
    T_in_camera_coords = [[0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10]]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_list,
        noised_img_pnts_list,
        points_3d,
    ) = prepare_test_data.prepare_multiple_test_images(
        False, "CURVE", rot_euler_degrees, T_in_camera_coords, f, width, height
    )

    calibrate_affine_camera(img_pnts_list)


if __name__ == "__main__":
    main()
