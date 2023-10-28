import sys

import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def optimal_correction_from_three_images(pnts_0, pnts_1, pnts_2, f_0):
    E_0 = sys.float_info.max
    x_0_move = 0
    x_1_move = 0
    x_2_move = 0
    x_0 = np.empty((pnts_0.shape[0], 3))
    x_1 = np.empty((pnts_1.shape[0], 3))
    x_2 = np.empty((pnts_2.shape[0], 3))
    for i in range(pnts_0.shape[0]):
        x_0[i] = np.array([pnts_0[i][0] / f_0, pnts_0[i][1] / f_0, 1])
        x_1[i] = np.array([pnts_1[i][0] / f_0, pnts_1[i][1] / f_0, 1])
        x_2[i] = np.array([pnts_2[i][0] / f_0, pnts_2[i][1] / f_0, 1])


def main():
    rot_euler_deg_0 = [0, -45, 0]
    rot_euler_deg_1 = [0, 0, 0]
    rot_euler_deg_2 = [0, 45, 0]
    T_0_in_camera_coord = [0, 0, 10]
    T_1_in_camera_coord = [0, 0, 10]
    T_2_in_camera_coord = [0, 0, 10]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_0,
        img_pnts_1,
        img_pnts_2,
        noised_img_pnts_0,
        noised_img_pnts_1,
        noised_img_pnts_2,
        points_3d,
    ) = prepare_test_data.prepare_test_data_three_images(
        False,
        "PLANE",
        rot_euler_deg_0,
        rot_euler_deg_1,
        rot_euler_deg_2,
        T_0_in_camera_coord,
        T_1_in_camera_coord,
        T_2_in_camera_coord,
        f,
        width,
        height,
    )
    f_0 = 1
    optimal_correction_from_three_images(
        noised_img_pnts_0, noised_img_pnts_1, noised_img_pnts_2, f_0
    )


if __name__ == "__main__":
    main()
