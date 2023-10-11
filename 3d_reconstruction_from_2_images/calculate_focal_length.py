import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calculate_focal_length(F):
    F_Ft = np.dot(F, F.T)
    Ft_F = np.dot(F.T, F)
    print("F_Ft: ", F_Ft)
    print("Ft_F: ", Ft_F)
    w, v = np.linalg.eig(F_Ft)
    e_1 = v[:, np.argmin(w)]
    w, v = np.linalg.eig(Ft_F)
    e_2 = v[:, np.argmin(w)]
    print("e_1: ", e_1)
    print("e_2: ", e_2)

    k = np.array([[0, 0, 1]]).T
    # print(k.shape)
    # input()
    # a = np.linalg.norm(np.dot(F, k)) ** 2 - np.dot(k, np.dot(np.dot(F_Ft, F), k)) * np.linalg.norm(np.cross(e_2, k)) ** 2 / np.dot(k, np.dot(F, k))
    print(k)
    print(np.dot(np.dot(F_Ft, F), k))
    # TODO: Fix error
    print(np.dot(k, np.dot(np.dot(F_Ft, F), k)))
    # print(k * np.dot(np.dot(F_Ft, F), k))
    # print(np.dot(np.dot(F_Ft, F), k).shape)


def main(draw_test_data, draw_epipolar):
    rot_euler_deg_0 = [0, 0, 0]
    rot_euler_deg_1 = [45, -30, 0]
    T_0_in_camera_coord = [0, 0, 10]
    T_1_in_camera_coord = [0, 0, 10]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_matrix,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
        points_3d,
    ) = prepare_test_data.prepare_test_data(
        draw_test_data,
        draw_epipolar,
        "CURVE",
        rot_euler_deg_0,
        rot_euler_deg_1,
        T_0_in_camera_coord,
        T_1_in_camera_coord,
        f,
        width,
        height,
    )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_matrix")
    print(F_matrix)
    calculate_focal_length(F_matrix)


if __name__ == "__main__":
    draw_test_data = False
    draw_epipolar = False
    main(draw_test_data, draw_epipolar)
