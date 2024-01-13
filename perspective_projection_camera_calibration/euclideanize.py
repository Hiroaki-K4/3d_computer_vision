import sys

import numpy as np

from calibrate_perspective_camera_by_primary_method import (
    calibrate_perspective_camera_by_primary_method,
    draw_reconstructed_points,
)

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calculate_mat_including_homography_mat(K, motion_mat):
    print(K.shape)
    print(motion_mat)
    # input()
    img_num = int(motion_mat.shape[0] / 3)
    A_k = np.zeros((4, 4, 4, 4))
    for idx in range(img_num):
        P = motion_mat[idx * 3 : idx * 3 + 3, :]
        Q = np.dot(np.linalg.inv(K[idx]), P)
        print(Q)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        A_k[i][j][k][l] += (
                            Q[0][i] * Q[0][j] * Q[0][k] * Q[0][l]
                            - Q[0][i] * Q[0][j] * Q[1][k] * Q[1][l]
                            - Q[1][i] * Q[1][j] * Q[0][k] * Q[0][l]
                            + Q[1][i] * Q[1][j] * Q[1][k] * Q[1][l]
                            + 1
                            / 4
                            * (
                                (Q[0][i] * Q[1][j] * Q[0][k] * Q[1][l])
                                + (Q[1][i] * Q[0][j] * Q[0][k] * Q[1][l])
                                + (Q[0][i] * Q[1][j] * Q[1][k] * Q[1][l])
                                + (Q[1][i] * Q[0][j] * Q[1][k] * Q[0][l])
                            )
                            + 1
                            / 4
                            * (
                                (Q[1][i] * Q[2][j] * Q[1][k] * Q[2][l])
                                + (Q[2][i] * Q[1][j] * Q[1][k] * Q[2][l])
                                + (Q[1][i] * Q[2][j] * Q[2][k] * Q[1][l])
                                + (Q[2][i] * Q[1][j] * Q[2][k] * Q[1][l])
                            )
                            + 1
                            / 4
                            * (
                                (Q[2][i] * Q[0][j] * Q[2][k] * Q[0][l])
                                + (Q[0][i] * Q[2][j] * Q[2][k] * Q[0][l])
                                + (Q[2][i] * Q[0][j] * Q[0][k] * Q[2][l])
                                + (Q[0][i] * Q[2][j] * Q[0][k] * Q[2][l])
                            )
                        )

    # TODO Calculate A mat
    A = np.array(
        [
            [
                A_k[0][0][0][0],
                A_k[0][0][1][1],
                A_k[0][0][2][2],
                A_k[0][0][3][3],
                np.sqrt(2) * A_k[0][0][0][1],
                np.sqrt(2) * A_k[0][0][0][2],
                np.sqrt(2) * A_k[0][0][0][3],
                np.sqrt(2) * A_k[0][0][1][2],
                np.sqrt(2) * A_k[0][0][1][3],
                np.sqrt(2) * A_k[0][0][2][3],
            ],
            [
                A_k[1][1][0][0],
                A_k[1][1][1][1],
                A_k[1][1][2][2],
                A_k[1][1][3][3],
                np.sqrt(2) * A_k[1][1][0][1],
                np.sqrt(2) * A_k[1][1][0][2],
                np.sqrt(2) * A_k[1][1][0][3],
                np.sqrt(2) * A_k[1][1][1][2],
                np.sqrt(2) * A_k[1][1][1][3],
                np.sqrt(2) * A_k[1][1][2][3],
            ],
        ]
    )
    print(A_k)
    input()


def euclideanize(motion_mat, shape_mat, f, f_0, opt_axis):
    J_med = sys.float_info.max
    img_num = int(motion_mat.shape[0] / 3)

    # Initialize intrinsic parameter K
    K = np.empty((img_num, 3, 3))
    for i in range(img_num):
        K_new = np.array([[f, 0, opt_axis[0]], [0, f, opt_axis[1]], [0, 0, f_0]])
        K[i] = K_new

    calculate_mat_including_homography_mat(K, motion_mat)


def main(show_flag: bool):
    rot_euler_degrees = [
        [-10, -30, 0],
        [15, -15, 0],
        [0, 0, 0],
        [15, 15, 0],
        [10, 30, 0],
    ]
    T_in_camera_coords = [[0, 0, 10], [0, 0, 7.5], [0, 0, 5], [0, 0, 7.5], [0, 0, 10]]
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

    f_0 = width
    motion_mat, shape_mat = calibrate_perspective_camera_by_primary_method(
        img_pnts_list, f_0, 2.0
    )
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    if show_flag:
        draw_reconstructed_points(
            img_pnts_list, motion_mat, shape_mat, width, height, f_0
        )

    euclideanize(motion_mat, shape_mat, f, f_0, [0.0, 0.0])


if __name__ == "__main__":
    # TODO Return flag when finish implementing
    show_flag = False
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
