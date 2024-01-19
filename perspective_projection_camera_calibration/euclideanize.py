import sys

import numpy as np

from calibrate_perspective_camera_by_primary_method import (
    calibrate_perspective_camera_by_primary_method,
    draw_reconstructed_points,
)

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def create_A_k_matrix(Q):
    A_k = np.zeros((4, 4, 4, 4))
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

    return A_k


def create_A_matrix(A_k):
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
            [
                A_k[2][2][0][0],
                A_k[2][2][1][1],
                A_k[2][2][2][2],
                A_k[2][2][3][3],
                np.sqrt(2) * A_k[2][2][0][1],
                np.sqrt(2) * A_k[2][2][0][2],
                np.sqrt(2) * A_k[2][2][0][3],
                np.sqrt(2) * A_k[2][2][1][2],
                np.sqrt(2) * A_k[2][2][1][3],
                np.sqrt(2) * A_k[2][2][2][3],
            ],
            [
                A_k[3][3][0][0],
                A_k[3][3][1][1],
                A_k[3][3][2][2],
                A_k[3][3][3][3],
                np.sqrt(2) * A_k[3][3][0][1],
                np.sqrt(2) * A_k[3][3][0][2],
                np.sqrt(2) * A_k[3][3][0][3],
                np.sqrt(2) * A_k[3][3][1][2],
                np.sqrt(2) * A_k[3][3][1][3],
                np.sqrt(2) * A_k[3][3][2][3],
            ],
            [
                np.sqrt(2) * A_k[0][1][0][0],
                np.sqrt(2) * A_k[0][1][1][1],
                np.sqrt(2) * A_k[0][1][2][2],
                np.sqrt(2) * A_k[0][1][3][3],
                2 * A_k[0][1][0][1],
                2 * A_k[0][1][0][2],
                2 * A_k[0][1][0][3],
                2 * A_k[0][1][1][2],
                2 * A_k[0][1][1][3],
                2 * A_k[0][1][2][3],
            ],
            [
                np.sqrt(2) * A_k[0][2][0][0],
                np.sqrt(2) * A_k[0][2][1][1],
                np.sqrt(2) * A_k[0][2][2][2],
                np.sqrt(2) * A_k[0][2][3][3],
                2 * A_k[0][2][0][1],
                2 * A_k[0][2][0][2],
                2 * A_k[0][2][0][3],
                2 * A_k[0][2][1][2],
                2 * A_k[0][2][1][3],
                2 * A_k[0][2][2][3],
            ],
            [
                np.sqrt(2) * A_k[0][3][0][0],
                np.sqrt(2) * A_k[0][3][1][1],
                np.sqrt(2) * A_k[0][3][2][2],
                np.sqrt(2) * A_k[0][3][3][3],
                2 * A_k[0][3][0][1],
                2 * A_k[0][3][0][2],
                2 * A_k[0][3][0][3],
                2 * A_k[0][3][1][2],
                2 * A_k[0][3][1][3],
                2 * A_k[0][3][2][3],
            ],
            [
                np.sqrt(2) * A_k[1][2][0][0],
                np.sqrt(2) * A_k[1][2][1][1],
                np.sqrt(2) * A_k[1][2][2][2],
                np.sqrt(2) * A_k[1][2][3][3],
                2 * A_k[1][2][0][1],
                2 * A_k[1][2][0][2],
                2 * A_k[1][2][0][3],
                2 * A_k[1][2][1][2],
                2 * A_k[1][2][1][3],
                2 * A_k[1][2][2][3],
            ],
            [
                np.sqrt(2) * A_k[1][3][0][0],
                np.sqrt(2) * A_k[1][3][1][1],
                np.sqrt(2) * A_k[1][3][2][2],
                np.sqrt(2) * A_k[1][3][3][3],
                2 * A_k[1][3][0][1],
                2 * A_k[1][3][0][2],
                2 * A_k[1][3][0][3],
                2 * A_k[1][3][1][2],
                2 * A_k[1][3][1][3],
                2 * A_k[1][3][2][3],
            ],
            [
                np.sqrt(2) * A_k[2][3][0][0],
                np.sqrt(2) * A_k[2][3][1][1],
                np.sqrt(2) * A_k[2][3][2][2],
                np.sqrt(2) * A_k[2][3][3][3],
                2 * A_k[2][3][0][1],
                2 * A_k[2][3][0][2],
                2 * A_k[2][3][0][3],
                2 * A_k[2][3][1][2],
                2 * A_k[2][3][1][3],
                2 * A_k[2][3][2][3],
            ],
        ]
    )

    return A


def calculate_mat_including_homography_mat(K, motion_mat):
    img_num = int(motion_mat.shape[0] / 3)
    A_k = np.zeros((4, 4, 4, 4))
    for idx in range(img_num):
        P = motion_mat[idx * 3 : idx * 3 + 3, :]
        Q = np.dot(np.linalg.inv(K[idx]), P)
        A_k += create_A_k_matrix(Q)

    A = create_A_matrix(A_k)
    w, v = np.linalg.eig(A)
    w = v[:, np.argmin(w)]
    omega = np.array(
        [
            [w[0], w[4] / np.sqrt(2), w[5] / np.sqrt(2), w[6] / np.sqrt(2)],
            [w[4] / np.sqrt(2), w[1], w[7] / np.sqrt(2), w[8] / np.sqrt(2)],
            [w[5] / np.sqrt(2), w[7] / np.sqrt(2), w[2], w[9] / np.sqrt(2)],
            [w[6] / np.sqrt(2), w[8] / np.sqrt(2), w[9] / np.sqrt(2), w[3]],
        ]
    )
    o_w, o_v = np.linalg.eig(omega)
    print("o_w: ", o_w)
    sort_idx = np.argsort(o_w)[::-1]
    print(sort_idx)
    print("sort: ", o_w[sort_idx[2]])
    # TODO Reshape to multipe array
    # o_v[sort_idx[0]] = np.reshape(o_v[sort_idx[0]], (4, 1))
    o_v[sort_idx[0]] = np.array([[o_v[sort_idx[0]]]])
    print(o_v[sort_idx[0]].shape)
    input()
    if o_w[sort_idx[2]] > 0:
        omega = o_v[sort_idx[0]] * np.dot(o_v[sort_idx[0]], o_v[sort_idx[0]].T)
    else:
        omega = o_v[sort_idx[0]] * np.dot(o_v[sort_idx[0]], o_v[sort_idx[0]].T)
    print("ome: ", omega)
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
