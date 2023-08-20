import sys

import cv2
import numpy as np

from triangulation import (
    calculate_camera_matrix_from_RT,
    optimal_correction,
    simple_triangulation,
)

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data_for_fundamental_matrix
from projective_transformation import (
    calculate_projective_trans_by_weighted_repetition as proj_trans,
)


def planner_triangulation(P_0, P_1, f_0, points_0, points_1, homography):
    theta = np.reshape(
        homography,
        [
            9,
        ],
    )

    p_ori = np.array([points_0[0], points_0[1], points_1[0], points_1[1]])
    p_est = np.copy(p_ori)
    p_move = np.zeros(4)
    prev_move_sum = sys.float_info.max
    while True:
        T_1 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-f_0, 0, 0, 0],
                [0, -f_0, 0, 0],
                [0, 0, 0, 0],
                [p_est[3], 0, 0, p_est[0]],
                [0, p_est[3], 0, p_est[1]],
                [0, 0, 0, f_0],
            ]
        )

        T_2 = np.array(
            [
                [f_0, 0, 0, 0],
                [0, f_0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-p_est[2], 0, -p_est[0], 0],
                [0, -p_est[2], -p_est[1], 0],
                [0, 0, -f_0, 0],
            ]
        )

        T_3 = np.array(
            [
                [-p_est[3], 0, 0, -p_est[0]],
                [0, -p_est[3], 0, -p_est[1]],
                [0, 0, 0, -f_0],
                [p_est[2], 0, p_est[0], 0],
                [0, p_est[2], p_est[1], 0],
                [0, 0, f_0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        T_list = [T_1, T_2, T_3]
        W = np.zeros((3, 3))
        for k in range(3):
            for l in range(3):
                V_0_xi = np.dot(T_list[k], T_list[l].T)
                W[k, l] = np.dot(theta, np.dot(V_0_xi, theta))

        w, v = np.linalg.eig(W)
        W_inv_rank_2 = np.zeros((3, 3))
        max_vec = np.array([v[:, np.argmax(w)]]).T
        W_inv_rank_2 += np.dot(max_vec, max_vec.T) / w[np.argmax(w)]
        v = np.delete(v, np.argmax(w), 1)
        w = np.delete(w, np.argmax(w), 0)
        max_vec = np.array([v[:, np.argmax(w)]]).T
        W_inv_rank_2 += np.dot(max_vec, max_vec.T) / w[np.argmax(w)]

        xi_1 = np.array(
            [
                0,
                0,
                0,
                -f_0 * p_est[0],
                -f_0 * p_est[1],
                -(f_0**2),
                p_est[0] * p_est[3],
                p_est[1] * p_est[3],
                f_0 * p_est[3],
            ]
        ) + np.dot(T_1, p_move)
        xi_2 = np.array(
            [
                f_0 * p_est[0],
                f_0 * p_est[1],
                f_0**2,
                0,
                0,
                0,
                -p_est[0] * p_est[2],
                -p_est[1] * p_est[2],
                -f_0 * p_est[2],
            ]
        ) + np.dot(T_2, p_move)
        xi_3 = np.array(
            [
                -p_est[0] * p_est[3],
                -p_est[1] * p_est[3],
                -f_0 * p_est[3],
                p_est[0] * p_est[2],
                p_est[1] * p_est[2],
                f_0 * p_est[2],
                0,
                0,
                0,
            ]
        ) + np.dot(T_3, p_move)
        xi_list = [xi_1, xi_2, xi_3]

        p_move_tmp = np.zeros(4)
        for i in range(3):
            for j in range(3):
                W = W_inv_rank_2[i, j]
                p_move_tmp += np.dot(
                    np.dot(np.dot(W, np.dot(xi_list[i], theta)), T_list[j].T), theta
                )

        p_move = p_move_tmp
        p_est = p_ori - p_move
        move_sum = p_move[0] ** 2 + p_move[1] ** 2 + p_move[2] ** 2 + p_move[3] ** 2
        if abs(prev_move_sum - move_sum) < 1e-5:
            break
        else:
            prev_move_sum = move_sum

    return np.array([p_est[0], p_est[1]]), np.array([p_est[2], p_est[3]])


def main():
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = prepare_test_data_for_fundamental_matrix.prepare_test_data(
        False, False, "PLANE"
    )
    f = 160
    P_0, P_1 = calculate_camera_matrix_from_RT(
        rot_1_to_2, trans_1_to_2_in_camera_coord, f
    )
    H = proj_trans.calculate_projective_trans_by_weighted_repetition(
        noised_img_pnts_0, noised_img_pnts_1
    )
    f_0 = 1
    for i in range(len(noised_img_pnts_0)):
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        pos = simple_triangulation(
            P_0, P_1, 640, noised_img_pnts_0[i][0], noised_img_pnts_1[i][0]
        )
        planner_point_0, planner_point_1 = planner_triangulation(
            P_0, P_1, f_0, noised_img_pnts_0[i][0], noised_img_pnts_1[i][0], H
        )
        planar_pos = simple_triangulation(
            P_0, P_1, 640, planner_point_0, planner_point_1
        )
        print("point: ", i)
        print(
            "2D points movement after planar triangulation: {0} -> {1}".format(
                noised_img_pnts_0[i][0], planner_point_0
            )
        )
        print(
            "2D points movement after planar triangulation: {0} -> {1}".format(
                noised_img_pnts_1[i][0], planner_point_1
            )
        )
        print(
            "3D points movement after planar triangulation: {0} -> {1}".format(
                pos, planar_pos
            )
        )


if __name__ == "__main__":
    main()
