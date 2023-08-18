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
                -f_0**2,
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

        print(xi_list)
        input()


def main():
    draw_test_data = False
    draw_epipolar = False
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
    width = 640
    height = 480
    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    P_0, P_1 = calculate_camera_matrix_from_RT(
        rot_1_to_2, trans_1_to_2_in_camera_coord, f
    )
    print("P_0: ", P_0)
    print("P_1: ", P_1)
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
        # x_0, y_0, x_1, y_1 = optimal_correction(
        #     F_true, f_0, noised_img_pnts_0[i][0], noised_img_pnts_1[i][0]
        # )
        # opt_pos = simple_triangulation(
        #     P_0, P_1, 640, np.array([x_0, y_0]), np.array([x_1, y_1])
        # )

        planner_pos = planner_triangulation(
            P_0, P_1, 640, noised_img_pnts_0[i][0], noised_img_pnts_1[i][0], H
        )

        print("point: ", i)
        print(
            "noised_pos: ",
            noised_img_pnts_0[i][0][0],
            noised_img_pnts_0[i][0][1],
            noised_img_pnts_1[i][0][0],
            noised_img_pnts_1[i][0][1],
        )
        # print("opt_pos: ", x_0, y_0, x_1, y_1)
        print(
            "ori_pos: ",
            img_pnts_0[i][0][0],
            img_pnts_0[i][0][1],
            img_pnts_1[i][0][0],
            img_pnts_1[i][0][1],
        )
        print("simple_triangulation: ", pos)
        # print("simple_triangulation_opt: ", opt_pos)
        # cv2.circle(img_0, (int(x_0), int(y_0)), 3, (0, 0, 0), -1)
        # cv2.circle(img_1, (int(x_1), int(y_1)), 3, (255, 0, 0), -1)

    # cv2.imshow("OPT_CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
    # cv2.imshow("OPT_CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
