import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calculate_camera_matrix_from_RT(R, T, f):
    focal_arr = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P = np.dot(focal_arr, I)

    Rt = np.dot(R.T, T)
    R_T = R.T
    moved_arr = np.array(
        [
            [R_T[0, 0], R_T[0, 1], R_T[0, 2], Rt[0, 0]],
            [R_T[1, 0], R_T[1, 1], R_T[1, 2], Rt[1, 0]],
            [R_T[2, 0], R_T[2, 1], R_T[2, 2], Rt[2, 0]],
        ]
    )
    P_after = np.dot(focal_arr, moved_arr)

    return P, P_after


def simple_triangulation(P_0, P_1, f_0, points_0, points_1):
    x_0 = points_0[0]
    y_0 = points_0[1]
    x_1 = points_1[0]
    y_1 = points_1[1]
    T = np.array(
        [
            [
                f_0 * P_0[0, 0] - x_0 * P_0[2, 0],
                f_0 * P_0[0, 1] - x_0 * P_0[2, 1],
                f_0 * P_0[0, 2] - x_0 * P_0[2, 2],
            ],
            [
                f_0 * P_0[1, 0] - y_0 * P_0[2, 0],
                f_0 * P_0[1, 1] - y_0 * P_0[2, 1],
                f_0 * P_0[1, 2] - y_0 * P_0[2, 2],
            ],
            [
                f_0 * P_1[0, 0] - x_1 * P_1[2, 0],
                f_0 * P_1[0, 1] - x_1 * P_1[2, 1],
                f_0 * P_1[0, 2] - x_1 * P_1[2, 2],
            ],
            [
                f_0 * P_1[1, 0] - y_1 * P_1[2, 0],
                f_0 * P_1[1, 1] - y_1 * P_1[2, 1],
                f_0 * P_1[1, 2] - y_1 * P_1[2, 2],
            ],
        ]
    )
    p = np.array(
        [
            f_0 * P_0[0, 3] - x_0 * P_0[2, 3],
            f_0 * P_0[1, 3] - y_0 * P_0[2, 3],
            f_0 * P_1[0, 3] - x_1 * P_1[2, 3],
            f_0 * P_1[1, 3] - y_1 * P_1[2, 3],
        ]
    )
    ans = (-1) * np.dot(np.dot(np.linalg.inv(np.dot(T.T, T)), T.T), p)

    return np.array([ans[0], ans[1], ans[2]])


def optimal_correction(F, f_0, points_0, points_1):
    S_0 = sys.float_info.max
    x_0 = points_0[0]
    y_0 = points_0[1]
    x_1 = points_1[0]
    y_1 = points_1[1]
    x_est_0 = x_0
    y_est_0 = y_0
    x_est_1 = x_1
    y_est_1 = y_1
    x_move_0 = 0
    y_move_0 = 0
    x_move_1 = 0
    y_move_1 = 0

    while True:
        V0_xi = np.array(
            [
                [
                    x_est_0**2 + x_est_1**2,
                    x_est_1 * y_est_1,
                    f_0 * x_est_1,
                    x_est_0 * y_est_0,
                    0,
                    0,
                    f_0 * x_est_0,
                    0,
                    0,
                ],
                [
                    x_est_1 * y_est_1,
                    x_est_0**2 + y_est_1**2,
                    f_0 * y_est_1,
                    0,
                    x_est_0 * y_est_0,
                    0,
                    0,
                    f_0 * x_est_0,
                    0,
                ],
                [f_0 * x_est_1, f_0 * y_est_1, f_0**2, 0, 0, 0, 0, 0, 0],
                [
                    x_est_0 * y_est_0,
                    0,
                    0,
                    y_est_0**2 + x_est_1**2,
                    x_est_1 * y_est_1,
                    f_0 * x_est_1,
                    f_0 * y_est_0,
                    0,
                    0,
                ],
                [
                    0,
                    x_est_0 * y_est_0,
                    0,
                    x_est_1 * y_est_1,
                    y_est_0**2 + y_est_1**2,
                    f_0 * y_est_1,
                    0,
                    f_0 * y_est_0,
                    0,
                ],
                [0, 0, 0, f_0 * x_est_1, f_0 * y_est_1, f_0**2, 0, 0, 0],
                [f_0 * x_est_0, 0, 0, f_0 * y_est_0, 0, 0, f_0**2, 0, 0],
                [0, f_0 * x_est_0, 0, 0, f_0 * y_est_0, 0, 0, f_0**2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        xi = np.array(
            [
                x_est_0 * x_est_1 + x_est_1 * x_move_0 + x_est_0 * x_move_1,
                x_est_0 * y_est_1 + y_est_1 * x_move_0 + x_est_0 * y_move_1,
                f_0 * (x_est_0 + x_move_0),
                y_est_0 * x_est_1 + x_est_1 * y_move_0 + y_est_0 * x_move_1,
                y_est_0 * y_est_1 + y_est_1 * y_move_0 + y_est_0 * y_move_1,
                f_0 * (y_est_0 + y_move_0),
                f_0 * (x_est_1 + x_move_1),
                f_0 * (y_est_1 + y_move_1),
                f_0**2,
            ]
        )

        theta = np.array(
            [
                F[0, 0],
                F[0, 1],
                F[0, 2],
                F[1, 0],
                F[1, 1],
                F[1, 2],
                F[2, 0],
                F[2, 1],
                F[2, 2],
            ]
        )
        theta_arr_0 = np.array(
            [[theta[0], theta[1], theta[2]], [theta[3], theta[4], theta[5]]]
        )
        theta_arr_1 = np.array(
            [[theta[0], theta[3], theta[6]], [theta[1], theta[4], theta[7]]]
        )
        point_move_0 = np.dot(
            np.dot(
                np.dot(xi, theta) / np.dot(theta, np.dot(V0_xi, theta)), theta_arr_0
            ),
            np.array([x_est_1, y_est_1, f_0]),
        )
        point_move_1 = np.dot(
            np.dot(
                np.dot(xi, theta) / np.dot(theta, np.dot(V0_xi, theta)), theta_arr_1
            ),
            np.array([x_est_0, y_est_0, f_0]),
        )
        x_move_0 = point_move_0[0]
        y_move_0 = point_move_0[1]
        x_move_1 = point_move_1[0]
        y_move_1 = point_move_1[1]
        x_est_0 = x_0 - x_move_0
        y_est_0 = y_0 - y_move_0
        x_est_1 = x_1 - x_move_1
        y_est_1 = y_1 - y_move_1
        S = x_move_0**2 + y_move_0**2 + x_move_1**2 + y_move_1**2
        if abs(S_0 - S) < 1e-5:
            break
        else:
            S_0 = S

    return x_est_0, y_est_0, x_est_1, y_est_1


def main(show=True):
    draw_test_data = False
    draw_epipolar = False
    rot_euler_deg_0 = [0, -10, 0]
    rot_euler_deg_1 = [0, 30, 0]
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
        F_true,
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
    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    P_0, P_1 = calculate_camera_matrix_from_RT(
        rot_1_to_2, trans_1_to_2_in_camera_coord, f
    )
    print("P_0: ", P_0)
    print("P_1: ", P_1)
    f_0 = 1
    for i in range(len(noised_img_pnts_0)):
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        pos = simple_triangulation(
            P_0, P_1, width, noised_img_pnts_0[i], noised_img_pnts_1[i]
        )
        x_0, y_0, x_1, y_1 = optimal_correction(
            F_true, f_0, noised_img_pnts_0[i], noised_img_pnts_1[i]
        )
        opt_pos = simple_triangulation(
            P_0, P_1, width, np.array([x_0, y_0]), np.array([x_1, y_1])
        )
        print("point: ", i)
        print(
            "noised_pos: ",
            noised_img_pnts_0[i][0],
            noised_img_pnts_0[i][1],
            noised_img_pnts_1[i][0],
            noised_img_pnts_1[i][1],
        )
        print("opt_pos: ", x_0, y_0, x_1, y_1)
        print(
            "ori_pos: ",
            img_pnts_0[i][0],
            img_pnts_0[i][1],
            img_pnts_1[i][0],
            img_pnts_1[i][1],
        )
        print("simple_triangulation_ori: ", pos)
        print("simple_triangulation_opt: ", opt_pos)
        cv2.circle(img_0, (int(x_0), int(y_0)), 3, (0, 0, 0), -1)
        cv2.circle(img_1, (int(x_1), int(y_1)), 3, (255, 0, 0), -1)

    if show:
        cv2.imshow("OPT_CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
        cv2.imshow("OPT_CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)


if __name__ == "__main__":
    show = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show = False

    main(show)
