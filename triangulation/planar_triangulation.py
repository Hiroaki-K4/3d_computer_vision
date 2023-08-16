import numpy as np
import cv2
import sys

from triangulation import calculate_camera_matrix_from_RT, simple_triangulation, optimal_correction

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data_for_fundamental_matrix


def planner_triangulation(P_0, P_1, f_0, points_0, points_1):
    p_ori = np.array([points_0[0], points_0[1], points_1[0], points_1[1]])
    p_est = np.copy(p_ori)
    p_move = np.zeros(4)
    while True:
        T_1 = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [-f_0, 0, 0, 0],
                        [0, -f_0, 0, 0],
                        [0, 0, 0, 0],
                        [p_est[3], 0, 0, p_est[0]],
                        [0, p_est[3], 0, p_est[1]],
                        [0, 0, 0, f_0]])

        T_2 = np.array([[f_0, 0, 0, 0],
                    [0, f_0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-p_est[2], 0, -p_est[0], 0],
                    [0, -p_est[2], -p_est[1], 0],
                    [0, 0, -f_0, 0]])

        T_3 = np.array([[-p_est[3], 0, 0, -p_est[0]],
                    [0, -p_est[3], 0, -p_est[1]],
                    [0, 0, 0, -f_0],
                    [p_est[2], 0, p_est[0], 0],
                    [0, p_est[2], p_est[1], 0],
                    [0, 0, f_0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

        print(T_1)
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
            P_0, P_1, 640, noised_img_pnts_0[i][0], noised_img_pnts_1[i][0]
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
