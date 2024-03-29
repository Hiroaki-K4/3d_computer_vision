import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def update_weight(theta, img_pnts_0, img_pnts_1, f_0):
    new_W = np.zeros((0, 3, 3))
    for i in range(len(img_pnts_0)):
        p_0 = img_pnts_0[i]
        p_1 = img_pnts_1[i]
        T_1 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-f_0, 0, 0, 0],
                [0, -f_0, 0, 0],
                [0, 0, 0, 0],
                [p_1[1], 0, 0, p_0[0]],
                [0, p_1[1], 0, p_0[1]],
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
                [-p_1[0], 0, -p_0[0], 0],
                [0, -p_1[0], -p_0[1], 0],
                [0, 0, -f_0, 0],
            ]
        )
        T_3 = np.array(
            [
                [-p_1[1], 0, 0, -p_0[0]],
                [0, -p_1[1], 0, -p_0[1]],
                [0, 0, 0, -f_0],
                [p_1[0], 0, p_0[0], 0],
                [0, p_1[0], p_0[1], 0],
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
        new_W = np.append(new_W, np.array([W_inv_rank_2]), axis=0)

    return new_W


def calculate_projective_trans_by_weighted_repetition(img_pnts_0, img_pnts_1):
    pre_theta = np.zeros(9)
    f_0 = 1
    M_sum = np.zeros([9, 9])
    W = np.zeros((0, 3, 3))
    for i in range(len(img_pnts_0)):
        W = np.append(W, np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), axis=0)

    count = 0
    count_thr = 1000
    while True:
        for i in range(len(img_pnts_0)):
            p_0 = img_pnts_0[i]
            p_1 = img_pnts_1[i]
            xi_1 = np.array(
                [
                    [
                        0,
                        0,
                        0,
                        -f_0 * p_0[0],
                        -f_0 * p_0[1],
                        -(f_0**2),
                        p_0[0] * p_1[1],
                        p_0[1] * p_1[1],
                        f_0 * p_1[1],
                    ]
                ]
            ).T
            xi_2 = np.array(
                [
                    [
                        f_0 * p_0[0],
                        f_0 * p_0[1],
                        f_0**2,
                        0,
                        0,
                        0,
                        -p_0[0] * p_1[0],
                        -p_0[1] * p_1[0],
                        -f_0 * p_1[0],
                    ]
                ]
            ).T
            xi_3 = np.array(
                [
                    [
                        -p_0[0] * p_1[1],
                        -p_0[1] * p_1[1],
                        -f_0 * p_1[1],
                        p_0[0] * p_1[0],
                        p_0[1] * p_1[0],
                        f_0 * p_1[0],
                        0,
                        0,
                        0,
                    ]
                ]
            ).T
            xi_list = [xi_1, xi_2, xi_3]
            for k in range(3):
                for l in range(3):
                    xi_k = xi_list[k]
                    xi_l = xi_list[l]
                    M_sum += np.dot(W[i, k, l], np.dot(xi_k, xi_l.T))

        M = M_sum / len(img_pnts_0)
        w, v = np.linalg.eig(M)
        theta = v[:, np.argmin(w)]
        if np.dot(theta, pre_theta) < 0:
            theta_diff = np.linalg.norm((-1) * theta - pre_theta)
        else:
            theta_diff = np.linalg.norm(theta - pre_theta)
        print("Theta diff: ", theta_diff)
        if theta_diff < 1e-8 or count > count_thr:
            break
        else:
            pre_theta = theta
            W = update_weight(theta, img_pnts_0, img_pnts_1, f_0)

        count += 1

    H = np.array(
        [
            [theta[0], theta[1], theta[2]],
            [theta[3], theta[4], theta[5]],
            [theta[6], theta[7], theta[8]],
        ]
    )

    return H


def evaluate(noised_img_pnts_0, noised_img_pnts_1, H):
    error = 0
    for i in range(len(noised_img_pnts_0)):
        pnt_0 = noised_img_pnts_0[i]
        pnt_1 = noised_img_pnts_1[i]
        from_pnt = np.array([pnt_0[0], pnt_0[1], 1])
        to_pnt = np.dot(H, from_pnt)
        to_1 = 1 / to_pnt[2]
        to_pnt = to_pnt * to_1
        error += abs(pnt_1[0] - to_pnt[0]) + abs(pnt_1[1] - to_pnt[1])

    return error / len(noised_img_pnts_0)


def main(show=True):
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
        False,
        False,
        "PLANE",
        rot_euler_deg_0,
        rot_euler_deg_1,
        T_0_in_camera_coord,
        T_1_in_camera_coord,
        f,
        width,
        height,
    )

    H = calculate_projective_trans_by_weighted_repetition(
        noised_img_pnts_0, noised_img_pnts_1
    )
    H_cv, _ = cv2.findHomography(
        np.float32(noised_img_pnts_0), np.float32(noised_img_pnts_1)
    )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Weighted repetition H: ")
    print(H)
    error_avg = evaluate(noised_img_pnts_0, noised_img_pnts_1, H)
    print("Error avg: ", error_avg)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Opencv H: ")
    print(H_cv)
    error_avg_cv = evaluate(noised_img_pnts_0, noised_img_pnts_1, H_cv)
    print("Error avg: ", error_avg_cv)

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    for pnt in noised_img_pnts_0:
        cv2.circle(img_0, (int(pnt[0]), int(pnt[1])), 3, (255, 0, 0), -1)
    for pnt in noised_img_pnts_1:
        cv2.circle(img_1, (int(pnt[0]), int(pnt[1])), 3, (255, 0, 0), -1)

    pers_img = cv2.warpPerspective(img_0, H, (width, height))
    pers_img_cv = cv2.warpPerspective(img_0, H_cv, (width, height))

    if show:
        cv2.imshow("CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
        cv2.imshow("CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
        cv2.imshow("pers_weighted_rep", cv2.resize(pers_img, None, fx=0.5, fy=0.5))
        cv2.imshow("pers_opencv", cv2.resize(pers_img_cv, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)


if __name__ == "__main__":
    show = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show = False

    main(show)
