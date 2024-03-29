import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calibrate_weak_perspective_camera(img_pnts_list):
    A = np.empty(len(img_pnts_list))
    C = np.empty(len(img_pnts_list))
    W = np.empty((2 * len(img_pnts_list), len(img_pnts_list[0])))
    for i in range(len(img_pnts_list)):
        points = img_pnts_list[i]
        t = np.mean(points, axis=0)
        t_x = t[0]
        t_y = t[1]
        A[i] = t_x * t_y
        C[i] = t_x**2 - t_y**2

        W[i * 2] = points[:, 0]
        W[i * 2 + 1] = points[:, 1]

    U, S, Vt = np.linalg.svd(W)
    U = U[:, :3]
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, S[2]]])
    V = Vt.T[:, :3]

    B_all = np.zeros((3, 3, 3, 3))
    for i in range(B_all.shape[0]):
        for j in range(B_all.shape[1]):
            for k in range(B_all.shape[2]):
                for l in range(B_all.shape[3]):
                    for m in range(len(img_pnts_list)):
                        u_k_1 = U[2 * m, :]
                        u_k_2 = U[2 * m + 1, :]
                        B_all[i, j, k, l] += (
                            u_k_1[i] * u_k_1[j] * u_k_1[k] * u_k_1[l]
                            - u_k_1[i] * u_k_1[j] * u_k_2[k] * u_k_2[l]
                            - u_k_2[i] * u_k_2[j] * u_k_1[k] * u_k_1[l]
                            + u_k_2[i] * u_k_2[j] * u_k_2[k] * u_k_2[l]
                        ) + 1 / 4 * (
                            u_k_1[i] * u_k_2[j] * u_k_1[k] * u_k_2[l]
                            + u_k_2[i] * u_k_1[j] * u_k_1[k] * u_k_2[l]
                            + u_k_1[i] * u_k_2[j] * u_k_2[k] * u_k_1[l]
                            + u_k_2[i] * u_k_1[j] * u_k_2[k] * u_k_1[l]
                        )

    B = np.array(
        [
            [
                B_all[0, 0, 0, 0],
                B_all[0, 0, 1, 1],
                B_all[0, 0, 2, 2],
                np.sqrt(2) * B_all[0, 0, 1, 2],
                np.sqrt(2) * B_all[0, 0, 2, 0],
                np.sqrt(2) * B_all[0, 0, 0, 1],
            ],
            [
                B_all[1, 1, 0, 0],
                B_all[1, 1, 1, 1],
                B_all[1, 1, 2, 2],
                np.sqrt(2) * B_all[1, 1, 1, 2],
                np.sqrt(2) * B_all[1, 1, 2, 0],
                np.sqrt(2) * B_all[1, 1, 0, 1],
            ],
            [
                B_all[2, 2, 0, 0],
                B_all[2, 2, 1, 1],
                B_all[2, 2, 2, 2],
                np.sqrt(2) * B_all[2, 2, 1, 2],
                np.sqrt(2) * B_all[2, 2, 2, 0],
                np.sqrt(2) * B_all[2, 2, 0, 1],
            ],
            [
                np.sqrt(2) * B_all[1, 2, 0, 0],
                np.sqrt(2) * B_all[1, 2, 1, 1],
                np.sqrt(2) * B_all[1, 2, 2, 2],
                2 * B_all[1, 2, 1, 2],
                2 * B_all[1, 2, 2, 0],
                2 * B_all[1, 2, 0, 1],
            ],
            [
                np.sqrt(2) * B_all[2, 0, 0, 0],
                np.sqrt(2) * B_all[2, 0, 1, 1],
                np.sqrt(2) * B_all[2, 0, 2, 2],
                2 * B_all[2, 0, 1, 2],
                2 * B_all[2, 0, 2, 0],
                2 * B_all[2, 0, 0, 1],
            ],
            [
                np.sqrt(2) * B_all[0, 1, 0, 0],
                np.sqrt(2) * B_all[0, 1, 1, 1],
                np.sqrt(2) * B_all[0, 1, 2, 2],
                2 * B_all[0, 1, 1, 2],
                2 * B_all[0, 1, 2, 0],
                2 * B_all[0, 1, 0, 1],
            ],
        ]
    )
    w, v = np.linalg.eig(B)
    r = v[:, np.argmin(w)]
    T = np.array(
        [
            [r[0], r[5] / np.sqrt(2), r[4] / np.sqrt(2)],
            [r[5] / np.sqrt(2), r[1], r[3] / np.sqrt(2)],
            [r[4] / np.sqrt(2), r[3] / np.sqrt(2), r[2]],
        ]
    )
    if np.linalg.det(T) < 0:
        T = (-1) * T

    w, v = np.linalg.eig(T)
    A = np.empty((3, 3))
    for i in range(w.shape[0]):
        if w[i] < 0:
            print("Matrix T is not a positive symetric matrix")
            return None, None

        A += np.dot(np.sqrt(w[i]), np.dot(v[:, i], v[:, i].T))

    motion_mat = np.dot(U, A)
    shape_mat = np.dot(np.dot(np.linalg.pinv(A), S), V.T)

    return motion_mat, shape_mat


def draw_reconstructed_points(W, width, height):
    for i in range(int(W.shape[0] / 2)):
        x_arr = W[i * 2]
        y_arr = W[i * 2 + 1]
        img = np.full((height, width, 3), (255, 255, 255), np.uint8)
        for j in range(int(x_arr.shape[0])):
            cv2.circle(img, (int(x_arr[j]), int(y_arr[j])), 3, (0, 0, 0), -1)
            cam_name = "CAM" + str(i)
            cv2.imshow(cam_name, cv2.resize(img, None, fx=0.5, fy=0.5))

    cv2.waitKey(0)


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

    motion_mat, shape_mat = calibrate_weak_perspective_camera(img_pnts_list)
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    W_est = np.dot(motion_mat, shape_mat)
    if show_flag:
        draw_reconstructed_points(W_est, width, height)


if __name__ == "__main__":
    show_flag = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
