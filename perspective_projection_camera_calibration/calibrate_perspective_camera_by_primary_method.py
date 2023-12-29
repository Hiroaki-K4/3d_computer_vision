import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def normalize_each_column(mat):
    norm_mat = np.empty(mat.shape)
    for col in range(mat.shape[1]):
        norm_mat[:, col] = mat[:, col] / np.linalg.norm(mat[:, col])

    return norm_mat


def calculate_mat_for_decreasing_error(
    img_pnts_list, U, point_idx, frm_idx_1, frm_idx_2, f_0
):
    A = 0
    x_1 = np.ones(3)
    x_1[0] = img_pnts_list[frm_idx_1][point_idx][0] / f_0
    x_1[1] = img_pnts_list[frm_idx_1][point_idx][1] / f_0
    x_2 = np.ones(3)
    x_2[0] = img_pnts_list[frm_idx_2][point_idx][0] / f_0
    x_2[1] = img_pnts_list[frm_idx_2][point_idx][1] / f_0
    x_norm = np.linalg.norm(x_1) * np.linalg.norm(x_2)
    u_1 = np.empty(3)
    u_2 = np.empty(3)
    for i in range(4):
        u_1[0] = U[3 * frm_idx_1][i]
        u_1[1] = U[3 * frm_idx_1 + 1][i]
        u_1[2] = U[3 * frm_idx_1 + 2][i]
        u_2[0] = U[3 * frm_idx_2][i]
        u_2[1] = U[3 * frm_idx_2 + 1][i]
        u_2[2] = U[3 * frm_idx_2 + 2][i]
        A += np.dot(x_1, u_1) * np.dot(x_2, u_2) / x_norm

    return A


def update_projective_depth(Z, img_pnts_list, xi, point_idx, f_0):
    x = np.ones(3)
    for frm_idx in range(len(img_pnts_list)):
        x[0] = img_pnts_list[frm_idx][point_idx][0] / f_0
        x[1] = img_pnts_list[frm_idx][point_idx][1] / f_0
        Z[frm_idx][point_idx] = xi[frm_idx] / np.linalg.norm(x)


def update_observation_matrix(W, Z, img_pnts_list, f_0):
    for frm_idx in range(len(img_pnts_list)):
        for point_idx in range(len(img_pnts_list[0])):
            point = img_pnts_list[frm_idx][point_idx]
            W[frm_idx * 3][point_idx] = Z[frm_idx][point_idx] * point[0] / f_0
            W[frm_idx * 3 + 1][point_idx] = Z[frm_idx][point_idx] * point[1] / f_0
            W[frm_idx * 3 + 2][point_idx] = Z[frm_idx][point_idx]


def calculate_reprojection_error(motion_mat, shape_mat, img_pnts_list, f_0):
    E = 0
    for frm_idx in range(len(img_pnts_list)):
        for point_idx in range(len(img_pnts_list[0])):
            x = np.ones(3)
            x[0] = img_pnts_list[frm_idx][point_idx][0] / f_0
            x[1] = img_pnts_list[frm_idx][point_idx][1] / f_0
            P = motion_mat[frm_idx * 3 : frm_idx * 3 + 3, :]
            X = shape_mat[:, point_idx]
            pred_img_points = np.dot(P, X)
            pred_img_points = pred_img_points * (1 / pred_img_points[2])
            E += np.linalg.norm(x - pred_img_points) ** 2

    E = f_0 * np.sqrt(E / (len(img_pnts_list) * len(img_pnts_list[0])))

    return E


def calibrate_perspective_camera_by_primary_method(img_pnts_list, f_0):
    A = np.empty((len(img_pnts_list), len(img_pnts_list)))
    W = np.empty((3 * len(img_pnts_list), len(img_pnts_list[0])))
    Z = np.ones((len(img_pnts_list), len(img_pnts_list[0])))
    E_thr = 0.05
    E = sys.float_info.max
    while E > E_thr:
        update_observation_matrix(W, Z, img_pnts_list, f_0)

        norm_W = normalize_each_column(W)
        U, S, Vt = np.linalg.svd(norm_W)
        U = U[:, :4]
        S = np.array(
            [[S[0], 0, 0, 0], [0, S[1], 0, 0], [0, 0, S[2], 0], [0, 0, 0, S[3]]]
        )
        V = Vt.T[:, :4]

        for point_idx in range(len(img_pnts_list[0])):
            for frm_idx_1 in range(len(img_pnts_list)):
                for frm_idx_2 in range(len(img_pnts_list)):
                    A[frm_idx_1][frm_idx_2] = calculate_mat_for_decreasing_error(
                        img_pnts_list, U, point_idx, frm_idx_1, frm_idx_2, f_0
                    )

            w, v = np.linalg.eig(A)
            xi = v[:, np.argmax(w)]
            if sum(xi) < 0:
                xi = (-1) * xi
            update_projective_depth(Z, img_pnts_list, xi, point_idx, f_0)

        motion_mat = U
        shape_mat = np.dot(S, V.T)
        E = calculate_reprojection_error(motion_mat, shape_mat, img_pnts_list, f_0)
        print("Reprojection error: ", E)

    print("Finish!!")
    print("Final reprojection error: ", E)
    return motion_mat, shape_mat


def draw_reconstructed_points(img_pnts_list, motion_mat, shape_mat, width, height, f_0):
    for frm_idx in range(len(img_pnts_list)):
        img = np.full((height, width, 3), (255, 255, 255), np.uint8)
        for point_idx in range(len(img_pnts_list[0])):
            P = motion_mat[frm_idx * 3 : frm_idx * 3 + 3, :]
            X = shape_mat[:, point_idx]
            pred_img_points = np.dot(P, X)
            pred_img_points = pred_img_points * (1 / pred_img_points[2]) * f_0
            cv2.circle(
                img,
                (int(pred_img_points[0]), int(pred_img_points[1])),
                3,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                img,
                (
                    int(img_pnts_list[frm_idx][point_idx][0]),
                    int(img_pnts_list[frm_idx][point_idx][1]),
                ),
                3,
                (0, 0, 0),
                -1,
            )

        cam_name = "CAM" + str(frm_idx)
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

    f_0 = width
    motion_mat, shape_mat = calibrate_perspective_camera_by_primary_method(
        img_pnts_list, f_0
    )
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    if show_flag:
        draw_reconstructed_points(
            img_pnts_list, motion_mat, shape_mat, width, height, f_0
        )


if __name__ == "__main__":
    show_flag = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
