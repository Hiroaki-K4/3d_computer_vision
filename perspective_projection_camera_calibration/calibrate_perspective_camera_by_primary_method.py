import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import euclideanize

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


def calibrate_perspective_camera_by_primary_method(img_pnts_list, f_0, error_thr):
    A = np.empty((len(img_pnts_list), len(img_pnts_list)))
    W = np.empty((3 * len(img_pnts_list), len(img_pnts_list[0])))
    Z = np.ones((len(img_pnts_list), len(img_pnts_list[0])))
    E = sys.float_info.max
    while E > error_thr:
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
        resize_img = cv2.resize(img, None, fx=0.5, fy=0.5)
        file_name = cam_name + ".png"
        save_path = os.path.join("images", file_name)
        cv2.imwrite(save_path, resize_img)
        cv2.imshow(cam_name, resize_img)

    cv2.waitKey(0)


def rotate_and_translate_original_points(R, t, X_s):
    moved_X = np.zeros(np.shape(X_s))
    for col in range(X_s.shape[1]):
        X = X_s[:, col]
        moved_X[:, col] = np.dot(R.T, (X - t))

    return moved_X


def reconstruct_3d_position(motion_mat, shape_mat, H, K):
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    print("H: ", H.shape)
    print("K: ", K.shape)
    fix_X = np.dot(np.linalg.inv(H), shape_mat)
    norm_X = fix_X / fix_X[3, :]
    norm_X = norm_X[:3, :]
    fix_P = np.dot(motion_mat, H)
    moved_Xs = np.zeros((K.shape[0], 3, shape_mat.shape[1]))
    for frm_idx in range(K.shape[0]):
        P_k = fix_P[frm_idx * 3 : frm_idx * 3 + 3, :]
        kp = np.dot(np.linalg.inv(K[frm_idx]), P_k)
        A_k = kp[:, :3]
        b_k = kp[:, 3]
        s = np.cbrt(np.linalg.det(A_k))
        A_k = A_k / s
        b_k = b_k / s
        U, S, Vh = np.linalg.svd(A_k)
        R_k = np.dot(Vh.T, U.T)
        t_k = -np.dot(R_k, b_k)
        moved_X = rotate_and_translate_original_points(R_k, t_k, norm_X)
        Z_s = moved_X[2, :]
        sign_sum = np.sum(np.sign(Z_s))
        if sign_sum <= 0:
            moved_X = rotate_and_translate_original_points(R_k, -t_k, -norm_X)
        moved_Xs[frm_idx, :, :] = moved_X

    return moved_Xs


def draw_reconstructed_3d_points(moved_Xs):
    for i in range(moved_Xs.shape[0]):
        ax = plt.figure().add_subplot(projection="3d")
        title = "IMG " + str(i)
        ax.set_title(title)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        for j in range(moved_Xs.shape[2]):
            X = moved_Xs[i][0][j]
            Y = moved_Xs[i][1][j]
            Z = moved_Xs[i][2][j]
            ax.scatter(X, Z, -Y, color="blue")

    plt.show()


def main(show_flag: bool, show_3d_points: bool):
    rot_euler_degrees = [
        [-10, -30, 0],
        [15, -15, 0],
        [0, 0, 0],
        [15, 15, 0],
        [10, 45, 0],
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
        img_pnts_list, f_0, 0.05
    )
    H, K = euclideanize.euclideanize(motion_mat, shape_mat, f, f_0, [0.0, 0.0])
    moved_Xs = reconstruct_3d_position(motion_mat, shape_mat, H, K)

    if show_3d_points:
        draw_reconstructed_3d_points(moved_Xs)

    if show_flag:
        draw_reconstructed_points(
            img_pnts_list, motion_mat, shape_mat, width, height, f_0
        )


if __name__ == "__main__":
    show_flag = True
    show_3d_points = False
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    if len(sys.argv) == 2 and sys.argv[1] == "Show3DPoints":
        show_3d_points = True
    main(show_flag, show_3d_points)
