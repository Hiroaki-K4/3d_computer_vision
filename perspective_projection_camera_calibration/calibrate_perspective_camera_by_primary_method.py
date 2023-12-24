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
    for i in range(len(img_pnts_list)):
        for j in range(len(img_pnts_list[0])):
            point = img_pnts_list[i][j]
            W[i * 3][j] = Z[i][j] * point[0] / f_0
            W[i * 3 + 1][j] = Z[i][j] * point[1] / f_0
            W[i * 3 + 2][j] = Z[i][j]


def calculate_reprojection_error(motion_mat, shape_mat, img_pnts_list, f_0):
    E = 0
    for frm_idx in range(len(img_pnts_list)):
        for point_idx in range(len(img_pnts_list[0])):
            x = np.ones(3)
            x[0] = img_pnts_list[frm_idx][point_idx][0] / f_0
            x[1] = img_pnts_list[frm_idx][point_idx][1] / f_0
            P = motion_mat[frm_idx*3:frm_idx*3+3, :]
            X = shape_mat[:, point_idx]
            pred_img_points = np.dot(P, X)
            pred_img_points = pred_img_points * (1 / pred_img_points[2])
            print("x: ", x)
            print("pred", pred_img_points)
            # TODO Implement reprojection error
            input()


def calibrate_perspective_camera_by_primary_method(img_pnts_list, f_0):
    A = np.empty((len(img_pnts_list), len(img_pnts_list)))
    W = np.empty((3 * len(img_pnts_list), len(img_pnts_list[0])))
    Z = np.ones((len(img_pnts_list), len(img_pnts_list[0])))
    update_observation_matrix(W, Z, img_pnts_list, f_0)

    norm_W = normalize_each_column(W)
    print("norm_W: ", norm_W)
    print("norm_W: ", norm_W.shape)

    U, S, Vt = np.linalg.svd(norm_W)
    U = U[:, :4]
    S = np.array([[S[0], 0, 0, 0], [0, S[1], 0, 0], [0, 0, S[2], 0], [0, 0, 0, S[3]]])
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

    print(Z)
    input()

    # TODO Add Camera matrix(P) and postion(X)
    motion_mat = U
    print(motion_mat.shape)
    shape_mat = np.dot(S, V.T)
    print(shape_mat.shape)
    input()
    calculate_reprojection_error(motion_mat, shape_mat, img_pnts_list, f_0)



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

    f_0 = 1
    calibrate_perspective_camera_by_primary_method(img_pnts_list, f_0)
    # motion_mat, shape_mat = calibrate_perspective_camera_by_primary_method(
    #     img_pnts_list, f_0
    # )
    # print("motion_mat: ", motion_mat.shape)
    # print("shape_mat: ", shape_mat.shape)
    # W_est = np.dot(motion_mat, shape_mat)
    # if show_flag:
    #     draw_reconstructed_points(W_est, width, height)


if __name__ == "__main__":
    show_flag = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
