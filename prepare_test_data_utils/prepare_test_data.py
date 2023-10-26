import math

import cv2
import numpy as np


def euler_angle_to_rot_mat(x_deg, y_deg, z_deg):
    x = x_deg / 180 * math.pi
    y = y_deg / 180 * math.pi
    z = z_deg / 180 * math.pi
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    return np.dot(np.dot(R_x, R_y), R_z)


def create_circle_surface_points(radius, interval_deg):
    points = np.zeros((0, 3))
    for theta in range(360):
        if theta % interval_deg == 0:
            x = radius * np.cos(np.deg2rad(theta))
            y = radius * np.sin(np.deg2rad(theta))
            points = np.append(points, [[x, y, 0]], axis=0)

    return points


def create_curve_surface_points(row, col, z_scale):
    points = np.zeros((0, 3))
    for i in range(row + 1):
        for j in range(col + 1):
            x = i - row / 2
            y = j - col / 2
            z = x**2 * z_scale
            points = np.append(points, [[x, y, z]], axis=0)

    return points


def create_outer_product(trans_in_camera_coord):
    res = np.array(
        [
            [0, -trans_in_camera_coord[2, 0], trans_in_camera_coord[1, 0]],
            [trans_in_camera_coord[2, 0], 0, -trans_in_camera_coord[0, 0]],
            [-trans_in_camera_coord[1, 0], trans_in_camera_coord[0, 0], 0],
        ]
    )

    return res


def normalize_F_matrix(F_matrix):
    factor_sum = 0
    for i in range(F_matrix.shape[0]):
        for j in range(F_matrix.shape[1]):
            factor_sum += F_matrix[j, i] ** 2

    normalize_F_matrix = F_matrix / factor_sum**0.5

    return normalize_F_matrix


def calculate_true_fundamental_matrix(
    rot_mat_before,
    rot_mat_after,
    T_in_camera_coord_before,
    T_in_camera_coord_after,
    camera_matrix,
):
    rot_1_to_2 = np.dot(rot_mat_after, rot_mat_before.T)
    trans_1_to_2_in_camera_coord = (
        np.matrix(T_in_camera_coord_after).T
        - rot_1_to_2 * np.matrix(T_in_camera_coord_before).T
    )
    trans_1_to_2_in_camera_coord_outer = create_outer_product(
        trans_1_to_2_in_camera_coord
    )
    A_inv = np.linalg.inv(camera_matrix)
    F_true_1_to_2 = A_inv.T * trans_1_to_2_in_camera_coord_outer * rot_1_to_2 * A_inv

    return normalize_F_matrix(F_true_1_to_2), rot_1_to_2, trans_1_to_2_in_camera_coord


def draw_epipolar_lines(img_0, img_pnts_0, img_pnts_1):
    F, mask = cv2.findFundamentalMat(img_pnts_0, img_pnts_1, cv2.FM_LMEDS)

    lines_CAM1 = cv2.computeCorrespondEpilines(img_pnts_1, 2, F)
    lines_CAM1 = lines_CAM1.reshape(-1, 3)
    width_CAM1 = img_0.shape[1]
    for lines in lines_CAM1:
        x0, y0 = map(int, [0, -lines[2] / lines[1]])
        x1, y1 = map(int, [width_CAM1, -(lines[2] + lines[0] * width_CAM1) / lines[1]])
        img_0 = cv2.line(img_0, (x0, y0), (x1, y1), (0, 255, 0), 1)

    cv2.imshow("EPI", img_0)
    cv2.waitKey(0)


def add_noise(img_pnts, noise_scale):
    noise = np.random.normal(0, noise_scale, img_pnts.shape)
    noised_points = img_pnts + noise

    return noised_points


def prepare_test_data(
    draw_test_data,
    draw_epipolar,
    surface_type,
    rot_euler_deg_0,
    rot_euler_deg_1,
    T_0_in_camera_coord,
    T_1_in_camera_coord,
    f,
    width,
    height,
):
    rot_mat_0 = euler_angle_to_rot_mat(
        rot_euler_deg_0[0], rot_euler_deg_0[1], rot_euler_deg_0[2]
    )
    trans_vec_0 = np.eye(3) * np.matrix(T_0_in_camera_coord).T
    rot_mat_1 = euler_angle_to_rot_mat(
        rot_euler_deg_1[0], rot_euler_deg_1[1], rot_euler_deg_1[2]
    )
    trans_vec_1 = np.eye(3) * np.matrix(T_1_in_camera_coord).T
    if surface_type == "CURVE":
        points_3d = create_curve_surface_points(5, 5, 0.2)
    elif surface_type == "PLANE":
        points_3d = create_curve_surface_points(5, 5, 0)
    elif surface_type == "CIRCLE":
        points_3d = create_circle_surface_points(5, 3)
    else:
        raise RuntimeError("Surface type is wrong")
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)

    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(
        np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype="double")
    )
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(
        points_3d, rodri_0, trans_vec_0, camera_matrix, dist_coeffs
    )
    img_pnts_1, jac = cv2.projectPoints(
        points_3d, rodri_1, trans_vec_1, camera_matrix, dist_coeffs
    )
    # print(points_3d)
    # input()

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)

    noise_scale = 0.2
    noised_img_pnts_0 = add_noise(img_pnts_0, noise_scale)
    noised_img_pnts_1 = add_noise(img_pnts_1, noise_scale)

    for pnt in noised_img_pnts_0:
        cv2.circle(img_0, (int(pnt[0][0]), int(pnt[0][1])), 3, (0, 0, 0), -1)
    for pnt in noised_img_pnts_1:
        cv2.circle(img_1, (int(pnt[0][0]), int(pnt[0][1])), 3, (255, 0, 0), -1)

    (
        F_true_1_to_2,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = calculate_true_fundamental_matrix(
        rot_mat_0, rot_mat_1, T_0_in_camera_coord, T_1_in_camera_coord, camera_matrix
    )

    if draw_epipolar:
        draw_epipolar_lines(img_0, noised_img_pnts_0, noised_img_pnts_1)

    if draw_test_data:
        cv2.imshow("CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
        cv2.imshow("CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)

    return (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true_1_to_2,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
        points_3d,
    )

def prepare_test_data_three_images(
    draw_test_data,
    surface_type,
    rot_euler_deg_0,
    rot_euler_deg_1,
    rot_euler_deg_2,
    T_0_in_camera_coord,
    T_1_in_camera_coord,
    T_2_in_camera_coord,
    f,
    width,
    height,
):
    rot_mat_0 = euler_angle_to_rot_mat(
        rot_euler_deg_0[0], rot_euler_deg_0[1], rot_euler_deg_0[2]
    )
    trans_vec_0 = np.eye(3) * np.matrix(T_0_in_camera_coord).T
    rot_mat_1 = euler_angle_to_rot_mat(
        rot_euler_deg_1[0], rot_euler_deg_1[1], rot_euler_deg_1[2]
    )
    trans_vec_1 = np.eye(3) * np.matrix(T_1_in_camera_coord).T
    if surface_type == "CURVE":
        points_3d = create_curve_surface_points(5, 5, 0.2)
    elif surface_type == "PLANE":
        points_3d = create_curve_surface_points(5, 5, 0)
    elif surface_type == "CIRCLE":
        points_3d = create_circle_surface_points(5, 3)
    else:
        raise RuntimeError("Surface type is wrong")
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)

    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(
        np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype="double")
    )
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(
        points_3d, rodri_0, trans_vec_0, camera_matrix, dist_coeffs
    )
    img_pnts_1, jac = cv2.projectPoints(
        points_3d, rodri_1, trans_vec_1, camera_matrix, dist_coeffs
    )
    # print(points_3d)
    # input()

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)

    noise_scale = 0.2
    noised_img_pnts_0 = add_noise(img_pnts_0, noise_scale)
    noised_img_pnts_1 = add_noise(img_pnts_1, noise_scale)

    for pnt in noised_img_pnts_0:
        cv2.circle(img_0, (int(pnt[0][0]), int(pnt[0][1])), 3, (0, 0, 0), -1)
    for pnt in noised_img_pnts_1:
        cv2.circle(img_1, (int(pnt[0][0]), int(pnt[0][1])), 3, (255, 0, 0), -1)

    (
        F_true_1_to_2,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = calculate_true_fundamental_matrix(
        rot_mat_0, rot_mat_1, T_0_in_camera_coord, T_1_in_camera_coord, camera_matrix
    )

    if draw_test_data:
        cv2.imshow("CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
        cv2.imshow("CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)

    return (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true_1_to_2,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
        points_3d,
    )
