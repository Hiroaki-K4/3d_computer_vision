import numpy as np
import math
import cv2


def euler_angle_to_rot_mat(x_deg, y_deg, z_deg):
    x = x_deg / 180 * math.pi
    y = y_deg / 180 * math.pi
    z = z_deg / 180 * math.pi
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    return np.dot(np.dot(R_x, R_y), R_z)


def create_curve_surface_points(row, col, z_scale):
    points = np.zeros((0, 3))
    for i in range(row+1):
        for j in range(col+1):
            x = i - row / 2
            y = j - col / 2
            z = x ** 2 * z_scale
            points = np.append(points, [[x, y, z]], axis=0)

    return points


def create_outer_product(trans_in_camera_coord):
    res = np.array([[0, -trans_in_camera_coord[2, 0], trans_in_camera_coord[1, 0]],
                    [trans_in_camera_coord[2, 0], 0, -trans_in_camera_coord[0, 0]],
                    [-trans_in_camera_coord[1, 0], trans_in_camera_coord[0, 0], 0]])

    return res


def normalize_F_matrix(F_matrix):
    factor_sum = 0
    for i in range(F_matrix.shape[0]):
        for j in range(F_matrix.shape[1]):
            factor_sum += F_matrix[j, i] ** 2
            # print("factor_sum: ", factor_sum)
    normalize_F_matrix = F_matrix / factor_sum ** 0.5

    return normalize_F_matrix


def calculate_true_fundamental_matrix(rot_mat_before, rot_mat_after, T_in_camera_coord_before, T_in_camera_coord_after, camera_matrix):
    rot_1_to_2 = rot_mat_after * rot_mat_before.T
    trans_1_to_2_in_camera_coord = np.matrix(T_in_camera_coord_after).T - rot_1_to_2 * np.matrix(T_in_camera_coord_before).T
    print(trans_1_to_2_in_camera_coord)
    print(np.matrix(T_in_camera_coord_after).T)
    print(rot_1_to_2 * np.matrix(T_in_camera_coord_before).T)
    trans_1_to_2_in_camera_coord_outer = create_outer_product(trans_1_to_2_in_camera_coord)
    print(trans_1_to_2_in_camera_coord_outer)
    A_inv = np.linalg.inv(camera_matrix)
    F_true_1_to_2 = A_inv.T * trans_1_to_2_in_camera_coord_outer * rot_1_to_2 * A_inv

    return normalize_F_matrix(F_true_1_to_2)


def calculate_f_matrix_by_least_squares(img_pnts_0, img_pnts_1):
    if len(img_pnts_0) != len(img_pnts_1):
        raise RuntimeError("The number points is wrong.")

    f_x = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_y = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_0 = max(f_x, f_y)
    f0 = 1

    xi_sum = np.zeros((9, 9))
    for i in range(len(img_pnts_0)):
        xi = np.array([[img_pnts_0[i, 0, 0] * img_pnts_1[i, 0, 0],
                       img_pnts_0[i, 0, 0] * img_pnts_1[i, 0, 1],
                       f_0 * img_pnts_0[i, 0, 0],
                       img_pnts_0[i, 0, 1] * img_pnts_1[i, 0, 0],
                       img_pnts_0[i, 0, 1] * img_pnts_1[i, 0, 1],
                       f_0 * img_pnts_0[i, 0, 1],
                       f_0 * img_pnts_1[i, 0, 0],
                       f_0 * img_pnts_1[i, 0, 1],
                       f_0 ** 2]])
        xi_sum += np.dot(xi.T, xi)

    M = xi_sum / len(img_pnts_0)
    w, v = np.linalg.eig(M)
    theta = v[:, np.argmin(w)]
    reshaped = np.reshape(theta, (3, 3))

    return normalize_F_matrix(reshaped)


def calculate_f_matrix_by_taubin(img_pnts_0, img_pnts_1):
    if len(img_pnts_0) != len(img_pnts_1):
        raise RuntimeError("The number points is wrong.")

    f_x = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_y = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_0 = max(f_x, f_y)
    f0 = 1

    xi_sum = np.zeros((9, 9))
    V0_xi_sum = np.zeros((9, 9))
    for i in range(len(img_pnts_0)):
        x_0 = img_pnts_0[i, 0, 0]
        y_0 = img_pnts_0[i, 0, 1]
        x_1 = img_pnts_1[i, 0, 0]
        y_1 = img_pnts_1[i, 0, 1]
        # TODO: Change img_pnts_* to x_* and y_*
        xi = np.array([[img_pnts_0[i, 0, 0] * img_pnts_1[i, 0, 0],
                       img_pnts_0[i, 0, 0] * img_pnts_1[i, 0, 1],
                       f_0 * img_pnts_0[i, 0, 0],
                       img_pnts_0[i, 0, 1] * img_pnts_1[i, 0, 0],
                       img_pnts_0[i, 0, 1] * img_pnts_1[i, 0, 1],
                       f_0 * img_pnts_0[i, 0, 1],
                       f_0 * img_pnts_1[i, 0, 0],
                       f_0 * img_pnts_1[i, 0, 1],
                       f_0 ** 2]])
        xi_sum += np.dot(xi.T, xi)

        # TODO: Add V0_xi
        V0_xi = np.array([[]])

    M = xi_sum / len(img_pnts_0)
    w, v = np.linalg.eig(M)
    theta = v[:, np.argmin(w)]
    reshaped = np.reshape(theta, (3, 3))

    return normalize_F_matrix(reshaped)


def prepare_test_data():
    rot_mat_0 = euler_angle_to_rot_mat(0, 0, 0)
    # print(rot_mat_0)
    T_0_in_camera_coord = (0, 0, 10)
    trans_vec_0 = np.eye(3) * np.matrix(T_0_in_camera_coord).T
    rot_mat_1 = euler_angle_to_rot_mat(0, 45, 0)
    T_1_in_camera_coord = (0, 0, 10)
    trans_vec_1 = np.eye(3) * np.matrix(T_1_in_camera_coord).T
    points = create_curve_surface_points(5, 5, 0.2)
    # print(points)
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)
    # print(rodri_0)
    # print(rodri_1)

    f = 160
    width = 640
    height = 480
    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(np.array([[f, 0, pp[0]],
                            [0, f, pp[1]],
                            [0, 0, 1]], dtype = "double"))
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(points, rodri_0, T_0_in_camera_coord, camera_matrix, dist_coeffs)
    img_pnts_1, jac = cv2.projectPoints(points, rodri_1, T_1_in_camera_coord, camera_matrix, dist_coeffs)

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    # print(img_pnts_0)
    # input()
    for pnt in img_pnts_0:
        cv2.circle(img_0, (int(pnt[0][0]), int(pnt[0][1])), 3, (0, 0, 0), -1)
    for pnt in img_pnts_1:
        cv2.circle(img_1, (int(pnt[0][0]), int(pnt[0][1])), 3, (255, 0, 0), -1)

    F_true_1_to_2 = calculate_true_fundamental_matrix(rot_mat_0, rot_mat_1, T_0_in_camera_coord, T_1_in_camera_coord, camera_matrix)

    F, mask = cv2.findFundamentalMat(img_pnts_0, img_pnts_1, cv2.FM_LMEDS)
    print("F: ", F)

    F_by_least_squares = calculate_f_matrix_by_least_squares(img_pnts_0, img_pnts_1)
    lines_CAM1 = cv2.computeCorrespondEpilines(img_pnts_1, 2, F)
    lines_CAM1 = lines_CAM1.reshape(-1,3) #行列の変形

    width_CAM1 = img_0.shape[1] #画像幅

    # imgCAM1 = cv2.resize(img_0, None, fx = 0.5, fy = 0.5)
    for lines in lines_CAM1:
        # print(lines)
        x0,y0 = map(int, [0,-lines[2]/lines[1]]) #左端
        x1,y1 = map(int, [width_CAM1,-(lines[2]+lines[0]*width_CAM1)/lines[1]]) #右端

        # print(x0, y0)
        # print(x1, y1)
        img_0 = cv2.line(img_0, (x0,y0), (x1,y1), (0, 255, 0), 1) #線の描画

    cv2.imshow("EPI", img_0)

    # cv2.imshow("CAM0", cv2.resize(img_0, None, fx = 0.5, fy = 0.5))
    # cv2.imshow("CAM1", cv2.resize(img_1, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)

    return img_pnts_0, img_pnts_1, F_true_1_to_2


def main():
    img_pnts_0, img_pnts_1, F_true = prepare_test_data()
    print("F_true: ", F_true)
    F_by_least_squares = calculate_f_matrix_by_least_squares(img_pnts_0, img_pnts_1)
    print("F_by_least_squares: ", F_by_least_squares)
    calculate_f_matrix_by_taubin(img_pnts_0, img_pnts_1)


if __name__ == '__main__':
    main()
