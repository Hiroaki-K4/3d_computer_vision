import numpy as np
import math
import cv2
import scipy


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

    normalize_F_matrix = F_matrix / factor_sum ** 0.5

    return normalize_F_matrix


def calculate_true_fundamental_matrix(rot_mat_before, rot_mat_after, T_in_camera_coord_before, T_in_camera_coord_after, camera_matrix):
    rot_1_to_2 = np.dot(rot_mat_after, rot_mat_before.T)
    trans_1_to_2_in_camera_coord = np.matrix(T_in_camera_coord_after).T - rot_1_to_2 * np.matrix(T_in_camera_coord_before).T
    trans_1_to_2_in_camera_coord_outer = create_outer_product(trans_1_to_2_in_camera_coord)
    A_inv = np.linalg.inv(camera_matrix)
    F_true_1_to_2 = A_inv.T * trans_1_to_2_in_camera_coord_outer * rot_1_to_2 * A_inv

    return normalize_F_matrix(F_true_1_to_2), rot_1_to_2, trans_1_to_2_in_camera_coord


def calculate_f_matrix_by_least_squares(img_pnts_0, img_pnts_1):
    if len(img_pnts_0) != len(img_pnts_1):
        raise RuntimeError("The number points is wrong.")

    f_x = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_y = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_0 = max(f_x, f_y)

    xi_sum = np.zeros((9, 9))
    for i in range(len(img_pnts_0)):
        x_0 = img_pnts_0[i, 0, 0]
        y_0 = img_pnts_0[i, 0, 1]
        x_1 = img_pnts_1[i, 0, 0]
        y_1 = img_pnts_1[i, 0, 1]
        xi = np.array([[x_0 * x_1,
                       x_0 * y_1,
                       f_0 * x_0,
                       y_0 * x_1,
                       y_0 * y_1,
                       f_0 * y_0,
                       f_0 * x_1,
                       f_0 * y_1,
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

    xi_sum = np.zeros((9, 9))
    V0_xi_sum = np.zeros((9, 9))
    for i in range(len(img_pnts_0)):
        x_0 = img_pnts_0[i, 0, 0]
        y_0 = img_pnts_0[i, 0, 1]
        x_1 = img_pnts_1[i, 0, 0]
        y_1 = img_pnts_1[i, 0, 1]
        xi = np.array([[x_0 * x_1,
                       x_0 * y_1,
                       f_0 * x_0,
                       y_0 * x_1,
                       y_0 * y_1,
                       f_0 * y_0,
                       f_0 * x_1,
                       f_0 * y_1,
                       f_0 ** 2]])
        xi_sum += np.dot(xi.T, xi)

        V0_xi = np.array([[x_0**2+x_1**2, x_1*y_1, f_0*x_1, x_0*y_0, 0, 0, f_0*x_0, 0, 0],
                        [x_1*y_1, x_0**2+y_1**2, f_0*y_1, 0, x_0*y_0, 0, 0, f_0*x_0, 0],
                        [f_0*x_1, f_0*y_1, f_0**2, 0, 0, 0, 0, 0, 0],
                        [x_0*y_0, 0, 0, y_0**2+x_1**2, x_1*y_1, f_0*x_1, f_0*y_0, 0, 0],
                        [0, x_0*y_0, 0, x_1*y_1, y_0**2+y_1**2, f_0*y_1, 0, f_0*y_0, 0],
                        [0, 0, 0, f_0*x_1, f_0*y_1, f_0**2, 0, 0, 0],
                        [f_0*x_0, 0, 0, f_0*y_0, 0, 0, f_0**2, 0, 0],
                        [0, f_0*x_0, 0, 0, f_0*y_0, 0, 0, f_0**2, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        V0_xi_sum += V0_xi

    M = xi_sum / len(img_pnts_0)
    N = V0_xi_sum / len(img_pnts_0)
    eig_val, eig_vec = scipy.linalg.eig(N, M)
    theta = eig_vec[:, np.argmax(eig_val)]
    reshaped = np.reshape(theta, (3, 3))

    return normalize_F_matrix(reshaped)


def calculate_f_matrix_by_fns(img_pnts_0, img_pnts_1):
    f_x = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_y = max(max(img_pnts_0[:, 0, 0]), max(img_pnts_1[:, 0, 1]))
    f_0 = max(f_x, f_y)

    W = np.ones(len(img_pnts_0), dtype="float64")
    theta_zero = np.zeros(9, dtype="float64")
    diff = 10000.0

    while diff > 1e-10:
        xi_sum = np.zeros((9, 9))
        L_sum = np.zeros((9, 9))
        V0_xi_list = []
        for i in range(len(img_pnts_0)):
            x_0 = img_pnts_0[i, 0, 0]
            y_0 = img_pnts_0[i, 0, 1]
            x_1 = img_pnts_1[i, 0, 0]
            y_1 = img_pnts_1[i, 0, 1]
            xi = np.array([[x_0 * x_1,
                        x_0 * y_1,
                        f_0 * x_0,
                        y_0 * x_1,
                        y_0 * y_1,
                        f_0 * y_0,
                        f_0 * x_1,
                        f_0 * y_1,
                        f_0 ** 2]])
            xi_sum += np.dot(W[i], np.dot(xi.T, xi))
            V0_xi = np.array([[x_0**2+x_1**2, x_1*y_1, f_0*x_1, x_0*y_0, 0, 0, f_0*x_0, 0, 0],
                            [x_1*y_1, x_0**2+y_1**2, f_0*y_1, 0, x_0*y_0, 0, 0, f_0*x_0, 0],
                            [f_0*x_1, f_0*y_1, f_0**2, 0, 0, 0, 0, 0, 0],
                            [x_0*y_0, 0, 0, y_0**2+x_1**2, x_1*y_1, f_0*x_1, f_0*y_0, 0, 0],
                            [0, x_0*y_0, 0, x_1*y_1, y_0**2+y_1**2, f_0*y_1, 0, f_0*y_0, 0],
                            [0, 0, 0, f_0*x_1, f_0*y_1, f_0**2, 0, 0, 0],
                            [f_0*x_0, 0, 0, f_0*y_0, 0, 0, f_0**2, 0, 0],
                            [0, f_0*x_0, 0, 0, f_0*y_0, 0, 0, f_0**2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
            V0_xi_list.append(V0_xi)
            L_sum += np.dot(np.dot(W[i]**2, np.dot(xi.T[:, 0], theta_zero)**2), V0_xi)

        M = xi_sum / len(img_pnts_0)
        L = L_sum / len(img_pnts_0)
        X = M - L
        eig_val, eig_vec = np.linalg.eig(X)
        theta = eig_vec[:, np.argmin(eig_val)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        if diff <= 1e-10:
            break

        for i in range(len(img_pnts_0)):
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi_list[i], theta))))
        theta_zero = theta

    reshaped = np.reshape(theta, (3, 3))

    return normalize_F_matrix(reshaped)


def draw_epipolar_lines(img_0, img_pnts_0, img_pnts_1):
    F, mask = cv2.findFundamentalMat(img_pnts_0, img_pnts_1, cv2.FM_LMEDS)

    lines_CAM1 = cv2.computeCorrespondEpilines(img_pnts_1, 2, F)
    lines_CAM1 = lines_CAM1.reshape(-1,3)
    width_CAM1 = img_0.shape[1]
    for lines in lines_CAM1:
        x0,y0 = map(int, [0,-lines[2]/lines[1]])
        x1,y1 = map(int, [width_CAM1,-(lines[2]+lines[0]*width_CAM1)/lines[1]])
        img_0 = cv2.line(img_0, (x0,y0), (x1,y1), (0, 255, 0), 1)

    cv2.imshow("EPI", img_0)
    cv2.waitKey(0)


def add_noise(img_pnts, noise_scale):
    noise = np.random.normal(0, noise_scale, img_pnts.shape)
    noised_points = img_pnts + noise

    return noised_points


def prepare_test_data(draw_test_data, draw_epipolar):
    rot_mat_0 = euler_angle_to_rot_mat(0, 0, 0)
    T_0_in_camera_coord = (0, 0, 10)
    trans_vec_0 = np.eye(3) * np.matrix(T_0_in_camera_coord).T
    rot_mat_1 = euler_angle_to_rot_mat(0, 45, 0)
    T_1_in_camera_coord = (0, 0, 10)
    trans_vec_1 = np.eye(3) * np.matrix(T_1_in_camera_coord).T
    points = create_curve_surface_points(5, 5, 0.2)
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)

    f = 160
    width = 640
    height = 480
    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(np.array([[f, 0, pp[0]],
                            [0, f, pp[1]],
                            [0, 0, 1]], dtype = "double"))
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(points, rodri_0, trans_vec_0, camera_matrix, dist_coeffs)
    img_pnts_1, jac = cv2.projectPoints(points, rodri_1, trans_vec_1, camera_matrix, dist_coeffs)

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)

    noise_scale = 0.2
    noised_img_pnts_0 = add_noise(img_pnts_0, noise_scale)
    noised_img_pnts_1 = add_noise(img_pnts_1, noise_scale)

    for pnt in noised_img_pnts_0:
        cv2.circle(img_0, (int(pnt[0][0]), int(pnt[0][1])), 3, (0, 0, 0), -1)
    for pnt in noised_img_pnts_1:
        cv2.circle(img_1, (int(pnt[0][0]), int(pnt[0][1])), 3, (255, 0, 0), -1)

    F_true_1_to_2, rot_1_to_2, trans_1_to_2_in_camera_coord = calculate_true_fundamental_matrix(rot_mat_0, rot_mat_1, T_0_in_camera_coord, T_1_in_camera_coord, camera_matrix)

    if draw_epipolar:
        draw_epipolar_lines(img_0, noised_img_pnts_0, noised_img_pnts_1)

    if draw_test_data:
        cv2.imshow("CAM0", cv2.resize(img_0, None, fx = 0.5, fy = 0.5))
        cv2.imshow("CAM1", cv2.resize(img_1, None, fx = 0.5, fy = 0.5))
        cv2.waitKey(0)

    return img_pnts_0, img_pnts_1, noised_img_pnts_0, noised_img_pnts_1, F_true_1_to_2, rot_1_to_2, trans_1_to_2_in_camera_coord


def rank_postcorrection_method(F):
    U, S, Vt = np.linalg.svd(F)
    S = np.array([S[0]/np.sqrt(S[0]**2+S[1]**2), S[1]/np.sqrt(S[0]**2+S[1]**2), 0])
    F_ans = np.dot(np.dot(U, np.diag(S)), Vt)

    return F_ans


def calculate_f_matrix_diff(F_true, F_est):
    f_sum = 0
    for i in range(F_true.shape[0]):
        for j in range(F_true.shape[1]):
            f_sum += (F_true[i, j] - F_est[i, j]) ** 2

    return math.sqrt(f_sum)


def main():
    draw_test_data = False
    draw_epipolar = True
    img_pnts_0, img_pnts_1, noised_img_pnts_0, noised_img_pnts_1, F_true, rot_1_to_2, trans_1_to_2_in_camera_coord = prepare_test_data(draw_test_data, draw_epipolar)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_true")
    print(F_true)

    F_by_least_squares = rank_postcorrection_method(calculate_f_matrix_by_least_squares(noised_img_pnts_0, noised_img_pnts_1))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_by_least_squares")
    print(F_by_least_squares)

    F_by_taubin = rank_postcorrection_method(calculate_f_matrix_by_taubin(noised_img_pnts_0, noised_img_pnts_1))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_by_taubin")
    print(F_by_taubin)

    F_by_fns = rank_postcorrection_method(calculate_f_matrix_by_fns(noised_img_pnts_0, noised_img_pnts_1))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_by_fns")
    print(F_by_fns)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    F_true_vs_F_least_squares = calculate_f_matrix_diff(F_true, F_by_least_squares)
    print("F_true_vs_F_least_squares: ", F_true_vs_F_least_squares)
    F_true_vs_F_by_taubin = calculate_f_matrix_diff(F_true, F_by_taubin)
    print("F_true_vs_F_by_taubin: ", F_true_vs_F_by_taubin)
    F_true_vs_F_by_fns = calculate_f_matrix_diff(F_true, F_by_fns)
    print("F_true_vs_F_by_fns: ", F_true_vs_F_by_fns)


if __name__ == '__main__':
    main()
