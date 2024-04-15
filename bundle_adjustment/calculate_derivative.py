import numpy as np

from utils import (
    calculate_antisymmetric_matrix,
    calculate_rows_of_dot_between_camera_mat_and_3d_position,
)


def calculate_derivative_of_reprojection_error(
    p, q, r, p_deriv, q_deriv, r_deriv, x, y, f_0
):
    deriv = (
        2
        / r**2
        * (
            (p / r - x / f_0) * (r * p_deriv - p * r_deriv)
            + (q / r - y / f_0) * (r * q_deriv - q * r_deriv)
        )
    )

    return deriv


def calculate_3d_position_derivative(Ps, points_2d, points_3d, f_0, first_deriv):
    for point_idx in range(len(points_2d)):
        x_deriv_sum = 0
        y_deriv_sum = 0
        z_deriv_sum = 0
        for camera_idx in range(Ps.shape[0]):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue
            P = Ps[camera_idx]
            p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
                P, points_3d["points_3d"][point_idx]
            )

            # Differential with respect to 3D position X
            p_deriv_X = P[0][0]
            q_deriv_X = P[1][0]
            r_deriv_X = P[2][0]
            x_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_X, q_deriv_X, r_deriv_X, x, y, f_0
            )

            # Differential with respect to 3D position Y
            p_deriv_Y = P[0][1]
            q_deriv_Y = P[1][1]
            r_deriv_Y = P[2][1]
            y_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_Y, q_deriv_Y, r_deriv_Y, x, y, f_0
            )

            # Differential with respect to 3D position Z
            p_deriv_Z = P[0][2]
            q_deriv_Z = P[1][2]
            r_deriv_Z = P[2][2]
            z_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_Z, q_deriv_Z, r_deriv_Z, x, y, f_0
            )

        first_deriv[3 * point_idx] = x_deriv_sum
        first_deriv[3 * point_idx + 1] = y_deriv_sum
        first_deriv[3 * point_idx + 2] = z_deriv_sum


def calculate_focal_length_derivative(
    Ps, Ks, points_2d, points_3d, f_0, first_deriv, start_pos
):
    for camera_idx in range(Ps.shape[0]):
        deriv_sum = 0
        P = Ps[camera_idx]
        K = Ks[camera_idx]
        points = points_3d["points_3d"]
        for point_idx in range(len(points)):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue

            p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
                P, points_3d["points_3d"][point_idx]
            )
            f = K[0][0]
            u = K[0][2]
            v = K[1][2]
            p_deriv = (p - u / f_0 * r) / f
            q_deriv = (q - v / f_0 * r) / f
            r_deriv = 0
            deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv, q_deriv, r_deriv, x, y, f_0
            )
        first_deriv[start_pos + camera_idx] = deriv_sum


def calculate_optical_axis_point_derivative(
    Ps, points_2d, points_3d, f_0, first_deriv, start_pos
):
    for camera_idx in range(Ps.shape[0]):
        u_deriv_sum = 0
        v_deriv_sum = 0
        P = Ps[camera_idx]
        points = points_3d["points_3d"]
        for point_idx in range(len(points)):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue

            p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
                P, points_3d["points_3d"][point_idx]
            )

            p_deriv_u = r / f_0
            q_deriv_u = 0
            r_deriv_u = 0
            u_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_u, q_deriv_u, r_deriv_u, x, y, f_0
            )

            p_deriv_v = 0
            q_deriv_v = r / f_0
            r_deriv_v = 0
            v_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_v, q_deriv_v, r_deriv_v, x, y, f_0
            )

        first_deriv[start_pos + camera_idx * 2] = u_deriv_sum
        first_deriv[start_pos + camera_idx * 2 + 1] = v_deriv_sum


def calculate_translation_derivative(
    Ps, Ks, Rs, points_2d, points_3d, f_0, first_deriv, start_pos
):
    for camera_idx in range(Ps.shape[0]):
        if camera_idx == 0:
            continue
        t1_deriv_sum = 0
        t2_deriv_sum = 0
        t3_deriv_sum = 0
        P = Ps[camera_idx]
        K = Ks[camera_idx]
        R = Rs[camera_idx]
        points = points_3d["points_3d"]
        for point_idx in range(len(points)):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue

            p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
                P, points_3d["points_3d"][point_idx]
            )
            f = K[0][0]
            u = K[0][2]
            v = K[1][2]
            r1 = np.array([R[0][0], R[1][0], R[2][0]])
            r2 = np.array([R[0][1], R[1][1], R[2][1]])
            r3 = np.array([R[0][2], R[1][2], R[2][2]])

            p_derivs = -(np.dot(f, r1) + np.dot(u, r3))
            q_derivs = -(np.dot(f, r2) + np.dot(v, r3))
            r_derivs = -f_0 * r3

            p_deriv_t1 = p_derivs[0]
            q_deriv_t1 = q_derivs[0]
            r_deriv_t1 = r_derivs[0]
            t1_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_t1, q_deriv_t1, r_deriv_t1, x, y, f_0
            )

            p_deriv_t2 = p_derivs[1]
            q_deriv_t2 = q_derivs[1]
            r_deriv_t2 = r_derivs[1]
            t2_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_t2, q_deriv_t2, r_deriv_t2, x, y, f_0
            )

            p_deriv_t3 = p_derivs[2]
            q_deriv_t3 = q_derivs[2]
            r_deriv_t3 = r_derivs[2]
            t3_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_t3, q_deriv_t3, r_deriv_t3, x, y, f_0
            )

        if camera_idx == 1:
            first_deriv[start_pos + (camera_idx - 1) * 3] = t1_deriv_sum
            first_deriv[start_pos + (camera_idx - 1) * 3 + 1] = t2_deriv_sum
        else:
            first_deriv[start_pos + (camera_idx - 1) * 3 - 1] = t1_deriv_sum
            first_deriv[start_pos + (camera_idx - 1) * 3] = t2_deriv_sum
            first_deriv[start_pos + (camera_idx - 1) * 3 + 1] = t3_deriv_sum


def calculate_rotation_derivative(
    Ps, Ks, Rs, ts, points_2d, points_3d, f_0, first_deriv, start_pos
):
    for camera_idx in range(Ps.shape[0]):
        if camera_idx == 0:
            continue
        w1_deriv_sum = 0
        w2_deriv_sum = 0
        w3_deriv_sum = 0
        P = Ps[camera_idx]
        K = Ks[camera_idx]
        R = Rs[camera_idx]
        t = ts[camera_idx]
        points = points_3d["points_3d"]
        for point_idx in range(len(points)):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue

            p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
                P, points_3d["points_3d"][point_idx]
            )
            f = K[0][0]
            u = K[0][2]
            v = K[1][2]
            r1 = np.array([R[0][0], R[1][0], R[2][0]])
            r2 = np.array([R[0][1], R[1][1], R[2][1]])
            r3 = np.array([R[0][2], R[1][2], R[2][2]])

            p_derivs = np.dot(
                calculate_antisymmetric_matrix(np.dot(f, r1) + np.dot(u, r3)),
                (points[point_idx] - t),
            )
            q_derivs = np.dot(
                calculate_antisymmetric_matrix(np.dot(f, r2) + np.dot(v, r3)),
                (points[point_idx] - t),
            )
            r_derivs = np.dot(
                calculate_antisymmetric_matrix(f_0 * r3), (points[point_idx] - t)
            )

            p_deriv_w1 = p_derivs[0]
            q_deriv_w1 = q_derivs[0]
            r_deriv_w1 = r_derivs[0]
            w1_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_w1, q_deriv_w1, r_deriv_w1, x, y, f_0
            )

            p_deriv_w2 = p_derivs[1]
            q_deriv_w2 = q_derivs[1]
            r_deriv_w2 = r_derivs[1]
            w2_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_w2, q_deriv_w2, r_deriv_w2, x, y, f_0
            )

            p_deriv_w3 = p_derivs[2]
            q_deriv_w3 = q_derivs[2]
            r_deriv_w3 = r_derivs[2]
            w3_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_w3, q_deriv_w3, r_deriv_w3, x, y, f_0
            )

        first_deriv[start_pos + (camera_idx - 1) * 3] = w1_deriv_sum
        first_deriv[start_pos + (camera_idx - 1) * 3 + 1] = w2_deriv_sum
        first_deriv[start_pos + (camera_idx - 1) * 3 + 2] = w3_deriv_sum
