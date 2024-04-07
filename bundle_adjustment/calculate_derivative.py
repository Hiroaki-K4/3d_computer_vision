import numpy as np

from utils import calculate_rows_of_dot_between_camera_mat_and_3d_position


def calculate_3d_coordinate_derivative(p, q, r, p_deriv, q_deriv, r_deriv, x, y, f_0):
    deriv = (
        2
        / r**2
        * (
            (p / r - x / f_0) * (r * p_deriv - p * r_deriv)
            + (q / r - y / f_0) * (r * q_deriv - q * r_deriv)
        )
    )

    return deriv


def calculate_3d_position_derivative(Ps, points_2d, points_3d, first_deriv, f_0):
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
            x_deriv_sum += calculate_3d_coordinate_derivative(
                p, q, r, p_deriv_X, q_deriv_X, r_deriv_X, x, y, f_0
            )

            # Differential with respect to 3D position Y
            p_deriv_Y = P[0][1]
            q_deriv_Y = P[1][1]
            r_deriv_Y = P[2][1]
            y_deriv_sum += calculate_3d_coordinate_derivative(
                p, q, r, p_deriv_Y, q_deriv_Y, r_deriv_Y, x, y, f_0
            )

            # Differential with respect to 3D position Z
            p_deriv_Z = P[0][2]
            q_deriv_Z = P[1][2]
            r_deriv_Z = P[2][2]
            z_deriv_sum += calculate_3d_coordinate_derivative(
                p, q, r, p_deriv_Z, q_deriv_Z, r_deriv_Z, x, y, f_0
            )

        first_deriv[3 * point_idx] = x_deriv_sum
        first_deriv[3 * point_idx + 1] = y_deriv_sum
        first_deriv[3 * point_idx + 2] = z_deriv_sum


def calculate_focal_length_derivative(P, K, first_deriv):
    # TODO Add derivative of focal length
    print()
