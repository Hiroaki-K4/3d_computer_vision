import numpy as np


def calculate_3d_position_derivative(Ps, points_2d, first_deriv):
    for point_idx in range(len(points_2d)):
        for camera_idx in range(Ps.shape[0]):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue
            P = Ps[camera_idx]
            first_deriv[3 * point_idx] = np.array([P[0][0], P[1][0], P[2][0]])
            first_deriv[3 * point_idx + 1] = np.array([P[0][1], P[1][1], P[2][1]])
            first_deriv[3 * point_idx + 2] = np.array([P[0][2], P[1][2], P[2][2]])


def calculate_focal_length_derivative(P, K, first_deriv):
    # TODO Add derivative of focal length
    print()
