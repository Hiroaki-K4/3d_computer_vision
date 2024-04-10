import csv
import json

import numpy as np

import calculate_derivative as deriv


def normalize_camera_params(R, t):
    norm_R = np.zeros(R.shape)
    norm_t = np.zeros(t.shape)
    for i in range(R.shape[0]):
        norm_R[i] = np.dot(R[i], np.linalg.inv(R[0]))
        if i == 0:
            norm_t[i] = np.array([0, 0, 0])
        else:
            norm_t[i] = t[i] / t[1][2]

    return norm_R, norm_t


def split_camera_params(camera_params):
    K = np.zeros((len(camera_params), 3, 3))
    R = np.zeros((len(camera_params), 3, 3))
    t = np.zeros((len(camera_params), 3))
    for i in range(len(camera_params)):
        K[i] = np.array(camera_params[i]["K"])
        R[i] = np.array(camera_params[i]["R"])
        t[i] = np.array(camera_params[i]["t"])

    return K, R, t


def calculate_reprojection_error(Ps, points_2d, points_3d, f_0):
    E = 0
    for point_idx in range(len(points_2d)):
        for camera_idx in range(Ps.shape[0]):
            x = float(points_2d[point_idx][camera_idx * 2])
            y = float(points_2d[point_idx][camera_idx * 2 + 1])
            if x == -1 or y == -1:
                continue
            P = Ps[camera_idx]
            X = points_3d["points_3d"][point_idx][0]
            Y = points_3d["points_3d"][point_idx][1]
            Z = points_3d["points_3d"][point_idx][2]
            E += (
                x / f_0
                - (P[0][0] * X + P[0][1] * Y + P[0][2] * Z + P[0][3])
                / (P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3])
            ) ** 2 + (
                y / f_0
                - (P[1][0] * X + P[1][1] * Y + P[1][2] * Z + P[1][3])
                / (P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3])
            ) ** 2
            # print(
            #     x,
            #     (P[0][0] * X + P[0][1] * Y + P[0][2] * Z + P[0][3])
            #     / (P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3]),
            # )
            # print(
            #     y,
            #     (P[1][0] * X + P[1][1] * Y + P[1][2] * Z + P[1][3])
            #     / (P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3]),
            # )

    return E


def calculate_first_order_derivative(K, R, t, P, points_3d, points_2d, f_0):
    # N: number of points, M: number of images
    # Order: 3D position(3N), focal length(M), optical axis point(2M), translation(3M), rotation(3M)
    # Number of derivatives: 3N+9M-7
    # -7: R1=I, t1=0, t22=1

    deriv_num = 3 * len(points_3d["points_3d"]) + 9 * K.shape[0] - 7
    first_deriv = np.zeros(deriv_num)
    deriv.calculate_3d_position_derivative(P, points_2d, points_3d, f_0, first_deriv)
    print("first_deriv: ", first_deriv)
    deriv.calculate_focal_length_derivative(P, K, points_2d, points_3d, f_0, first_deriv)


def calculate_camera_matrix(K, R, t):
    P = np.zeros((K.shape[0], 3, 4))
    for camera_idx in range(K.shape[0]):
        t_mat = np.zeros((3, 4))
        t_mat[:, :3] = np.identity(3)
        t_mat[:, 3] = t[camera_idx]
        P[camera_idx] = np.dot(np.dot(K[camera_idx], R[camera_idx].T), -t_mat)

    return P


def run_bundle_adjustment(K, R, t, points_2d, points_3d, f_0):
    P = calculate_camera_matrix(K, R, t)
    E = calculate_reprojection_error(P, points_2d, points_3d, f_0)
    print("E: ", E)
    c = 0.0001
    calculate_first_order_derivative(K, R, t, P, points_3d, points_2d, f_0)


def main(camera_parameters_file, tracked_2d_points_file, tracked_3d_points_file):
    with open(camera_parameters_file) as f:
        camera_params = json.load(f)

    with open(tracked_2d_points_file) as f:
        reader = csv.reader(f, delimiter=" ")
        points_2d = [row for row in reader]

    with open(tracked_3d_points_file) as f:
        points_3d = json.load(f)

    K, R, t = split_camera_params(camera_params)
    R, t = normalize_camera_params(R, t)
    f_0 = 400
    run_bundle_adjustment(K, R, t, points_2d, points_3d, f_0)


if __name__ == "__main__":
    camera_parameters_file = "camera_parameters.json"
    tracked_2d_points_file = "2d_tracked_points.csv"
    tracked_3d_points_file = "3d_tracked_points.json"
    main(camera_parameters_file, tracked_2d_points_file, tracked_3d_points_file)
