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


def calculate_second_derivative_of_reprojection_error(
    p, q, r, p_deriv_0, q_deriv_0, r_deriv_0, p_deriv_1, q_deriv_1, r_deriv_1
):
    deriv = (
        2
        / r**4
        * (
            (r * p_deriv_0 - p * r_deriv_0) * (r * p_deriv_1 - p * r_deriv_1)
            + (r * q_deriv_0 - q * r_deriv_0) * (r * q_deriv_1 - q * r_deriv_1)
        )
    )

    return deriv


def calculate_3d_position_x_derivative(P):
    p_deriv = P[0][0]
    q_deriv = P[1][0]
    r_deriv = P[2][0]

    return p_deriv, q_deriv, r_deriv


def calculate_3d_position_y_derivative(P):
    p_deriv = P[0][1]
    q_deriv = P[1][1]
    r_deriv = P[2][1]

    return p_deriv, q_deriv, r_deriv


def calculate_3d_position_z_derivative(P):
    p_deriv = P[0][2]
    q_deriv = P[1][2]
    r_deriv = P[2][2]

    return p_deriv, q_deriv, r_deriv


def calculate_focal_length_derivative(K, p, q, r, f_0):
    f = K[0][0]
    u = K[0][2]
    v = K[1][2]
    p_deriv = (p - u / f_0 * r) / f
    q_deriv = (q - v / f_0 * r) / f
    r_deriv = 0

    return p_deriv, q_deriv, r_deriv


def calculate_optical_axis_point_u_derivative(r, f_0):
    p_deriv = r / f_0
    q_deriv = 0
    r_deriv = 0

    return p_deriv, q_deriv, r_deriv


def calculate_optical_axis_point_v_derivative(r, f_0):
    p_deriv = 0
    q_deriv = r / f_0
    r_deriv = 0

    return p_deriv, q_deriv, r_deriv


def calculate_translation_derivative(K, R, f_0):
    f = K[0][0]
    u = K[0][2]
    v = K[1][2]
    r1 = np.array([R[0][0], R[1][0], R[2][0]])
    r2 = np.array([R[0][1], R[1][1], R[2][1]])
    r3 = np.array([R[0][2], R[1][2], R[2][2]])

    p_derivs = -(np.dot(f, r1) + np.dot(u, r3))
    q_derivs = -(np.dot(f, r2) + np.dot(v, r3))
    r_derivs = -f_0 * r3

    return p_derivs, q_derivs, r_derivs


def calculate_rotation_derivative(K, R, t, f_0, points, point_idx):
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
    r_derivs = np.dot(calculate_antisymmetric_matrix(f_0 * r3), (points[point_idx] - t))

    return p_derivs, q_derivs, r_derivs


def calculate_3d_position_derivative_of_reprojection_error(
    Ps, points_2d, points_3d, f_0, first_deriv
):
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
            p_deriv_X, q_deriv_X, r_deriv_X = calculate_3d_position_x_derivative(P)
            x_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_X, q_deriv_X, r_deriv_X, x, y, f_0
            )

            # Differential with respect to 3D position Y
            p_deriv_Y, q_deriv_Y, r_deriv_Y = calculate_3d_position_y_derivative(P)
            y_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_Y, q_deriv_Y, r_deriv_Y, x, y, f_0
            )

            # Differential with respect to 3D position Z
            p_deriv_Z, q_deriv_Z, r_deriv_Z = calculate_3d_position_z_derivative(P)
            z_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_Z, q_deriv_Z, r_deriv_Z, x, y, f_0
            )

        first_deriv[3 * point_idx] = x_deriv_sum
        first_deriv[3 * point_idx + 1] = y_deriv_sum
        first_deriv[3 * point_idx + 2] = z_deriv_sum


def calculate_focal_length_derivative_of_reprojection_error(
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
            p_deriv, q_deriv, r_deriv = calculate_focal_length_derivative(
                K, p, q, r, f_0
            )
            deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv, q_deriv, r_deriv, x, y, f_0
            )

        first_deriv[start_pos + camera_idx] = deriv_sum


def calculate_optical_axis_point_derivative_of_reprojection_error(
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

            p_deriv_u, q_deriv_u, r_deriv_u = calculate_optical_axis_point_u_derivative(
                r, f_0
            )
            u_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_u, q_deriv_u, r_deriv_u, x, y, f_0
            )

            p_deriv_v, q_deriv_v, r_deriv_v = calculate_optical_axis_point_v_derivative(
                r, f_0
            )
            v_deriv_sum += calculate_derivative_of_reprojection_error(
                p, q, r, p_deriv_v, q_deriv_v, r_deriv_v, x, y, f_0
            )

        first_deriv[start_pos + camera_idx * 2] = u_deriv_sum
        first_deriv[start_pos + camera_idx * 2 + 1] = v_deriv_sum


def calculate_translation_derivative_of_reprojection_error(
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
            p_derivs, q_derivs, r_derivs = calculate_translation_derivative(K, R, f_0)

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


def calculate_rotation_derivative_of_reprojection_error(
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
            p_derivs, q_derivs, r_derivs = calculate_rotation_derivative(
                K, R, t, f_0, points, point_idx
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


def calculate_second_derivative_about_point(
    point_idx_0, point_idx_1, Ps, points_2d, points_3d
):
    deriv = 0
    point_idx = int(point_idx_0 / 3)
    for camera_idx in range(Ps.shape[0]):
        x = float(points_2d[point_idx][camera_idx * 2])
        y = float(points_2d[point_idx][camera_idx * 2 + 1])
        if x == -1 or y == -1:
            continue
        P = Ps[camera_idx]
        p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
            P, points_3d["points_3d"][point_idx]
        )

        if point_idx_0 % 3 == 0:
            p_deriv_0, q_deriv_0, r_deriv_0 = calculate_3d_position_x_derivative(P)
        elif point_idx_0 % 3 == 1:
            p_deriv_0, q_deriv_0, r_deriv_0 = calculate_3d_position_y_derivative(P)
        elif point_idx_0 % 3 == 2:
            p_deriv_0, q_deriv_0, r_deriv_0 = calculate_3d_position_z_derivative(P)
        if point_idx_1 % 3 == 0:
            p_deriv_1, q_deriv_1, r_deriv_1 = calculate_3d_position_x_derivative(P)
        elif point_idx_1 % 3 == 1:
            p_deriv_1, q_deriv_1, r_deriv_1 = calculate_3d_position_y_derivative(P)
        elif point_idx_1 % 3 == 2:
            p_deriv_1, q_deriv_1, r_deriv_1 = calculate_3d_position_z_derivative(P)

        deriv += calculate_second_derivative_of_reprojection_error(
            p, q, r, p_deriv_0, q_deriv_0, r_deriv_0, p_deriv_1, q_deriv_1, r_deriv_1
        )

    return deriv


def extract_camera_idx(
    point_idx,
    focal_length_range,
    optimal_axis_point_range,
    translation_range,
    rotation_range,
    target_types,
):
    target_type = None
    if point_idx >= focal_length_range[0] and point_idx <= focal_length_range[1]:
        camera_idx = point_idx - focal_length_range[0]
        target_type = target_types[0]
    elif (
        point_idx >= optimal_axis_point_range[0]
        and point_idx <= optimal_axis_point_range[1]
    ):
        camera_idx = int((point_idx - optimal_axis_point_range[0]) / 2)
        target_type = target_types[1]
    elif point_idx >= translation_range[0] and point_idx <= translation_range[1]:
        camera_idx = int((point_idx - translation_range[0]) / 3)
        target_type = target_types[2]
    elif point_idx >= rotation_range[0] and point_idx <= rotation_range[1]:
        camera_idx = int((point_idx - rotation_range[0]) / 3)
        target_type = target_types[3]

    return camera_idx, target_type


def extract_derivative(point_idx, camera_idx, target_type, target_types):
    print("extract_derivative")
    # TODO Update this function


def calculate_second_derivative_about_image(
    point_idx_0,
    point_idx_1,
    Ps,
    points_2d,
    points_3d,
    f_0,
    focal_length_range,
    optimal_axis_point_range,
    translation_range,
    rotation_range,
):
    target_types = ["focal", "opt", "trans", "rot"]
    camera_idx_0, target_type_0 = extract_camera_idx(
        point_idx_0,
        focal_length_range,
        optimal_axis_point_range,
        translation_range,
        rotation_range,
        target_types,
    )
    camera_idx_1, target_type_1 = extract_camera_idx(
        point_idx_1,
        focal_length_range,
        optimal_axis_point_range,
        translation_range,
        rotation_range,
        target_types,
    )
    if camera_idx_0 != camera_idx_1:
        # Not same frame
        return 0

    print(
        target_type_0,
        target_type_1,
        camera_idx_0,
        camera_idx_1,
        point_idx_0,
        point_idx_1,
    )
    points = points_3d["points_3d"]
    P = Ps[camera_idx_0]
    for point_idx in range(len(points)):
        x = float(points_2d[point_idx][camera_idx_0 * 2])
        y = float(points_2d[point_idx][camera_idx_0 * 2 + 1])
        if x == -1 or y == -1:
            continue

        p, q, r = calculate_rows_of_dot_between_camera_mat_and_3d_position(
            P, points_3d["points_3d"][point_idx]
        )
        print(p, q, r)
        input()
        extract_derivative(point_idx_0, camera_idx_0, target_type_0, target_types)

    return 0
