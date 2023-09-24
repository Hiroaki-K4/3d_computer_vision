import numpy as np
from matplotlib import pyplot as plt


def plot_base():
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()


def get_elliptic_points_with_slope(a, b, slope, center):
    x = []
    y = []
    n_x = []
    n_y = []
    R = np.array(
        [
            [np.cos(np.deg2rad(slope)), -np.sin(np.deg2rad(slope))],
            [np.sin(np.deg2rad(slope)), np.cos(np.deg2rad(slope))],
        ]
    )
    for theta in range(360):
        point = np.array([a * np.cos(np.deg2rad(theta)), b * np.sin(np.deg2rad(theta))])
        noise = np.random.normal(0, 0.2, point.shape)
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0] + center[0])
        y.append(rotated_point[1] + center[1])
        if theta % 3 == 0:
            with_noise = rotated_point + noise
            n_x.append(with_noise[0] + center[0])
            n_y.append(with_noise[1] + center[1])

    return x, y, n_x, n_y


def normalize_ellipse(theta):
    return theta / np.linalg.norm(theta.astype(float))


def check_conic_mat(x, y, f_0, q):
    for i in range(len(x)):
        res = (
            q[0] * x[i] ** 2
            + 2 * q[1] * x[i] * y[i]
            + q[2] * y[i] ** 2
            + 2 * f_0 * (q[3] * x[i] + q[4] * y[i])
            + f_0**2 * q[5]
        )
        print("res: ", res)


def draw_lines(n1, n2_0, n2_1, n3_0, n3_1, f_0):
    start_x = 7
    end_x = -5
    line1_y_start = -(n1 * start_x + n3_0 * f_0) / n2_0
    line1_y_end = -(n1 * end_x + n3_0 * f_0) / n2_0
    line2_y_start = -(n1 * start_x + n3_1 * f_0) / n2_1
    line2_y_end = -(n1 * end_x + n3_1 * f_0) / n2_1
    plt.plot([start_x, end_x], [line1_y_start, line1_y_end], c="black")
    plt.plot([start_x, end_x], [line2_y_start, line2_y_end], c="black")


def calculate_cofactor(mat, i, j):
    x = np.delete(mat, i, axis=0)
    x = np.delete(x, j, axis=1)
    cof = (-1) ** (i + j) * np.linalg.det(x)

    return cof


def create_cofactor_mat(mat):
    cof_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            cof_mat[j, i] = calculate_cofactor(mat, i, j)

    return cof_mat


def convert_to_conic_mat(theta):
    q = np.array(
        [
            [theta[0], theta[1], theta[3]],
            [theta[1], theta[2], theta[4]],
            [theta[3], theta[4], theta[5]],
        ]
    )

    return q


def convert_to_ellipse_mat(q):
    theta = np.array([q[0, 0], q[0, 1], q[1, 1], q[0, 2], q[1, 2], q[2, 2]])

    return theta


def convert_ellipse_to_conic(a, b, slope, center, f_0):
    theta = np.deg2rad(slope)
    A = np.cos(theta) ** 2 / a**2 + np.sin(theta) ** 2 / b**2
    B = (1 / a**2 - 1 / b**2) * np.sin(theta) * np.cos(theta)
    C = np.sin(theta) ** 2 / a**2 + np.cos(theta) ** 2 / b**2
    D = (
        (
            (center[0] * np.cos(theta) ** 2 + center[1] * np.sin(theta) * np.cos(theta))
            / a**2
        )
        + (
            (center[0] * np.sin(theta) ** 2 - center[1] * np.sin(theta) * np.cos(theta))
            / b**2
        )
    ) / -f_0
    E = (
        (
            (center[0] * np.sin(theta) * np.cos(theta) + center[1] * np.sin(theta) ** 2)
            / a**2
        )
        + (
            (center[1] * np.cos(theta) ** 2 - center[0] * np.sin(theta) * np.cos(theta))
            / b**2
        )
    ) / -f_0
    F = (
        (
            center[0] ** 2 * np.cos(theta) ** 2
            + 2 * center[0] * center[1] * np.sin(theta) * np.cos(theta)
            + center[1] ** 2 * np.sin(theta) ** 2
        )
        / a**2
        + (
            center[0] ** 2 * np.sin(theta) ** 2
            - 2 * center[0] * center[1] * np.sin(theta) * np.cos(theta)
            + center[1] ** 2 * np.cos(theta) ** 2
        )
        / b**2
        - 1
    ) / f_0**2

    return np.array([A, B, C, D, E, F])
