import utils
import math
import sympy
import numpy as np
import elliptic_fitting_by_least_squares
from matplotlib import pyplot as plt


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


def calculate_two_straight_lines(theta, f_0):
    n1 = theta[0]
    n2_0 = theta[1] - math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    n2_1 = theta[1] + math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    n3_0 = (
        theta[3]
        - (theta[1] * theta[3] - theta[0] * theta[4])
        / math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    ) * f_0
    n3_1 = (
        theta[3]
        + (theta[1] * theta[3] - theta[0] * theta[4])
        / math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    ) * f_0

    return n1, n2_0, n2_1, n3_0, n3_1


def get_intersection_with_substituting_x(n1, n2, n3, theta, f_0):
    y = sympy.Symbol("y")
    print("get_intersection_with_substituting_x: ", theta)

    eq = (
        (theta[0] * n2**2 - 2 * theta[1] * n1 * n2 + theta[2] * n1**2) * y**2
        + 2
        * f_0
        * (
            theta[4] * n1**2
            + theta[0] * n2 * n3
            - theta[1] * n1 * n3
            - theta[3] * n1 * n2
        )
        * y
        + (theta[0] * n3**2 - 2 * theta[3] * n1 * n3 + theta[5] * n1**2) * f_0**2
    )
    print(sympy.solve([eq]))


def get_intersection_with_substituting_y(n1, n2, n3, theta, f_0):
    x = sympy.Symbol("x")
    print("theta: ", theta)
    print("n: ", n1, n2, n3)
    input()

    eq = (
        (theta[0] * n2**2 - 2 * theta[1] * n1 * n2 + theta[2] * n1**2) * x**2
        + 2
        * f_0
        * (
            theta[3] * n2**2
            + theta[2] * n1 * n3
            - theta[1] * n2 * n3
            - theta[4] * n1 * n2
        )
        * x
        + (theta[2] * n3**2 - 2 * theta[4] * n2 * n3 + theta[5] * n2**2) * f_0**2
    )
    print(sympy.solve([eq]))
    # input()


def calculate_ellipse_intersection(q1, q2, f_0):
    print("Q1: ", q1)
    print("Q2: ", q2)
    x = sympy.Symbol("x")

    eq = (
        x**3 * np.linalg.det(q1)
        + x**2 * np.trace(np.dot(create_cofactor_mat(q1), q2))
        + x * np.trace(np.dot(q1, create_cofactor_mat(q2)))
        + np.linalg.det(q2)
    )

    lam = sympy.solve([eq])[0][x]

    new_q = np.dot(lam, q1) + q2
    new_theta = convert_to_ellipse_mat(new_q)
    n1, n2_0, n2_1, n3_0, n3_1 = calculate_two_straight_lines(new_theta, f_0)
    if abs(n2_0) >= abs(n1):
        get_intersection_with_substituting_y(n1, n2_0, n3_0, new_theta, f_0)
    else:
        get_intersection_with_substituting_x(n1, n2_0, n3_0, new_theta, f_0)

    return sympy.re(sympy.solve([eq])[0][x])


def convert_ellipse_to_conic(a, b, tilt, center, f_0):
    theta = np.deg2rad(tilt)
    A = np.cos(theta) ** 2 / a**2 + np.sin(theta) ** 2 / b**2
    B = (1 / a**2 - 1 / b**2) * np.sin(theta) * np.cos(theta)
    C = np.sin(theta) ** 2 / a**2 + np.cos(theta) ** 2 / b**2
    D = (
        (center[0] ** np.cos(theta) ** 2 + center[1] * np.sin(theta) * np.cos(theta))
        / a**2
    ) + (
        (center[0] * np.sin(theta) ** 2 - center[1] * np.sin(theta) * np.cos(theta))
        / b**2
    ) / (
        -f_0
    )
    E = (
        (center[0] ** np.sin(theta) * np.cos(theta) + center[1] * np.sin(theta) ** 2)
        / a**2
    ) + (
        (center[1] * np.cos(theta) ** 2 - center[0] * np.sin(theta) * np.cos(theta))
        / b**2
    ) / (
        -f_0
    )
    F = ((
        center[0] ** 2 * np.cos(theta) ** 2
        + 2 * center[0] * center[1] * np.sin(theta) * np.cos(theta)
        + center[1] ** 2 * np.sin(theta) ** 2
    ) / a**2 + (
        center[0] ** 2 * np.sin(theta) ** 2
        - 2 * center[0] * center[1] * np.sin(theta) * np.cos(theta)
        + center[1] ** 2 * np.cos(theta) ** 2
    ) / b**2 - 1) / f_0 ** 2

    return np.array([A, B, C, D, E, F])


def prepare_test_data(f_0):
    utils.plot_base()

    a = 7.5
    b = 5
    tilt = 45
    center = np.array([0, 0])
    q1_corr_x, q1_corr_y, q1_noise_x, q1_noise_y = utils.get_elliptic_points_with_tilt(
        a, b, tilt, center
    )
    # q1 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
    #     q1_corr_x, q1_corr_y, f_0
    # )
    q1 = convert_ellipse_to_conic(a, b, tilt, center, f_0)
    q1 = convert_to_conic_mat(q1)

    a = 7.5
    b = 5
    tilt = 120
    center = np.array([0, 0])
    q2_corr_x, q2_corr_y, q2_noise_x, q2_noise_y = utils.get_elliptic_points_with_tilt(
        a, b, tilt, center
    )
    # q2 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
        # q2_corr_x, q2_corr_y, f_0
    # )
    q2 = convert_ellipse_to_conic(a, b, tilt, center, f_0)
    q2 = convert_to_conic_mat(q2)

    plt.scatter(q1_corr_x, q1_corr_y, marker="o", c="green", s=20)
    plt.scatter(q2_corr_x, q2_corr_y, marker="o", c="blue", s=20)

    return q1, q2


if __name__ == "__main__":
    f_0 = 1
    q1, q2 = prepare_test_data(f_0)
    lam = calculate_ellipse_intersection(q1, q2, f_0)
    plt.show()
