import utils
import math
import sympy
import numpy as np
from matplotlib import pyplot as plt


def change_zero_sign(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (mat[i, j]) == -0:
                mat[i, j] = abs(mat[i, j])

    return mat


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


def calculate_two_straight_lines(theta):
    n1 = theta[0]
    n2_0 = theta[1] - math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    n2_1 = theta[1] + math.sqrt(theta[1] ** 2 - theta[0] * theta[2])
    n3_0 = theta[3] - (theta[1] * theta[3] - theta[0] * theta[4]) / math.sqrt(
        theta[1] ** 2 - theta[0] * theta[2]
    )
    n3_1 = theta[3] + (theta[1] * theta[3] - theta[0] * theta[4]) / math.sqrt(
        theta[1] ** 2 - theta[0] * theta[2]
    )

    return n1, n2_0, n2_1, n3_0, n3_1


def get_intersection_with_substituting_x(n1, n2, n3, theta, f_0):
    y = sympy.Symbol("y")

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
    x_ans = []
    y_ans = []
    y_sols = sympy.solve([eq])
    for y_sol in y_sols:
        x = -(n2*y_sol[y]+n3*f_0)/n1
        x_ans.append(x)
        y_ans.append(y_sol[y])

    return x_ans, y_ans


def get_intersection_with_substituting_y(n1, n2, n3, theta, f_0):
    x = sympy.Symbol("x")

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
    x_ans = []
    y_ans = []
    x_sols = sympy.solve([eq])
    for x_sol in x_sols:
        y = -(n1*x_sol[x]+n3*f_0)/n2
        x_ans.append(x_sol[x])
        y_ans.append(y)

    return x_ans, y_ans


def draw_lines(n1, n2_0, n2_1, n3_0, n3_1, f_0):
    start_x = 7
    end_x = -5
    line1_y_start = -(n1 * start_x + n3_0 * f_0) / n2_0
    line1_y_end = -(n1 * end_x + n3_0 * f_0) / n2_0
    line2_y_start = -(n1 * start_x + n3_1 * f_0) / n2_1
    line2_y_end = -(n1 * end_x + n3_1 * f_0) / n2_1
    plt.plot([start_x, end_x], [line1_y_start, line1_y_end], c="black")
    plt.plot([start_x, end_x], [line2_y_start, line2_y_end], c="black")


def convert_ellipse_to_conic(a, b, tilt, center, f_0):
    theta = np.deg2rad(tilt)
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


def prepare_test_data(f_0):
    utils.plot_base()

    a = 7.5
    b = 5
    tilt = 20
    center = np.array([1, 1])
    q1_corr_x, q1_corr_y, q1_noise_x, q1_noise_y = utils.get_elliptic_points_with_tilt(
        a, b, tilt, center
    )
    q1 = convert_ellipse_to_conic(a, b, tilt, center, f_0)
    # utils.check_conic_mat(q1_corr_x, q1_corr_y, f_0, q1)
    q1 = convert_to_conic_mat(q1)

    a = 7.5
    b = 5
    tilt = 60
    center = np.array([1, 1])
    q2_corr_x, q2_corr_y, q2_noise_x, q2_noise_y = utils.get_elliptic_points_with_tilt(
        a, b, tilt, center
    )
    q2 = convert_ellipse_to_conic(a, b, tilt, center, f_0)
    # utils.check_conic_mat(q2_corr_x, q2_corr_y, f_0, q2)
    q2 = convert_to_conic_mat(q2)

    plt.scatter(q1_corr_x, q1_corr_y, marker="o", c="green", s=20)
    plt.scatter(q2_corr_x, q2_corr_y, marker="o", c="blue", s=20)

    return q1, q2


def calculate_ellipse_intersection(q1, q2, f_0):
    x = sympy.Symbol("x", real=True)

    eq = (
        x**3 * np.linalg.det(q1)
        + x**2 * np.trace(np.dot(create_cofactor_mat(q1), q2))
        + x * np.trace(np.dot(q1, create_cofactor_mat(q2)))
        + np.linalg.det(q2)
    )

    sols = sympy.solve([eq])
    if len(sols) > 1:
        lam = sols[1][x]
    elif len(sols) == 1:
        lam = sols[0][x]
    else:
        raise RuntimeError("There is no real solution.")

    new_q = np.dot(lam, q1) + q2
    new_theta = convert_to_ellipse_mat(new_q)
    n1, n2_0, n2_1, n3_0, n3_1 = calculate_two_straight_lines(new_theta)
    draw_lines(n1, n2_0, n2_1, n3_0, n3_1, f_0)
    print("Line1: {0}x+{1}y+{2}=0".format(n1, n2_0, n3_0 * f_0))
    print("Line2: {0}x+{1}y+{2}=0".format(n1, n2_1, n3_1 * f_0))

    if abs(n2_0) >= abs(n1):
        x_ans_1, y_ans_1 = get_intersection_with_substituting_y(
            n1, n2_0, n3_0, convert_to_ellipse_mat(q2), f_0
        )
    else:
        x_ans_1, y_ans_1 = get_intersection_with_substituting_x(
            n1, n2_0, n3_0, convert_to_ellipse_mat(q2), f_0
        )
    if abs(n2_1) >= abs(n1):
        x_ans_2, y_ans_2 = get_intersection_with_substituting_y(
            n1, n2_1, n3_1, convert_to_ellipse_mat(q1), f_0
        )
    else:
        x_ans_2, y_ans_2 = get_intersection_with_substituting_x(
            n1, n2_1, n3_1, convert_to_ellipse_mat(q1), f_0
        )

    intersection_x = x_ans_1 + x_ans_2
    intersection_y = y_ans_1 + y_ans_2

    plt.scatter(intersection_x, intersection_y, marker="o", c="red", s=40)

    print()
    print("[Intersection]")
    for i in range(len(intersection_x)):
        print("{0}: ({1}, {2})".format(i+1, intersection_x[i], intersection_y[i]))

    return x_ans_1, y_ans_1, x_ans_2, y_ans_2


if __name__ == "__main__":
    f_0 = 1
    q1, q2 = prepare_test_data(f_0)
    x_ans_1, y_ans_1, x_ans_2, y_ans_2 = calculate_ellipse_intersection(q1, q2, f_0)
    plt.show()
