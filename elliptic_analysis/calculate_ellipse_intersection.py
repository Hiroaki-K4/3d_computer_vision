import math
import sys

import numpy as np
import sympy
from matplotlib import pyplot as plt

import utils


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
        x = -(n2 * y_sol[y] + n3 * f_0) / n1
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
        y = -(n1 * x_sol[x] + n3 * f_0) / n2
        x_ans.append(x_sol[x])
        y_ans.append(y)

    return x_ans, y_ans


def calculate_ellipse_intersection(q1, q2, f_0):
    x = sympy.Symbol("x", real=True)

    eq = (
        x**3 * np.linalg.det(q1)
        + x**2 * np.trace(np.dot(utils.create_cofactor_mat(q1), q2))
        + x * np.trace(np.dot(q1, utils.create_cofactor_mat(q2)))
        + np.linalg.det(q2)
    )

    sols = sympy.solve([eq])
    have_sol = False
    for i in range(len(sols)):
        sol = float(sols[i][x])
        if sol.is_integer():
            lam = sol
            have_sol = True
            break

    if not have_sol:
        raise RuntimeError("There is no real solution.")

    new_q = np.dot(lam, q1) + q2
    new_theta = utils.convert_to_ellipse_mat(new_q)
    n1, n2_0, n2_1, n3_0, n3_1 = calculate_two_straight_lines(new_theta)
    utils.draw_lines(n1, n2_0, n2_1, n3_0, n3_1, f_0)
    print("Line1: {0}x+{1}y+{2}=0".format(n1, n2_0, n3_0 * f_0))
    print("Line2: {0}x+{1}y+{2}=0".format(n1, n2_1, n3_1 * f_0))

    if abs(n2_0) >= abs(n1):
        x_ans_1, y_ans_1 = get_intersection_with_substituting_y(
            n1, n2_0, n3_0, utils.convert_to_ellipse_mat(q2), f_0
        )
    else:
        x_ans_1, y_ans_1 = get_intersection_with_substituting_x(
            n1, n2_0, n3_0, utils.convert_to_ellipse_mat(q2), f_0
        )
    if abs(n2_1) >= abs(n1):
        x_ans_2, y_ans_2 = get_intersection_with_substituting_y(
            n1, n2_1, n3_1, utils.convert_to_ellipse_mat(q1), f_0
        )
    else:
        x_ans_2, y_ans_2 = get_intersection_with_substituting_x(
            n1, n2_1, n3_1, utils.convert_to_ellipse_mat(q1), f_0
        )

    print(x_ans_1)
    print(x_ans_2)

    intersection_x = x_ans_1 + x_ans_2
    intersection_y = y_ans_1 + y_ans_2

    plt.scatter(intersection_x, intersection_y, marker="o", c="red", s=40)

    print()
    print("[Intersection]")
    for i in range(len(intersection_x)):
        print("{0}: ({1}, {2})".format(i + 1, intersection_x[i], intersection_y[i]))

    return x_ans_1, y_ans_1, x_ans_2, y_ans_2


if __name__ == "__main__":
    a = 7.5
    b = 5
    center = np.array([1, 1])
    f_0 = 1
    q1 = utils.prepare_test_data(a, b, 20, center, f_0)
    q2 = utils.prepare_test_data(a, b, 60, center, f_0)
    x_ans_1, y_ans_1, x_ans_2, y_ans_2 = calculate_ellipse_intersection(q1, q2, f_0)
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
