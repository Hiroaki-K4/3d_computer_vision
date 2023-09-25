import math

import numpy as np
import sympy
from matplotlib import pyplot as plt

import utils
import calculate_ellipse_intersection as inter


def get_quadratic_curve_about_perpendicular_line(q, p, f_0):
    print(q)
    theta = utils.convert_to_ellipse_mat(q)
    # TODO Check equation of d is correct
    d = np.array(
        [
            [
                theta[1],
                (theta[2] - theta[0]) / 2,
                (theta[0] * p[1] - theta[1] * p[0] + theta[4] * f_0) / 2 * f_0,
            ],
            [
                (theta[2] - theta[0]) / 2,
                -theta[1],
                (theta[1] * p[1] - theta[2] * p[0] - theta[3] * f_0) / 2 * f_0,
            ],
            [
                (theta[0] * p[1] - theta[1] * p[0] + theta[4] * f_0) / 2 * f_0,
                (theta[1] * p[1] - theta[2] * p[0] - theta[3] * f_0) / 2 * f_0,
                (theta[3] * p[1] - theta[4] * p[0]) / f_0,
            ],
        ]
    )

    print(d)
    return d


if __name__ == "__main__":
    a = 7.5
    b = 5
    center = np.array([1, 1])
    f_0 = 1
    q = utils.prepare_test_data(a, b, 0, center, f_0)
    point = np.array([1, 7])
    plt.scatter([point[0]], [point[1]], marker="o", c="red", s=40)
    d = get_quadratic_curve_about_perpendicular_line(q, point, f_0)
    x_ans_1, y_ans_1, x_ans_2, y_ans_2 = inter.calculate_ellipse_intersection(q, d, f_0)
    plt.show()
