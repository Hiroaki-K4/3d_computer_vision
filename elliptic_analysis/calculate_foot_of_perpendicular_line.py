import sys

import numpy as np
from matplotlib import pyplot as plt

import utils


def calculate_foot_of_perpendicular_line(q, p, f):
    theta = utils.convert_to_ellipse_mat(q)
    a = p[0]
    b = p[1]
    a_ori = p[0]
    b_ori = p[1]
    a_move = 0
    b_move = 0
    J_0 = sys.float_info.max
    while True:
        V0_xi = 4 * np.array(
            [
                [a**2, a * b, 0, f * a, 0, 0],
                [a * b, a**2 + b**2, a * b, f * b, f * a, 0],
                [0, a * b, b**2, 0, f * b, 0],
                [f * a, f * b, 0, f**2, 0, 0],
                [0, f * a, f * b, 0, f**2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        xi = np.array(
            [
                a**2 + 2 * a * a_move,
                2 * (a * b + b * a_move + a * b_move),
                b**2 + 2 * b * b_move,
                2 * f * (a + a_move),
                2 * f * (b + b_move),
                f**2,
            ]
        )
        theta_reshape = np.array(
            [[theta[0], theta[1], theta[3]], [theta[1], theta[2], theta[4]]]
        )
        move = np.dot(
            np.dot(
                2 * np.dot(xi, theta) / np.dot(theta, np.dot(V0_xi, theta)),
                theta_reshape,
            ),
            np.array([a, b, f]),
        )
        a_move = move[0]
        b_move = move[1]
        a = a_ori - a_move
        b = b_ori - b_move
        J = a_move**2 + b_move**2
        if abs(J_0 - J) < 1e-5:
            break
        else:
            J_0 = J

    return [a, b]


def main():
    a = 7.5
    b = 5
    slope = 0
    center = np.array([0, 0])
    f_0 = 1
    q = utils.prepare_test_data(a, b, slope, center, f_0)

    q_corr_x, q_corr_y, q_noise_x, q_noise_y = utils.get_elliptic_points_with_slope(
        a + 1, b + 1, slope, center
    )

    points_x = []
    points_y = []
    ans_points_x = []
    ans_points_y = []
    for i in range(len(q_corr_x)):
        if i % 10 == 0:
            point = np.array([q_corr_x[i], q_corr_y[i]])
            ans_point = calculate_foot_of_perpendicular_line(q, point, f_0)
            print("Initial point: ({0}, {1})".format(point[0], point[1]))
            print("Foot point: ({0}, {1})".format(ans_point[0], ans_point[1]))
            points_x.append(point[0])
            points_y.append(point[1])
            ans_points_x.append(ans_point[0])
            ans_points_y.append(ans_point[1])
            plt.plot([point[0], ans_point[0]], [point[1], ans_point[1]], color="red")

    plt.scatter(points_x, points_y, marker="o", c="red", s=40)
    plt.scatter(ans_points_x, ans_points_y, marker="o", c="red", s=40)
    plt.show()


if __name__ == "__main__":
    main()
