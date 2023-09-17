import utils
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


def calculate_ellipse_intersection(q1, q2):
    print("Q1: ", q1)
    print("Q2: ", q2)
    x = sympy.Symbol("x")

    eq = (
        x**3 * np.linalg.det(q1)
        + x**2 * np.trace(np.dot(create_cofactor_mat(q1), q2))
        + x * np.trace(np.dot(q1, create_cofactor_mat(q2)))
        + np.linalg.det(q2)
    )

    lam = sympy.re(sympy.solve([eq])[0][x])

    net_q = np.dot(lam, q1) + q2

    # TODO Caclculate n1, n2, n3 and intersection

    return sympy.re(sympy.solve([eq])[0][x])


def prepare_test_data():
    utils.plot_base()

    q1_corr_x, q1_corr_y, q1_noise_x, q1_noise_y = utils.get_elliptic_points_with_tilt(
        45
    )
    f_0 = 20
    q1 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
        q1_corr_x, q1_corr_y, f_0
    )
    q1 = convert_to_conic_mat(q1)

    q2_corr_x, q2_corr_y, q2_noise_x, q2_noise_y = utils.get_elliptic_points_with_tilt(
        120
    )
    f_0 = 20
    q2 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
        q2_corr_x, q2_corr_y, f_0
    )
    q2 = convert_to_conic_mat(q2)

    plt.scatter(q1_corr_x, q1_corr_y, marker="o", c="green", s=20)
    plt.scatter(q2_corr_x, q2_corr_y, marker="o", c="blue", s=20)

    return q1, q2


if __name__ == "__main__":
    q1, q2 = prepare_test_data()
    lam = calculate_ellipse_intersection(q1, q2)
    plt.show()
