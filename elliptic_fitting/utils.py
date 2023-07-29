from matplotlib import pyplot as plt
import sympy
from tqdm import tqdm
import math
import numpy as np


def plot_base():
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()


def solve_fitting(theta, corr_x, f_0):
    y = sympy.Symbol("y")
    fit_x = []
    fit_y = []
    for x in tqdm(corr_x):
        f = (
            theta[0] * x**2
            + 2 * theta[1] * x * y
            + theta[2] * y**2
            + 2 * f_0 * (theta[3] * x + theta[4] * y)
            + f_0**2 * theta[5]
        )
        solutions = sympy.solve(f, y)
        for y_ans in solutions:
            if type(y_ans) == sympy.core.add.Add:
                continue
            fit_x.append(x)
            fit_y.append(y_ans)

    return fit_x, fit_y


def eval_pos_diff(corr_x, corr_y, est_x, est_y):
    diff_sum = 0
    for i in range(len(est_x)):
        x_idx = corr_x.index(est_x[i])
        diff_sum += math.dist([corr_x[x_idx], corr_y[x_idx]], [est_x[i], est_y[i]])

    diff_avg = diff_sum / len(est_x)

    return diff_sum, diff_avg


def get_elliptic_points_with_tilt():
    x = []
    y = []
    n_x = []
    n_y = []
    tilt = 45
    R = np.array(
        [
            [np.cos(np.deg2rad(tilt)), -np.sin(np.deg2rad(tilt))],
            [np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))],
        ]
    )
    for theta in range(360):
        point = np.array(
            [7.5 * np.cos(np.deg2rad(theta)), 5 * np.sin(np.deg2rad(theta))]
        )
        noise = np.random.normal(0, 0.2, point.shape)
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0])
        y.append(rotated_point[1])
        if theta % 3 == 0:
            with_noise = rotated_point + noise
            n_x.append(with_noise[0])
            n_y.append(with_noise[1])

    return x, y, n_x, n_y
