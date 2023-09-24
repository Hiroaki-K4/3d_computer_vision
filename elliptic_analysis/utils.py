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


def get_elliptic_points_with_tilt(a, b, tilt, center):
    x = []
    y = []
    n_x = []
    n_y = []
    R = np.array(
        [
            [np.cos(np.deg2rad(tilt)), -np.sin(np.deg2rad(tilt))],
            [np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))],
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
