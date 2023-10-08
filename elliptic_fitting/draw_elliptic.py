import sys

import numpy as np
from matplotlib import pyplot as plt


def plot_base():
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()


def get_circle_points():
    x = []
    y = []
    for theta in range(360):
        point = np.array([5 * np.cos(np.deg2rad(theta)), 5 * np.sin(np.deg2rad(theta))])
        x.append(point[0])
        y.append(point[1])

    return x, y


def get_elliptic_points():
    x = []
    y = []
    for theta in range(360):
        point = np.array(
            [7.5 * np.cos(np.deg2rad(theta)), 5 * np.sin(np.deg2rad(theta))]
        )
        x.append(point[0])
        y.append(point[1])

    return x, y


def get_elliptic_points_with_tilt():
    x = []
    y = []
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
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0])
        y.append(rotated_point[1])

    return x, y


def main():
    plot_base()
    c_x, c_y = get_circle_points()
    e_x, e_y = get_elliptic_points()
    r_x, r_y = get_elliptic_points_with_tilt()

    plt.scatter(c_x, c_y, marker="o", c="green")
    plt.scatter(e_x, e_y, marker="o", c="blue")
    plt.scatter(r_x, r_y, marker="o", c="red")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
