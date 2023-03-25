from matplotlib import pyplot as plt
import numpy as np


def plot_base():
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()


def get_circle_points():
    init_point = np.array([5, 0])
    x = []
    y = []
    for theta in range(360):
        point = np.array([5 * np.cos(theta), 5 * np.sin(theta)])
        x.append(point[0])
        y.append(point[1])

    return x, y


def get_elliptic_points(c_x):
    init_point = np.array([5, 0])
    x = []
    y = []
    for theta in range(360):
        point = np.array([7.5 * np.cos(theta), 5 * np.sin(theta)])
        x.append(point[0])
        y.append(point[1])

    return x, y


def main():
    plot_base()
    c_x, c_y = get_circle_points()
    e_x, e_y = get_elliptic_points(c_x)

    plt.scatter(c_x, c_y, marker='o', c="green")
    plt.scatter(e_x, e_y, marker='o', c="blue")
    plt.show()


if __name__ == '__main__':
    main()
