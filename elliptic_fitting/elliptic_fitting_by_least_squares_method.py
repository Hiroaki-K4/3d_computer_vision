from matplotlib import pyplot as plt
import numpy as np


def plot_base():
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()


def get_elliptic_points_with_tilt():
    x = []
    y = []
    n_x = []
    n_y = []
    tilt = 45
    R = np.array([[np.cos(np.deg2rad(tilt)), -np.sin(np.deg2rad(tilt))], [np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))]])
    for theta in range(360):
        point = np.array([7.5 * np.cos(np.deg2rad(theta)), 5 * np.sin(np.deg2rad(theta))])
        noise = np.random.normal(0, 0.2, point.shape)
        print(noise)
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0])
        y.append(rotated_point[1])
        if theta % 3 == 0:
            with_noise = rotated_point + noise
            n_x.append(with_noise[0])
            n_y.append(with_noise[1])

    return x, y, n_x, n_y


def main():
    plot_base()
    r_x, r_y, n_x, n_y = get_elliptic_points_with_tilt()

    plt.scatter(r_x, r_y, marker='o', c="black", s=20)
    plt.scatter(n_x, n_y, marker='o', c="blue", s=20)
    plt.show()


if __name__ == '__main__':
    main()
