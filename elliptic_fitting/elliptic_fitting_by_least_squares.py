from matplotlib import pyplot as plt
import numpy as np
import sympy


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
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0])
        y.append(rotated_point[1])
        if theta % 3 == 0:
            with_noise = rotated_point + noise
            n_x.append(with_noise[0])
            n_y.append(with_noise[1])

    return x, y, n_x, n_y


def elliptic_fitting_by_least_squares(noise_x, noise_y):
    xi_sum = np.zeros((6, 6))
    for i in range(len(noise_x)):
        x = noise_x[i]
        y = noise_y[i]
        f = 20
        xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
        xi_sum += np.dot(xi.T, xi)
        print("xi_sum: ", xi_sum)
        # print(xi.T.shape)
        # print(M)
    M = xi_sum / len(noise_x)
    w, v = np.linalg.eig(M)
    min_eig_vec = v[:, np.argmin(w)]

    return min_eig_vec


def main():
    plot_base()
    corr_x, corr_y, noise_x, noise_y = get_elliptic_points_with_tilt()

    theta = elliptic_fitting_by_least_squares(noise_x, noise_y)
    # for i in range(corr_x):
        

    # plt.scatter(corr_x, corr_y, marker='o', c="black", s=20)
    plt.scatter(noise_x, noise_y, marker='o', c="blue", s=20)
    plt.show()


if __name__ == '__main__':
    main()
