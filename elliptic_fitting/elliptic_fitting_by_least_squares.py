from matplotlib import pyplot as plt
import numpy as np
import sympy
from tqdm import tqdm


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


def elliptic_fitting_by_least_squares(noise_x, noise_y, f):
    xi_sum = np.zeros((6, 6))
    for i in range(len(noise_x)):
        x = noise_x[i]
        y = noise_y[i]
        xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
        xi_sum += np.dot(xi.T, xi)

    M = xi_sum / len(noise_x)
    w, v = np.linalg.eig(M)
    min_eig_vec = v[:, np.argmin(w)]

    return min_eig_vec


def main():
    plot_base()
    corr_x, corr_y, noise_x, noise_y = get_elliptic_points_with_tilt()

    f_0 = 20
    theta = elliptic_fitting_by_least_squares(noise_x, noise_y, f_0)
    y = sympy.Symbol('y')
    fit_x = []
    fit_y = []
    for x in tqdm(corr_x):
        f = theta[0]*x**2 + 2*theta[1]*x*y + theta[2]*y**2 + 2*f_0*(theta[3]*x+theta[4]*y)+f_0**2*theta[5]
        solutions=sympy.solve(f, y)
        # print("solutions: ", solutions)
        for y_ans in solutions:
            if type(y_ans) == sympy.core.add.Add:
                continue
            fit_x.append(x)
            fit_y.append(y_ans)

    # plt.scatter(corr_x, corr_y, marker='o', c="black", s=20)
    plt.scatter(noise_x, noise_y, marker='o', c="blue", s=20)
    plt.scatter(fit_x, fit_y, marker='o', c="red", s=20)
    plt.show()


if __name__ == '__main__':
    main()
