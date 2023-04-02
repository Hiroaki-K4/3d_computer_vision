from matplotlib import pyplot as plt
import numpy as np
import sympy
from tqdm import tqdm
import math


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
        noise = np.random.normal(0, 0.5, point.shape)
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


def elliptic_fitting_by_weighted_repetition(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0

    while diff > 1e-10:
        xi_sum = np.zeros((6, 6))
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
            xi_sum += np.dot(W[i], np.dot(xi.T, xi))

        M = xi_sum / len(noise_x)
        w, v = np.linalg.eig(M)
        theta = v[:, np.argmin(w)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        print("diff: ", diff)
        if diff <= 1e-10:
            break

        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            V0_xi = 4 * np.array([[x**2, x*y, 0, f*x, 0, 0],
                            [x*y, x**2+y**2, x*y, f*y, f*x, 0],
                            [0, x*y, y**2, 0, f*y, 0],
                            [f*x, f*y, 0, f**2, 0, 0],
                            [0, f*x, f*y, 0, f**2, 0],
                            [0, 0, 0, 0, 0, 0]])
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi, theta))))
        theta_zero = theta

    return theta


def solve_fitting(theta, corr_x, f_0):
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

    return fit_x, fit_y


def eval_pos_diff(corr_x, corr_y, est_x, est_y):
    diff_sum = 0
    for i in range(len(corr_x)):
        diff_sum += math.dist([corr_x[i], corr_y[i]], [est_x[i], est_y[i]])

    diff_avg = diff_sum / len(corr_x)

    return diff_sum, diff_avg


def main():
    plot_base()
    corr_x, corr_y, noise_x, noise_y = get_elliptic_points_with_tilt()

    f_0 = 20
    theta = elliptic_fitting_by_least_squares(noise_x, noise_y, f_0)
    w_theta = elliptic_fitting_by_weighted_repetition(noise_x, noise_y, f_0)
    print("theta: ", theta)
    print("weighted_theta: ", w_theta)

    fit_x, fit_y = solve_fitting(theta, corr_x, f_0)
    w_fit_x, w_fit_y = solve_fitting(w_theta, corr_x, f_0)

    least_sq_diff, least_sq_diff_avg = eval_pos_diff(corr_x, corr_y, fit_x, fit_y)
    weighted_diff, weighted_diff_avg = eval_pos_diff(corr_x, corr_y, w_fit_x, w_fit_y)
    print("least_sq_diff: ", least_sq_diff)
    print("weighted_diff: ", weighted_diff)
    print("least_sq_diff_avg: ", least_sq_diff_avg)
    print("weighted_diff_avg: ", weighted_diff_avg)

    plt.scatter(corr_x, corr_y, marker='o', c="black", s=20)
    plt.scatter(noise_x, noise_y, marker='o', c="blue", s=20)
    plt.scatter(fit_x, fit_y, marker='o', c="red", s=20)
    plt.scatter(w_fit_x, w_fit_y, marker='o', c="green", s=20)
    plt.show()


if __name__ == '__main__':
    main()
