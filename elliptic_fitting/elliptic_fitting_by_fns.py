from matplotlib import pyplot as plt
import numpy as np
import utils
import elliptic_fitting_by_least_squares


def elliptic_fitting_by_fns(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0

    while diff > 1e-10:
        xi_sum = np.zeros((6, 6))
        L_sum = np.zeros((6, 6))
        V0_xi_list = []
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
            xi_sum += np.dot(W[i], np.dot(xi.T, xi))
            V0_xi = 4 * np.array([[x**2, x*y, 0, f*x, 0, 0],
                            [x*y, x**2+y**2, x*y, f*y, f*x, 0],
                            [0, x*y, y**2, 0, f*y, 0],
                            [f*x, f*y, 0, f**2, 0, 0],
                            [0, f*x, f*y, 0, f**2, 0],
                            [0, 0, 0, 0, 0, 0]])
            V0_xi_list.append(V0_xi)
            L_sum += np.dot(np.dot(W[i]**2, np.dot(xi.T[:, 0], theta_zero)**2), V0_xi)

        M = xi_sum / len(noise_x)
        L = L_sum / len(noise_x)
        X = M - L
        eig_val, eig_vec = np.linalg.eig(X)
        theta = eig_vec[:, np.argmin(eig_val)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        print("diff: ", diff)
        if diff <= 1e-10:
            break

        for i in range(len(noise_x)):
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi_list[i], theta))))
        theta_zero = theta

    return theta


def main():
    utils.plot_base()
    corr_x, corr_y, noise_x, noise_y = utils.get_elliptic_points_with_tilt()

    f_0 = 20
    theta = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(noise_x, noise_y, f_0)
    fns_theta = elliptic_fitting_by_fns(noise_x, noise_y, f_0)
    print("theta: ", theta)
    print("fns_theta: ", fns_theta)

    fit_x, fit_y = utils.solve_fitting(theta, corr_x, f_0)
    f_fit_x, f_fit_y = utils.solve_fitting(fns_theta, corr_x, f_0)

    least_sq_diff, least_sq_diff_avg = utils.eval_pos_diff(corr_x, corr_y, fit_x, fit_y)
    fns_diff, fns_diff_avg = utils.eval_pos_diff(corr_x, corr_y, f_fit_x, f_fit_y)
    print("least_sq_diff_avg: ", least_sq_diff_avg)
    print("fns_diff_avg: ", fns_diff_avg)

    plt.scatter(corr_x, corr_y, marker='o', c="black", s=20)
    plt.scatter(noise_x, noise_y, marker='o', c="blue", s=20)
    plt.scatter(fit_x, fit_y, marker='o', c="red", s=20)
    plt.scatter(f_fit_x, f_fit_y, marker='o', c="green", s=20)
    plt.show()


if __name__ == '__main__':
    main()
