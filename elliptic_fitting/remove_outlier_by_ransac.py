from matplotlib import pyplot as plt
import numpy as np
import utils
import random


def get_elliptic_points_with_outlier():
    x = []
    y = []
    n_x = []
    n_y = []
    tilt = 45
    R = np.array([[np.cos(np.deg2rad(tilt)), -np.sin(np.deg2rad(tilt))], [np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))]])
    for theta in range(360):
        point = np.array([7.5 * np.cos(np.deg2rad(theta)), 5 * np.sin(np.deg2rad(theta))])
        rotated_point = np.dot(R, point.T)
        x.append(rotated_point[0])
        y.append(rotated_point[1])
        if theta % 3 == 0:
            if theta % 30 == 0:
                noise = np.random.normal(0, 5.0, point.shape)
            else:
                noise = np.random.normal(0, 0.5, point.shape)
            with_noise = rotated_point + noise
            n_x.append(with_noise[0])
            n_y.append(with_noise[1])

    return x, y, n_x, n_y


def elliptic_fitting_by_fns(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0

    count = 0
    while diff > 1e-10 and count < 100:
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

        count += 1

    return theta


def removed_outlier(noise_x, noise_y, theta, f, threshold):
    removed_x = []
    removed_y = []
    for i in range(len(noise_x)):
        x = noise_x[i]
        y = noise_y[i]
        xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
        V0_xi = 4 * np.array([[x**2, x*y, 0, f*x, 0, 0],
                            [x*y, x**2+y**2, x*y, f*y, f*x, 0],
                            [0, x*y, y**2, 0, f*y, 0],
                            [f*x, f*y, 0, f**2, 0, 0],
                            [0, f*x, f*y, 0, f**2, 0],
                            [0, 0, 0, 0, 0, 0]])
        d = (np.dot(xi, theta))**2 / (np.dot(theta, np.dot(V0_xi, theta)))
        if d[0] < threshold:
            removed_x.append(x)
            removed_y.append(y)

    return removed_x, removed_y


def remove_outlier_by_ransac(noise_x, noise_y, f):
    inliner_count_max = 0
    no_update_count = 0
    desired_theta = np.zeros(6, dtype="float64")
    threshold = 1
    while no_update_count < 5000:
        random_list = random.sample(range(len(noise_x)), k=5)
        M = np.zeros((6, 6))
        for idx in random_list:
            x = noise_x[idx]
            y = noise_y[idx]
            xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
            M += np.dot(xi.T, xi)

        eig_val, eig_vec = np.linalg.eig(M)
        theta = eig_vec[:, np.argmin(eig_val)]

        removed_x = []
        removed_y = []
        inlier_count = 0
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2*x*y, y**2, 2*f*x, 2*f*y, f*f]])
            V0_xi = 4 * np.array([[x**2, x*y, 0, f*x, 0, 0],
                                [x*y, x**2+y**2, x*y, f*y, f*x, 0],
                                [0, x*y, y**2, 0, f*y, 0],
                                [f*x, f*y, 0, f**2, 0, 0],
                                [0, f*x, f*y, 0, f**2, 0],
                                [0, 0, 0, 0, 0, 0]])
            d = (np.dot(xi, theta))**2 / (np.dot(theta, np.dot(V0_xi, theta)))
            if d[0] < threshold:
                inlier_count += 1

        if inlier_count > inliner_count_max:
            inliner_count_max = inlier_count
            desired_theta = theta
            no_update_count = 0
        else:
            no_update_count += 1

    removed_x, removed_y = removed_outlier(noise_x, noise_y, desired_theta, f, threshold)

    return removed_x, removed_y


def main():
    utils.plot_base()
    corr_x, corr_y, noise_x, noise_y = get_elliptic_points_with_outlier()

    f_0 = 20
    removed_x, removed_y = remove_outlier_by_ransac(noise_x, noise_y, f_0)

    fns_theta = elliptic_fitting_by_fns(removed_x, removed_y, f_0)
    print("fns_theta: ", fns_theta)

    f_fit_x, f_fit_y = utils.solve_fitting(fns_theta, corr_x, f_0)

    fns_diff, fns_diff_avg = utils.eval_pos_diff(corr_x, corr_y, f_fit_x, f_fit_y)
    print("fns_diff_avg: ", fns_diff_avg)

    plt.scatter(corr_x, corr_y, marker='o', c="black", s=20)
    plt.scatter(noise_x, noise_y, marker='o', c="blue", s=20)
    plt.scatter(removed_x, removed_y, marker='o', c="red", s=20)
    plt.scatter(f_fit_x, f_fit_y, marker='o', c="green", s=20)
    plt.show()


if __name__ == '__main__':
    main()
