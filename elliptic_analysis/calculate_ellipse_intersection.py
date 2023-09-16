import utils
import elliptic_fitting_by_least_squares
from matplotlib import pyplot as plt


def calculate_ellipse_intersection(q1, q2):
    print("Q1: ", q1)
    print("Q2: ", q2)


def prepare_test_data():
    utils.plot_base()

    q1_corr_x, q1_corr_y, q1_noise_x, q1_noise_y = utils.get_elliptic_points_with_tilt(
        45
    )
    f_0 = 20
    q1 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
        q1_corr_x, q1_corr_y, f_0
    )

    q2_corr_x, q2_corr_y, q2_noise_x, q2_noise_y = utils.get_elliptic_points_with_tilt(
        120
    )
    f_0 = 20
    q2 = elliptic_fitting_by_least_squares.elliptic_fitting_by_least_squares(
        q2_corr_x, q2_corr_y, f_0
    )

    plt.scatter(q1_corr_x, q1_corr_y, marker="o", c="green", s=20)
    plt.scatter(q2_corr_x, q2_corr_y, marker="o", c="blue", s=20)

    return q1, q2


if __name__ == "__main__":
    q1, q2 = prepare_test_data()
    calculate_ellipse_intersection(q1, q2)
    plt.show()
