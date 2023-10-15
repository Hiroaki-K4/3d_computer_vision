import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

import utils

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def convert_points_to_xy(points):
    points_x = []
    points_y = []
    for point in points:
        points_x.append(point[0][0])
        points_y.append(point[0][1])

    return points_x, points_y


def convert_left_top_to_left_down(points_y, height):
    conv_points_y = []
    for pnt in points_y:
        conv_points_y.append(height - 1 - pnt)

    return conv_points_y


def reconstruct_support_plane(theta, f_0, f):
    q = utils.convert_to_conic_mat(theta)
    conv_f = np.array([[1 / f_0, 0, 0], [0, 1 / f_0, 0], [0, 0, 1 / f]])
    q_conv = np.dot(np.dot(conv_f, q), conv_f)
    q_norm = q_conv / np.cbrt((-1) * np.linalg.det(q_conv))

    w, v = np.linalg.eig(q_norm)
    sort_idx = np.argsort(w)
    lam_1 = w[sort_idx[1]]
    lam_2 = w[sort_idx[2]]
    lam_3 = w[sort_idx[0]]
    u_1 = v[:, sort_idx[1]]
    u_2 = v[:, sort_idx[2]]
    u_3 = v[:, sort_idx[0]]

    nomral_vec = np.sqrt(lam_2 - lam_1) * u_2 + np.sqrt(lam_1 - lam_3) * u_3
    nomral_vec = nomral_vec / np.linalg.norm(nomral_vec)

    return nomral_vec


def main():
    rot_euler_deg_0 = [0, 0, 0]
    rot_euler_deg_1 = [45, -30, 0]
    T_0_in_camera_coord = [0, 0, 10]
    T_1_in_camera_coord = [0, 0, 10]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
        points_3d,
    ) = prepare_test_data.prepare_test_data(
        False,
        False,
        "CIRCLE",
        rot_euler_deg_0,
        rot_euler_deg_1,
        T_0_in_camera_coord,
        T_1_in_camera_coord,
        f,
        width,
        height,
    )
    points_x, points_y = convert_points_to_xy(img_pnts_1)
    points_y = convert_left_top_to_left_down(points_y, height)
    f_0 = 20
    theta = utils.elliptic_fitting_by_least_squares(points_x, points_y, f_0)
    print("theta: ", theta)
    utils.draw_elliptic_fitting(theta, f_0, points_x, points_y)
    nomral_vec = reconstruct_support_plane(theta, f_0, f)
    print("nomral_vec: ", nomral_vec)
    ax = utils.plot_base_3d()
    ax.quiver(
        [0], [0], [0], [nomral_vec[0]], [nomral_vec[1]], [nomral_vec[2]], colors="r"
    )


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
