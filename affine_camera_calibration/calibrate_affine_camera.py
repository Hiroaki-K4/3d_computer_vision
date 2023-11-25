import sys

import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calibrate_affine_camera(img_pnts_list):
    A = np.empty(len(img_pnts_list))
    C = np.empty(len(img_pnts_list))
    W = np.empty((2 * len(img_pnts_list), len(img_pnts_list[0])))
    for i in range(len(img_pnts_list)):
        points = img_pnts_list[i]
        t = np.mean(points, axis=0)
        t_x = t[0]
        t_y = t[1]
        A[i] = t_x * t_y
        C[i] = t_x**2 - t_y**2

        W[i * 2] = points[:, 0]
        W[i * 2 + 1] = points[:, 1]

    U, S, Vt = np.linalg.svd(W)
    U = U[:, :3]
    print(U)
    input()
    print(S)
    print(Vt)


def main():
    rot_euler_degrees = [[0, -30, 0], [0, -15, 0], [0, 0, 0], [0, 15, 0], [0, 30, 0]]
    T_in_camera_coords = [[0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10]]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_list,
        noised_img_pnts_list,
        points_3d,
    ) = prepare_test_data.prepare_multiple_test_images(
        False, "CURVE", rot_euler_degrees, T_in_camera_coords, f, width, height
    )

    calibrate_affine_camera(img_pnts_list)


if __name__ == "__main__":
    main()
