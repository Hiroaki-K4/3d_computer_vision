import sys
import numpy as np

from calibrate_perspective_camera_by_primary_method import (
    calibrate_perspective_camera_by_primary_method,
    draw_reconstructed_points,
)

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def calculate_mat_including_homography_mat(K, motion_mat):
    print(K.shape)
    print(motion_mat.shape)
    img_num = int(motion_mat.shape[0] / 3)
    for i in range(img_num):
        print(i)
        # TODO Calcualte omega


def euclideanize(motion_mat, shape_mat, f, f_0, opt_axis):
    J_med = sys.float_info.max
    img_num = int(motion_mat.shape[0] / 3)
    K = np.empty((img_num, 3, 3))
    for i in range(img_num):
        K_new = np.array([[f, 0, opt_axis[0]], [0, f, opt_axis[1]], [0, 0, f_0]])
        K[i] = K_new

    calculate_mat_including_homography_mat(K, motion_mat)


def main(show_flag: bool):
    rot_euler_degrees = [
        [-10, -30, 0],
        [15, -15, 0],
        [0, 0, 0],
        [15, 15, 0],
        [10, 30, 0],
    ]
    T_in_camera_coords = [[0, 0, 10], [0, 0, 7.5], [0, 0, 5], [0, 0, 7.5], [0, 0, 10]]
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

    f_0 = width
    motion_mat, shape_mat = calibrate_perspective_camera_by_primary_method(
        img_pnts_list, f_0, 2.0
    )
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    if show_flag:
        draw_reconstructed_points(
            img_pnts_list, motion_mat, shape_mat, width, height, f_0
        )

    euclideanize(motion_mat, shape_mat, f, f_0, [0.0, 0.0])


if __name__ == "__main__":
    # TODO Return flag when finish implementing
    show_flag = False
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
