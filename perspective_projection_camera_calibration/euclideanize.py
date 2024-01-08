import sys

from calibrate_perspective_camera_by_primary_method import (
    calibrate_perspective_camera_by_primary_method,
    draw_reconstructed_points,
)

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def euclideanize(motion_mat, shape_mat):
    print("euclideanize")


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
        img_pnts_list, f_0
    )
    print("motion_mat: ", motion_mat.shape)
    print("shape_mat: ", shape_mat.shape)
    if show_flag:
        draw_reconstructed_points(
            img_pnts_list, motion_mat, shape_mat, width, height, f_0
        )

    # TODO Add euclideanize func
    euclideanize(motion_mat, shape_mat)


if __name__ == "__main__":
    show_flag = False
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    main(show_flag)
