import sys

import cv2
import numpy as np
import utils

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def main():
    rot_euler_deg_0 = [0, 0, 0]
    rot_euler_deg_1 = [45, -30, 0]
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = prepare_test_data.prepare_test_data(
        False, False, "CIRCLE", rot_euler_deg_0, rot_euler_deg_1
    )
    print(img_pnts_1)
    input()
    f_0 = 20
    theta = utils.elliptic_fitting_by_least_squares(img_pnts_1, f_0)
    print("theta: ", theta)


if __name__ == "__main__":
    main()
