import cv2
import numpy as np
import sys
sys.path.append('../')
from prepare_test_data_utils import prepare_test_data_for_fundamental_matrix


def calculate_focal_length(F_matrix):
    # F_Ft = np.dot()
    # Ft_F


def main(draw_test_data, draw_epipolar):
    img_pnts_0, img_pnts_1, noised_img_pnts_0, noised_img_pnts_1, F_matrix, rot_1_to_2, trans_1_to_2_in_camera_coord = prepare_test_data_for_fundamental_matrix.prepare_test_data(draw_test_data, draw_epipolar)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F_matrix")
    print(F_matrix)
    calculate_focal_length(F_matrix)


if __name__ == '__main__':
    draw_test_data = False
    draw_epipolar = False
    main(draw_test_data, draw_epipolar)
