import sys
import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data_for_fundamental_matrix


def decompose_homography(H, f):
    f_0 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, f]])
    f_1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1/f]])

    H_new = np.dot(np.dot(f_0, H), f_1)
    print("H_new: ", H_new)
    H_norm = H_new / np.linalg.norm(H_new)
    print(H_norm)
    U, S, Vt = np.linalg.svd(H_norm)
    print(Vt)
    # v_1 = 
    input()


def main():
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = prepare_test_data_for_fundamental_matrix.prepare_test_data(
        False, False, "PLANE"
    )

    H_cv, _ = cv2.findHomography(
        np.float32(noised_img_pnts_0), np.float32(noised_img_pnts_1)
    )

    print(H_cv)

    f = 160
    width = 640
    height = 480
    pp = (width / 2, height / 2)
    K = np.matrix(
        np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype="double")
    )

    decompose_homography(H_cv, f)

    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H_cv, K)
    print(num)
    print(Rs)
    print(Ts)
    print(Ns)



if __name__ == '__main__':
    main()
