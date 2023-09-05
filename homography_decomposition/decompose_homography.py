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
    H_norm = H_new / np.linalg.det(H_new) ** 1/3
    print(np.linalg.det(H_norm))
    input()
    U, S, Vt = np.linalg.svd(H_norm)
    V = Vt.T
    print("V", V)
    v_1 = V[:, 0]
    v_2 = V[:, 1]
    v_3 = V[:, 2]
    s_1 = S[0]
    s_2 = S[1]
    s_3 = S[2]
    print(s_1)
    n_1 = np.sqrt(s_1 ** 2 - s_2 ** 2) * v_1 + np.sqrt(s_2 ** 2 - s_3 ** 2) * v_3
    n_2 = np.sqrt(s_1 ** 2 - s_2 ** 2) * v_1 - np.sqrt(s_2 ** 2 - s_3 ** 2) * v_3
    n_1 = n_1 / np.linalg.norm(n_1)
    n_2 = n_2 / np.linalg.norm(n_2)
    h = s_2 / (s_1 - s_3)
    print(n_1)
    print(n_2)
    print(h)
    t_1 = -s_3 * np.sqrt(s_1 ** 2 - s_2 ** 2) * v_1 + s_1 * np.sqrt(s_2 ** 2 - s_3 ** 2) * v_3
    t_2 = -s_3 * np.sqrt(s_1 ** 2 - s_2 ** 2) * v_1 - s_1 * np.sqrt(s_2 ** 2 - s_3 ** 2) * v_3
    t_1 = t_1 / np.linalg.norm(t_1)
    t_2 = t_2 / np.linalg.norm(t_2)
    print(t_1)
    print(t_2)
    # R = 
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
