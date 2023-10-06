import sys

import cv2
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def decompose_homography(H, f):
    f_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, f]])
    f_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 / f]])

    H_new = np.dot(np.dot(f_0, H), f_1)
    H_norm = H_new / np.cbrt(np.linalg.det(H_new))
    U, S, Vt = np.linalg.svd(H_norm)
    V = Vt.T
    v_1 = V[:, 0]
    v_2 = V[:, 1]
    v_3 = V[:, 2]
    s_1 = S[0]
    s_2 = S[1]
    s_3 = S[2]
    n_1 = np.sqrt(s_1**2 - s_2**2) * v_1 + np.sqrt(s_2**2 - s_3**2) * v_3
    n_2 = np.sqrt(s_1**2 - s_2**2) * v_1 - np.sqrt(s_2**2 - s_3**2) * v_3
    n_1_1 = n_1 / np.linalg.norm(n_1)
    n_1_2 = -n_1_1
    n_2_1 = n_2 / np.linalg.norm(n_2)
    n_2_2 = -n_2_1
    normal_vecs = [n_1_1, n_1_2, n_2_1, n_2_2]
    h = s_2 / (s_1 - s_3)
    t_1 = (
        -s_3 * np.sqrt(s_1**2 - s_2**2) * v_1
        + s_1 * np.sqrt(s_2**2 - s_3**2) * v_3
    )
    t_2 = (
        -s_3 * np.sqrt(s_1**2 - s_2**2) * v_1
        - s_1 * np.sqrt(s_2**2 - s_3**2) * v_3
    )
    t_1_1 = t_1 / np.linalg.norm(t_1)
    t_1_2 = -t_1_1
    t_2_1 = t_2 / np.linalg.norm(t_2)
    t_2_2 = -t_2_1
    trans_vecs = [t_1_1, t_1_2, t_2_1, t_2_2]
    rotations = []
    for idx in range(len(normal_vecs)):
        R = (
            1
            / s_2
            * np.dot(
                (np.identity(3) + s_2**3 * normal_vecs[idx] * trans_vecs[idx].T / h),
                H_norm.T,
            )
        )
        rotations.append(R)

    return (
        len(normal_vecs),
        np.array(rotations),
        np.array(trans_vecs),
        np.array(normal_vecs),
        h,
    )


def main():
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = prepare_test_data.prepare_test_data(
        False, False, "PLANE"
    )

    H_cv, _ = cv2.findHomography(
        np.float32(noised_img_pnts_0), np.float32(noised_img_pnts_1)
    )

    print("Homography matrix: ")
    print(H_cv)
    print("")

    f = 160
    num, Rs, Ts, Ns, h = decompose_homography(H_cv, f)

    print("~~~~~~~~~~~~~Homography decomposition~~~~~~~~~~~~~")
    print("Solutions num: ", num)
    print("Rs: ", Rs)
    print("Ts: ", Ts)
    print("Ns: ", Ns)
    print("h: ", h)


if __name__ == "__main__":
    main()
