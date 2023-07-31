import sys
import numpy as np

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data_for_fundamental_matrix


def update_weight(theta, img_pnts_0, img_pnts_1, f_0):
    new_W = np.zeros((0, 3, 3))
    for i in range(len(img_pnts_0)):
        p_0 = img_pnts_0[i][0]
        p_1 = img_pnts_1[i][0]
        T_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [-f_0, 0, 0, 0], [0, -f_0, 0, 0], [0, 0, 0, 0],
                        [p_1[1], 0, 0, p_0[0]], [0, p_1[1], 0, p_0[1]], [0, 0, 0, f_0]])
        T_2 = np.array([[f_0, 0, 0, 0], [0, f_0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [-p_1[0], 0, -p_0[0], 0], [0, -p_1[0], -p_0[1], 0], [0, 0, -f_0, 0]])
        T_3 = np.array([[-p_1[1], 0, 0, -p_0[0]], [0, -p_1[1], 0, -p_0[1]], [0, 0, 0, -f_0],
                        [p_1[0], 0, p_0[0], 0], [0, p_1[0], p_0[1], 0], [0, 0, f_0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        T_list = [T_1, T_2, T_3]
        W = np.zeros((3, 3))
        for k in range(3):
            for l in range(3):
                V_0_xi = np.dot(T_list[k], T_list[l].T)
                W[k, l] = np.dot(theta, np.dot(V_0_xi, theta))

        w, v = np.linalg.eig(W)
        W_inv_rank_2 = np.zeros((3, 3))
        max_vec = np.array([v[:, np.argmax(w)]]).T
        W_inv_rank_2 += np.dot(max_vec, max_vec.T) / w[np.argmax(w)]
        v = np.delete(v, np.argmax(w), 1)
        w = np.delete(w, np.argmax(w), 0)
        max_vec = np.array([v[:, np.argmax(w)]]).T
        W_inv_rank_2 += np.dot(max_vec, max_vec.T) / w[np.argmax(w)]
        new_W = np.append(new_W, np.array([W_inv_rank_2]), axis=0)

    return new_W


def calculate_projective_transformation(img_pnts_0, img_pnts_1):
    pre_theta = np.zeros(9)
    # W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    f_0 = 480
    M_sum = np.zeros([9, 9])
    W = np.zeros((0, 3, 3))
    for i in range(len(img_pnts_0)):
        W = np.append(W, np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), axis=0)

    count = 0
    count_thr = 100
    while True:
        for i in range(len(img_pnts_0)):
            p_0 = img_pnts_0[i][0]
            p_1 = img_pnts_1[i][0]
            xi_1 = np.array([[0, 0, 0, -f_0*p_0[0], -f_0*p_0[1], -f_0**2, p_0[0]*p_1[1], p_0[1]*p_1[1], f_0*p_1[1]]]).T
            xi_2 = np.array([[f_0*p_0[0], f_0*p_0[1], f_0**2, 0, 0, 0, -p_0[0]*p_1[0], -p_0[1]*p_1[0], -f_0*p_1[0]]]).T
            xi_3 = np.array([[-p_0[0]*p_1[1], -p_0[1]*p_1[1], -f_0*p_1[1], p_0[0]*p_1[0], p_0[1]*p_1[0], f_0*p_1[0], 0, 0, 0]]).T
            xi_list = [xi_1, xi_2, xi_3]
            for k in range(3):
                for l in range(3):
                    xi_k = xi_list[k]
                    xi_l = xi_list[l]
                    M_sum += np.dot(W[i, k, l], np.dot(xi_k, xi_l.T))

        M = M_sum / len(img_pnts_0)
        w, v = np.linalg.eig(M)
        theta = v[:, np.argmin(w)]
        theta_diff = np.linalg.norm(theta - pre_theta)
        print("theta: ", theta)
        print("Theta diff: ", theta_diff)
        if theta_diff < 1e-7 or count > count_thr:
            return theta
        else:
            pre_theta = theta
            W = update_weight(theta, img_pnts_0, img_pnts_1, f_0)

        count += 1


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

    theta = calculate_projective_transformation(noised_img_pnts_0, noised_img_pnts_1)
    print("Final theta: ", theta)
    # TODO: Compare opencv homography result


if __name__ == "__main__":
    main()
