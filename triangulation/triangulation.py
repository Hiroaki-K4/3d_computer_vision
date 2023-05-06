import numpy as np
import sys
sys.path.insert(0, '/home/hiroakik4/mypro/3d_computer_vision/fundamental_matrix')
import calculate_f_matrix


def calculate_camera_matrix_from_RT(R, T, f):
    focal_arr = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P = np.dot(focal_arr, I)

    Rt = np.dot(R.T, T)
    R_T = R.T
    moved_arr = np.array([[R_T[0, 0], R_T[0, 1], R_T[0, 2], Rt[0, 0]],
                        [R_T[1, 0], R_T[1, 1], R_T[1, 2], Rt[1, 0]],
                        [R_T[2, 0], R_T[2, 1], R_T[2, 2], Rt[2, 0]]])
    P_after = np.dot(focal_arr, moved_arr)

    return P, P_after


def simple_triangulation(P_0, P_1, f_0, points_0, points_1):
    x_0 = points_0[0]
    y_0 = points_0[1]
    x_1 = points_1[0]
    y_1 = points_1[1]
    T = np.array([[f_0*P_0[0,0]-x_0*P_0[2,0], f_0*P_0[0,1]-x_0*P_0[2,1], f_0*P_0[0,2]-x_0*P_0[2,2]],
                [f_0*P_0[1,0]-y_0*P_0[2,0], f_0*P_0[1,1]-y_0*P_0[2,1], f_0*P_0[1,2]-y_0*P_0[2,2]],
                [f_0*P_1[0,0]-x_1*P_1[2,0], f_0*P_1[0,1]-x_1*P_1[2,1], f_0*P_1[0,2]-x_1*P_1[2,2]],
                [f_0*P_1[1,0]-y_1*P_1[2,0], f_0*P_1[1,1]-y_1*P_1[2,1], f_0*P_1[1,2]-y_1*P_1[2,2]]])
    p = np.array([f_0*P_0[0,3]-x_0*P_0[2,3], f_0*P_0[1,3]-y_0*P_0[2,3], f_0*P_1[0,3]-x_1*P_1[2,3], f_0*P_1[1,3]-y_1*P_1[2,3]])
    ans = (-1) * np.dot(np.dot(np.linalg.inv(np.dot(T.T, T)), T.T), p)

    return np.array([ans[0], ans[1], ans[2]])


def main():
    draw_test_data = False
    draw_epipolar = False
    img_pnts_0, img_pnts_1, F_true, rot_1_to_2, trans_1_to_2_in_camera_coord = calculate_f_matrix.prepare_test_data(draw_test_data, draw_epipolar)
    f = 160
    P_0, P_1 = calculate_camera_matrix_from_RT(rot_1_to_2, trans_1_to_2_in_camera_coord, f)
    print("P_0: ", P_0)
    print("P_1: ", P_1)
    f_0 = 1280
    # pos = simple_triangulation(P_0, P_1, f_0, img_pnts_0[0][0], img_pnts_1[0][0])
    pos = simple_triangulation(P_0, P_1, f_0, img_pnts_0[15][0], img_pnts_1[15][0])
    print("result: ", pos)


if __name__ == '__main__':
    main()
