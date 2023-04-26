import numpy as np
import math
import cv2


def euler_angle_to_rot_mat(roll, pitch, yaw):
    x = np.cos(roll/180*math.pi)
    y = np.cos(pitch/180*math.pi)
    z = np.cos(yaw/180*math.pi)
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    return R_x * R_y * R_z


def create_curve_surface_points(row, col, z_scale):
    points = np.zeros((0, 3))
    for i in range(row+1):
        for j in range(col+1):
            x = i - row / 2
            y = j - col / 2
            z = x ** 2 * z_scale
            points = np.append(points, [[x, y, z]], axis=0)

    return points


def prepare_test_data(points_num: int):
    # Camera Extrinsic Parameter 0
    rot_mat_0 = euler_angle_to_rot_mat(0, 0, -30)
    print(rot_mat_0)
    trans_0 = (0, 0, 10)
    trans_vec_0 = np.eye(3) * np.matrix(trans_0).T
    # Camera Extrinsic Parameter 1
    rot_mat_1 = euler_angle_to_rot_mat(0, 0, 30)
    print(rot_mat_1)
    # input()
    trans_1 = (0, 0, 10)
    trans_vec_1 = np.eye(3) * np.matrix(trans_1).T
    points = create_curve_surface_points(10, 10, 0.2)
    print(points)
    rod_0 = cv2.Rodrigues(rot_mat_0)
    rod_1 = cv2.Rodrigues(rot_mat_1)
    print(rod_0)
    print(rod_1)

    f = 160
    width = 640
    height = 480
    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(np.array([[f, 0, pp[0]],
                            [0, f, pp[1]],
                            [0, 0, 1]], dtype = "double"))
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(points, rod_0, trans_0, camera_matrix, dist_coeffs)
    print(img_pnts_0)


if __name__ == '__main__':
    points_num = 10
    prepare_test_data(points_num)
