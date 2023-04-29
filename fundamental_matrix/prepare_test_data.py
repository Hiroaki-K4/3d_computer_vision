import numpy as np
import math
import cv2


def euler_angle_to_rot_mat(x_deg, y_deg, z_deg):
    x = x_deg / 180 * math.pi
    y = y_deg / 180 * math.pi
    z = z_deg / 180 * math.pi
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    return np.dot(np.dot(R_x, R_y), R_z)


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
    rot_mat_0 = euler_angle_to_rot_mat(0, -30, 0)
    print(rot_mat_0)
    trans_0 = (0, 0, 10)
    trans_vec_0 = np.eye(3) * np.matrix(trans_0).T
    rot_mat_1 = euler_angle_to_rot_mat(0, 30, 0)
    trans_1 = (0, 0, 10)
    trans_vec_1 = np.eye(3) * np.matrix(trans_1).T
    points = create_curve_surface_points(10, 10, 0.2)
    print(points)
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)
    print(rodri_0)
    print(rodri_1)

    f = 160
    width = 640
    height = 480
    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(np.array([[f, 0, pp[0]],
                            [0, f, pp[1]],
                            [0, 0, 1]], dtype = "double"))
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(points, rodri_0, trans_0, camera_matrix, dist_coeffs)
    print(img_pnts_0[5])
    img_pnts_1, jac = cv2.projectPoints(points, rodri_1, trans_1, camera_matrix, dist_coeffs)
    print(img_pnts_1[5])
    print(trans_0)
    print(trans_1)
    # input()


    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    for pnt in img_pnts_0:
        cv2.circle(img_0, (int(pnt[0][0]), int(pnt[0][1])), 3, (0, 0, 0), -1)
    for pnt in img_pnts_1:
        cv2.circle(img_1, (int(pnt[0][0]), int(pnt[0][1])), 3, (255, 0, 0), -1)

    cv2.imshow("CAM0", cv2.resize(img_0, None, fx = 0.5, fy = 0.5))
    cv2.imshow("CAM1", cv2.resize(img_1, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)



if __name__ == '__main__':
    points_num = 10
    prepare_test_data(points_num)
