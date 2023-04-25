import numpy as np
import math

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
            # TODO: Fix invalid curve points

    print(points)
    return points


def prepare_test_data(points_num: int):
    # Camera Extrinsic Parameter 0
    rot_mat_0 = euler_angle_to_rot_mat(0, 0, -30)
    print(rot_mat_0)
    trans_0 = (0, 0, 10)
    # Camera Extrinsic Parameter 1
    rot_mat_1 = euler_angle_to_rot_mat(0, 0, 30)
    trans_1 = (0, 0, 10)
    points = create_curve_surface_points(10, 10, 0.2)


if __name__ == '__main__':
    points_num = 10
    prepare_test_data(points_num)
