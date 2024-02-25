import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../triangulation")
from triangulation import simple_triangulation


def calculate_3d_points_by_triangulation(points_2d, camera_mat, img_width):
    points_3d = np.zeros((len(points_2d), 3))
    for idx in range(len(points_2d)):
        point = points_2d[idx]
        point_0 = []
        point_1 = []
        for i in range(int((len(point) - 1) / 2)):
            x_0 = point[i * 2]
            y_0 = point[i * 2 + 1]
            if x_0 == -1.0 or y_0 == -1.0:
                continue
            point_0.append(float(x_0))
            point_0.append(float(y_0))

            p_find = False
            for j in range(int((len(point) - 1) / 2)):
                if i == j:
                    continue
                x_1 = point[j * 2]
                y_1 = point[j * 2 + 1]
                if x_1 == -1.0 or y_1 == -1.0:
                    continue
                point_1.append(float(x_1))
                point_1.append(float(y_1))
                p_find = True
                break

            if p_find:
                break
            else:
                raise RuntimeError("There are no points for triangulation.")

        P_0 = np.array(camera_mat[i])
        P_1 = np.array(camera_mat[j])
        pos = simple_triangulation(P_0, P_1, img_width, point_0, point_1)
        points_3d[idx] = pos

    return points_3d


def main(
    tracked_2d_points_file,
    camera_matrix_file,
    camera_params_file,
    img_width,
    output_3d_points_file,
    show_flag,
):
    with open(tracked_2d_points_file) as f:
        reader = csv.reader(f, delimiter=" ")
        points_2d = [row for row in reader]

    with open(camera_matrix_file) as f:
        camera_mat = json.load(f)["P"]

    with open(camera_params_file) as f:
        camera_params = json.load(f)

    camera_pos_x = []
    camera_pos_y = []
    camera_pos_z = []
    for param in camera_params:
        camera_pos_x.append(param["t"][0])
        camera_pos_y.append(param["t"][1])
        camera_pos_z.append(param["t"][2])

    points_3d = calculate_3d_points_by_triangulation(points_2d, camera_mat, img_width)
    points_3d_dict = {"points_3d": points_3d.tolist()}
    with open(output_3d_points_file, "w") as f:
        json.dump(points_3d_dict, f, indent=4)

    if show_flag:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D positions of tracked points")
        plt.show()


if __name__ == "__main__":
    show_flag = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    tracked_2d_points_file = "2d_tracked_points.csv"
    camera_matrix_file = "dinosaur.json"
    camera_params_file = "camera_parameters.json"
    output_3d_points_file = "3d_tracked_points.json"
    img_width = 720
    main(
        tracked_2d_points_file,
        camera_matrix_file,
        camera_params_file,
        img_width,
        output_3d_points_file,
        show_flag,
    )
