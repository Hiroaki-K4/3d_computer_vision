import csv
import json


def main(tracked_2d_points_file, camera_matrix_file, img_width):
    with open(tracked_2d_points_file) as f:
        reader = csv.reader(f, delimiter=" ")
        points_2d = [row for row in reader]

    with open(camera_matrix_file) as f:
        camera_mat = json.load(f)["P"]

    for point in points_2d:
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
        print(point_0)
        print(point_1)
        P_0 = camera_mat[i]
        P_1 = camera_mat[j]
        print(P_0)
        print(P_1)
        # TODO Add triangulation
        input()


if __name__ == "__main__":
    tracked_2d_points_file = "2d_tracked_points.csv"
    camera_matrix_file = "dinosaur.json"
    img_width = 720
    main(tracked_2d_points_file, camera_matrix_file, img_width)
