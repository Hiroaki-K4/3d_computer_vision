import csv
import json

import numpy as np


def normalize_camera_params(R, t):
    norm_R = np.zeros(R.shape)
    norm_t = np.zeros(t.shape)
    for i in range(R.shape[0]):
        norm_R[i] = np.dot(R[i], np.linalg.inv(R[0]))
        if i == 0:
            norm_t[i] = np.array([0, 0, 0])
        else:
            norm_t[i] = t[i] / t[1][2]

    return norm_R, norm_t


def split_camera_params(camera_params):
    K = np.zeros((len(camera_params), 3, 3))
    R = np.zeros((len(camera_params), 3, 3))
    t = np.zeros((len(camera_params), 3))
    for i in range(len(camera_params)):
        K[i] = np.array(camera_params[i]["K"])
        R[i] = np.array(camera_params[i]["R"])
        t[i] = np.array(camera_params[i]["t"])

    return K, R, t


def main(camera_parameters_file, tracked_2d_points_file, tracked_3d_points_file):
    with open(camera_parameters_file) as f:
        camera_params = json.load(f)

    with open(tracked_2d_points_file) as f:
        reader = csv.reader(f, delimiter=" ")
        points_2d = [row for row in reader]

    with open(tracked_3d_points_file) as f:
        points_3d = json.load(f)

    K, R, t = split_camera_params(camera_params)
    R, t = normalize_camera_params(R, t)


if __name__ == "__main__":
    camera_parameters_file = "camera_parameters.json"
    tracked_2d_points_file = "2d_tracked_points.csv"
    tracked_3d_points_file = "3d_tracked_points.json"
    main(camera_parameters_file, tracked_2d_points_file, tracked_3d_points_file)
