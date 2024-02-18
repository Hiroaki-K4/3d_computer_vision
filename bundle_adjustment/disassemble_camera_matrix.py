import json
import numpy as np


def main(camera_matrix_file):
    with open(camera_matrix_file) as f:
        camera_mat = json.load(f)

    camera_p = camera_mat["P"]

    for idx in range(len(camera_p)):
        print(np.array(camera_p[idx]).shape)


if __name__ == "__main__":
    camera_matrix_file = (
        "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/dinosaur.json"
    )
    main(camera_matrix_file)
