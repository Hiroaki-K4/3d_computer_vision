import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy


def disassemble_camera_matrix(P):
    Q = P[:, :3]
    q = P[:, 3]
    if np.linalg.det(Q) < 0:
        Q = (-1) * Q
        q = (-1) * q

    t = np.dot(-np.linalg.inv(Q), q)
    C = scipy.linalg.cholesky(np.linalg.inv(np.dot(Q, Q.T)))
    K = np.linalg.inv(C)
    R = np.dot(Q.T, C.T)

    return K, R, t


def main(camera_matrix_file, show_flag):
    with open(camera_matrix_file) as f:
        camera_mat = json.load(f)

    camera_p = camera_mat["P"]
    params = []
    show_pos_x = []
    show_pos_y = []
    show_pos_z = []
    for idx in range(len(camera_p)):
        P = np.array(camera_p[idx])
        K, R, t = disassemble_camera_matrix(P)
        show_pos_x.append(t[0])
        show_pos_y.append(t[1])
        show_pos_z.append(t[2])
        param = {}
        param["K"] = K.tolist()
        param["R"] = R.tolist()
        param["t"] = t.tolist()
        params.append(param)

    output_path = os.path.join(
        os.path.dirname(camera_matrix_file), "camera_parameters.json"
    )
    with open(output_path, "w") as f:
        json.dump(params, f, indent=4)

    if show_flag:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(show_pos_x, show_pos_y, show_pos_z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D positions of camera")
        plt.show()


if __name__ == "__main__":
    show_flag = True
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        show_flag = False
    camera_matrix_file = (
        "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/dinosaur.json"
    )
    main(camera_matrix_file, show_flag)
