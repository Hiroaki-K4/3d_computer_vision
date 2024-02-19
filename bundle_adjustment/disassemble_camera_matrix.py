import json
import os

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


def main(camera_matrix_file):
    with open(camera_matrix_file) as f:
        camera_mat = json.load(f)

    camera_p = camera_mat["P"]
    params = []
    for idx in range(len(camera_p)):
        P = np.array(camera_p[idx])
        K, R, t = disassemble_camera_matrix(P)
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


if __name__ == "__main__":
    camera_matrix_file = (
        "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/dinosaur.json"
    )
    main(camera_matrix_file)
