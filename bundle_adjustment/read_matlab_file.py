import json

import scipy.io


def main(input_path: str, output_path: str):
    mat = scipy.io.loadmat(input_path)
    camera_mat = mat["P"][0]
    camera_mat_list = []
    for i in range(camera_mat.shape[0]):
        camera_mat_list.append(camera_mat[i].tolist())
    input_data = {"P": camera_mat_list}
    with open(output_path, "w") as f:
        json.dump(input_data, f, indent=2)


if __name__ == "__main__":
    input_path = "dinosaur.mat"
    output_path = "dinosaur.json"
    main(input_path, output_path)
