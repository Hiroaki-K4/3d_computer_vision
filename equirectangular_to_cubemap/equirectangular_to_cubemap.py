from tqdm import tqdm
import cv2
import numpy as np
import math


def get_theta(x, y):
    if y < 0:
        theta = -1 * np.arctan2(y, x)
    else:
        theta = math.pi + (math.pi - np.arctan2(y, x))

    return theta


def create_equirectangler_to_bottom_map(input_w, input_h, output_sqr, normalized_f):
    map_x = np.zeros((output_sqr, output_sqr))
    map_y = np.zeros((output_sqr, output_sqr))
    for row in tqdm(range(output_sqr)):
        for col in range(output_sqr):
            x = float(row - output_sqr / 2.0)
            y = float(col - output_sqr / 2.0)
            # TODO: If you change z -> -z, you can make top image map
            z = float(output_sqr / (2.0 * normalized_f))

            rho = float(np.sqrt(x * x + y * y + z * z))
            norm_theta = float(get_theta(x, y) / (2 * math.pi))
            norm_phi = float((math.pi - np.arccos(z / rho)) / math.pi)
            ix = float(norm_theta * input_w)
            iy = float(norm_phi * input_h)

            if input_w <= ix:
                ix = ix - input_w
            if input_h <= iy:
                iy = iy - input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def main(image_path):
    img = cv2.imread(image_path)
    map_x , map_y = create_equirectangler_to_bottom_map(img.shape[1], img.shape[0], 800, 1.0)
    bottom_img = cv2.remap(img, map_x.astype('float32'), map_y.astype('float32'), cv2.INTER_CUBIC)
    cv2.imwrite("bottom_img.png", bottom_img)


if __name__ == '__main__':
    image_path = "equi_sample.png"
    main(image_path)
