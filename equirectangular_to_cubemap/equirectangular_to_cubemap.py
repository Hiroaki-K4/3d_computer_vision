from tqdm import tqdm
import cv2
import numpy as np
import math


def get_theta(x, y):
    if y < 0:
        theta = (-1) * np.arctan2(y, x)
    else:
        theta = 2 * math.pi - np.arctan2(y, x)

    return theta


def create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z):
    map_x = np.zeros((output_sqr, output_sqr))
    map_y = np.zeros((output_sqr, output_sqr))
    for row in tqdm(range(output_sqr)):
        for col in range(output_sqr):
            x = row - output_sqr / 2.0
            y = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + z * z)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            if input_w <= ix:
                ix = ix - input_w
            if input_h <= iy:
                iy = iy - input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x):
    map_x = np.zeros((output_sqr, output_sqr))
    map_y = np.zeros((output_sqr, output_sqr))
    for row in tqdm(range(output_sqr)):
        for col in range(output_sqr):
            z = row - output_sqr / 2.0
            y = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + x * x)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            if input_w <= ix:
                ix = ix - input_w
            if input_h <= iy:
                iy = iy - input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y):
    map_x = np.zeros((output_sqr, output_sqr))
    map_y = np.zeros((output_sqr, output_sqr))
    for row in tqdm(range(output_sqr)):
        for col in range(output_sqr):
            z = row - output_sqr / 2.0
            x = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + x * x)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            if input_w <= ix:
                ix = ix - input_w
            if input_h <= iy:
                iy = iy - input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_cube_map(bottom_img, top_img, front_img, back_img, left_img, right_img, output_sqr):
    output_w = output_sqr * 4
    output_h = output_sqr * 3
    cube_map_img = np.zeros((output_h, output_w, 3), dtype = np.uint8)

    cube_map_img[output_sqr*2:output_h, output_sqr:output_sqr*2] = bottom_img
    cube_map_img[0:output_sqr, output_sqr:output_sqr*2] = top_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr:output_sqr*2] = front_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr*3:output_w] = back_img
    cube_map_img[output_sqr:output_sqr*2, 0:output_sqr] = left_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr*2:output_sqr*3] = right_img

    return cube_map_img


def main(image_path):
    img = cv2.imread(image_path)
    output_sqr = 800
    normalized_f = 1.0
    input_w = img.shape[1]
    input_h = img.shape[0]

    # Create bottom image
    z = output_sqr / (2.0 * normalized_f)
    bottom_map_x , bottom_map_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    bottom_img = cv2.remap(img, bottom_map_x.astype('float32'), bottom_map_y.astype('float32'), cv2.INTER_CUBIC)
    cv2.imwrite("bottom.png", bottom_img)

    # Create top image
    z = (-1) * (output_sqr / (2.0 * normalized_f))
    top_map_x , top_map_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    top_img = cv2.remap(img, top_map_x.astype('float32'), top_map_y.astype('float32'), cv2.INTER_CUBIC)
    top_img = cv2.flip(top_img, 0)
    cv2.imwrite("top.png", top_img)

    # Create front image
    x = (-1) * (output_sqr / (2.0 * normalized_f))
    front_map_x , front_map_y = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    front_img = cv2.remap(img, front_map_x.astype('float32'), front_map_y.astype('float32'), cv2.INTER_CUBIC)
    cv2.imwrite("front.png", front_img)

    # Create back image
    x = output_sqr / (2.0 * normalized_f)
    back_map_x , back_map_y = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    back_img = cv2.remap(img, back_map_x.astype('float32'), back_map_y.astype('float32'), cv2.INTER_CUBIC)
    back_img = cv2.flip(back_img, 1)
    cv2.imwrite("back.png", back_img)

    # Create left image
    y = (-1) * (output_sqr / (2.0 * normalized_f))
    left_map_x , left_map_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    left_img = cv2.remap(img, left_map_x.astype('float32'), left_map_y.astype('float32'), cv2.INTER_CUBIC)
    left_img = cv2.flip(left_img, 1)
    cv2.imwrite("left.png", left_img)

    # Create right image
    y = output_sqr / (2.0 * normalized_f)
    right_map_x , right_map_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    right_img = cv2.remap(img, right_map_x.astype('float32'), right_map_y.astype('float32'), cv2.INTER_CUBIC)
    cv2.imwrite("right.png", right_img)

    # Create cube map image
    cube_map_img = create_cube_map(bottom_img, top_img, front_img, back_img, left_img, right_img, output_sqr)
    cv2.imwrite("cube_map.png", cube_map_img)


if __name__ == '__main__':
    image_path = "original_equirectangular.png"
    main(image_path)
