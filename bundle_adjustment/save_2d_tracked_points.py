import csv
import glob
import os

import cv2
from natsort import natsorted


def main(image_dir, points_file, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    input_paths = os.path.join(image_dir, "*.jpg")
    input_imgs = natsorted(glob.glob(input_paths))
    with open(points_file) as f:
        reader = csv.reader(f, delimiter=" ")
        points = [row for row in reader]

    for idx in range(1, len(input_imgs)):
        img = cv2.imread(input_imgs[idx])
        for point in points:
            x = float(point[(idx - 1) * 2])
            y = float(point[(idx - 1) * 2 + 1])
            if x == -1.0 or y == -1.0:
                continue
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), thickness=-1)

        output_path = os.path.join(save_dir, os.path.basename(input_imgs[idx]))
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    image_dir = "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/images_jpg"
    points_file = "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/2d_tracked_points.csv"
    save_dir = "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/init_points"
    main(image_dir, points_file, save_dir)
