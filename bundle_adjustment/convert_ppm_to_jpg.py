import glob
import os

from PIL import Image


def main(input_dir, output_dir):
    input_paths = os.path.join(input_dir, "*.ppm")
    input_ppms = glob.glob(input_paths)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for input_ppm in input_ppms:
        img = Image.open(input_ppm)
        filename = os.path.splitext(os.path.basename(input_ppm))[0]
        save_name = filename + ".jpg"
        output_path = os.path.join(output_dir, save_name)
        img.save(output_path)


if __name__ == "__main__":
    input_dir = "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/images"
    output_dir = (
        "/home/hiroakik4/mypro2/3d_computer_vision/bundle_adjustment/images_jpg"
    )
    main(input_dir, output_dir)
