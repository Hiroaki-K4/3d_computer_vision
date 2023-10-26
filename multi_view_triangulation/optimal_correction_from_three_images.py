import sys

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def main():
    # prepare_test_data_three_images()
    rot_euler_deg_0 = [0, 0, 0]
    rot_euler_deg_1 = [0, -45, 0]
    rot_euler_deg_2 = [0, 45, 0]
    T_0_in_camera_coord = [0, 0, 10]
    T_1_in_camera_coord = [0, 0, 10]
    T_2_in_camera_coord = [0, 0, 10]
    f = 160
    width = 640
    height = 480
    (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,
        noised_img_pnts_1,
        F_true,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
        points_3d,
    ) = prepare_test_data.prepare_test_data_three_images(
        True,
        "PLANE",
        rot_euler_deg_0,
        rot_euler_deg_1,
        rot_euler_deg_2,
        T_0_in_camera_coord,
        T_1_in_camera_coord,
        T_2_in_camera_coord,
        f,
        width,
        height,
    )
    print("ok")


if __name__ == "__main__":
    main()
