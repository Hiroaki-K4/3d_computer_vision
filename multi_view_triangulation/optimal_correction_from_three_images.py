import sys

sys.path.append("../")
from prepare_test_data_utils import prepare_test_data


def optimal_correction_from_three_images(pnts_0, pnts_1, pnts_2):
    print(pnts_0)
    E_0 = sys.float_info.max


def main():
    rot_euler_deg_0 = [0, -45, 0]
    rot_euler_deg_1 = [0, 0, 0]
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
        img_pnts_2,
        noised_img_pnts_0,
        noised_img_pnts_1,
        noised_img_pnts_2,
        points_3d,
    ) = prepare_test_data.prepare_test_data_three_images(
        False,
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
    prepare_test_data.optimal_correction_from_three_images(
        noised_img_pnts_0, noised_img_pnts_1, noised_img_pnts_2
    )


if __name__ == "__main__":
    main()
