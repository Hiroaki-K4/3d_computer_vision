#!/bin/bash

COLOR_RESET="\033[m"
COLOR_RED="\033[31m"
COLOR_GREEN="\033[32m"

check_result() {
	if [ $? -ne 0 ]; then
		printf "${COLOR_RED}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [ERROR]'
		exit 1
	fi
	printf "${COLOR_GREEN}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [OK]'
}

test_elliptic_analysis() {
    cd elliptic_analysis
    python3 calculate_ellipse_intersection.py NotShow
    check_result "calculate_ellipse_intersection.py"
    python3 calculate_foot_of_perpendicular_line.py NotShow
    check_result "calculate_foot_of_perpendicular_line.py"
    python3 reconstruct_support_plane.py NotShow
    check_result "reconstruct_support_plane.py"
    cd ../
}

test_elliptic_fitting_by_fns() {
    cd elliptic_fitting
    python3 draw_elliptic.py NotShow
    check_result "draw_elliptic.py"
    python3 elliptic_fitting_by_fns.py NotShow
    check_result "elliptic_fitting_by_fns.py"
    cd ../
}

test_elliptic_fitting_by_least_squares() {
    cd elliptic_fitting
    python3 elliptic_fitting_by_least_squares.py NotShow
    check_result "elliptic_fitting_by_least_squares.py"
    cd ../
}

test_elliptic_fitting_by_renormalization() {
    cd elliptic_fitting
    python3 elliptic_fitting_by_renormalization.py NotShow
    check_result "elliptic_fitting_by_renormalization.py"
    cd ../
}

test_elliptic_fitting_by_weighted_repetition() {
    cd elliptic_fitting
    python3 elliptic_fitting_by_weighted_repetition.py NotShow
    check_result "elliptic_fitting_by_weighted_repetition.py"
    cd ../
}

test_remove_outlier_by_ransac() {
    cd elliptic_fitting
    python3 remove_outlier_by_ransac.py NotShow
    check_result "remove_outlier_by_ransac.py"
    cd ../
}

test_equirectangular_to_cubemap() {
    cd equirectangular_to_cubemap
    python3 equirectangular_to_cubemap.py
    check_result "equirectangular_to_cubemap.py"
    cd ../
}

test_equirectangular_to_sphere() {
    cd equirectangular_to_sphere
    python3 equirectangular_to_sphere.py NotShow
    check_result "equirectangular_to_sphere.py"
    cd ../
}

test_fundamental_matrix() {
    cd fundamental_matrix
    python3 calculate_f_matrix.py
    check_result "calculate_f_matrix.py"
    cd ../
}

test_homography_decomposition() {
    cd homography_decomposition
    python3 decompose_homography.py
    check_result "decompose_homography.py"
    cd ../
}

test_projective_transformation() {
    cd projective_transformation
    python3 calculate_projective_trans_by_weighted_repetition.py NotShow
    check_result "calculate_projective_trans_by_weighted_repetition.py"
    cd ../
}

test_triangulation() {
    cd triangulation
    python3 triangulation.py NotShow
    check_result "triangulation.py"
    python3 planar_triangulation.py
    check_result "planar_triangulation.py"
    cd ../
}

test_perspective_projection_camera_calibration() {
    cd perspective_projection_camera_calibration
    python3 calibrate_perspective_camera_by_primary_method.py NotShow
    check_result "calibrate_perspective_camera_by_primary_method.py"
    cd ../
}

test_bundle_adjustment() {
    cd bundle_adjustment
    python3 disassemble_camera_matrix.py NotShow
    check_result "disassemble_camera_matrix.py"
    python3 calculate_3d_points_by_triangulation.py NotShow
    check_result "calculate_3d_points_by_triangulation.py"
    python3 disassemble_camera_matrix.py NotShow
    check_result "disassemble_camera_matrix.py"
    python3 run_bundle_adjustment.py
    check_result "run_bundle_adjustment.py"
    cd ../
}

python3 -m pip install -r requirements.txt

if [ $# -eq 1 ]; then
    if [ $1 = "elliptic_analysis" ]; then
        test_elliptic_analysis
    elif [ $1 = "elliptic_fitting_by_fns" ]; then
        test_elliptic_fitting_by_fns
    elif [ $1 = "elliptic_fitting_by_least_squares" ]; then
        test_elliptic_fitting_by_least_squares
    elif [ $1 = "elliptic_fitting_by_renormalization" ]; then
        test_elliptic_fitting_by_renormalization
    elif [ $1 = "elliptic_fitting_by_weighted_repetition" ]; then
        test_elliptic_fitting_by_weighted_repetition
    elif [ $1 = "remove_outlier_by_ransac" ]; then
        test_remove_outlier_by_ransac
    elif [ $1 = "equirectangular_to_cubemap" ]; then
        test_equirectangular_to_cubemap
    elif [ $1 = "equirectangular_to_sphere" ]; then
        test_equirectangular_to_sphere
    elif [ $1 = "fundamental_matrix" ]; then
        test_fundamental_matrix
    elif [ $1 = "homography_decomposition" ]; then
        test_homography_decomposition
    elif [ $1 = "projective_transformation" ]; then
        test_projective_transformation
    elif [ $1 = "triangulation" ]; then
        test_triangulation
    elif [ $1 = "perspective_projection_camera_calibration" ]; then
        test_perspective_projection_camera_calibration
    elif [ $1 = "bundle_adjustment" ]; then
        test_bundle_adjustment
    else
        echo "Argument is wrong"
        exit 1
    fi

else
    test_elliptic_analysis
    test_elliptic_fitting_by_fns
    test_elliptic_fitting_by_least_squares
    test_elliptic_fitting_by_renormalization
    test_elliptic_fitting_by_weighted_repetition
    test_remove_outlier_by_ransac
    test_equirectangular_to_cubemap
    test_equirectangular_to_sphere
    test_fundamental_matrix
    test_homography_decomposition
    test_projective_transformation
    test_triangulation
    test_perspective_projection_camera_calibration
    test_bundle_adjustment
fi
