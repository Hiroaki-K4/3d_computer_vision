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

test_elliptic_fitting() {
    cd elliptic_fitting
    python3 draw_elliptic.py NotShow
    check_result "draw_elliptic.py"
    python3 elliptic_fitting_by_fns.py NotShow
    check_result "elliptic_fitting_by_fns.py"
    python3 elliptic_fitting_by_least_squares.py NotShow
    check_result "elliptic_fitting_by_least_squares.py"
    python3 elliptic_fitting_by_renormalization.py NotShow
    check_result "elliptic_fitting_by_renormalization.py"
    python3 elliptic_fitting_by_weighted_repetition.py NotShow
    check_result "elliptic_fitting_by_weighted_repetition.py"
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

python3 -m pip install -r requirements.txt

if [ $# -eq 1 ]; then
    if [ $1 = "elliptic_analysis" ]; then
        test_elliptic_analysis
    elif [ $1 = "elliptic_fitting" ]; then
        test_elliptic_fitting
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
    else
        echo "Argument is wrong"
        exit 1
    fi

else
    test_elliptic_analysis
    test_elliptic_fitting
    test_equirectangular_to_cubemap
    test_equirectangular_to_sphere
    test_fundamental_matrix
    test_homography_decomposition
    test_projective_transformation
    test_triangulation
fi
