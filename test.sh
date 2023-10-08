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

python3 -m pip install -r requirements.txt

# Test elliptic_analysis
cd elliptic_analysis
python3 calculate_ellipse_intersection.py NotShow
check_result "calculate_ellipse_intersection.py"
python3 calculate_foot_of_perpendicular_line.py NotShow
check_result "calculate_foot_of_perpendicular_line.py"
python3 reconstruct_support_plane.py NotShow
check_result "reconstruct_support_plane.py"
cd ../

# Test elliptic_fitting
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

# Test equirectangular_to_cubemap
cd equirectangular_to_cubemap
python3 equirectangular_to_cubemap.py
check_result "equirectangular_to_cubemap.py"
cd ../

# Test equirectangular_to_sphere
cd equirectangular_to_sphere
python3 equirectangular_to_sphere.py
check_result "equirectangular_to_sphere.py"
cd ../