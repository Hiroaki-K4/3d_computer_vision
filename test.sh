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

source 3d/bin/activate

# Test elliptic_analysis
cd elliptic_analysis
python3 calculate_ellipse_intersection.py NotShow
check_result "calculate_ellipse_intersection.py"
python3 calculate_foot_of_perpendicular_line.py NotShow
check_result "calculate_foot_of_perpendicular_line.py"
python3 reconstruct_support_plane.py NotShow
check_result "reconstruct_support_plane.py"
cd ../

