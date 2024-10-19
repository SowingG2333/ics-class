# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(styletransfer-applet)

add_definitions(-D__APPLET__)
option(NPU_HOST "Flag for Atlas 200DK device" OFF)

# Compile options
add_compile_options(-std=c++17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ".")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# Header path
include_directories( /usr/local/include )

# add host lib path
link_directories( /usr/local/lib )

add_library(styletransfer SHARED
	src/styletransfer.cc)

add_executable( styleapplet
    src/styletransfer.cc
    src/main.cc)

target_link_libraries(styleapplet
    airemote stdc++ opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio )

target_link_libraries(styletransfer
    airemote stdc++ opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio )
install(TARGETS styleapplet DESTINATION ${CMAKE_BINARY_DIR})

