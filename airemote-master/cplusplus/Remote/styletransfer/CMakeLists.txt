# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(styletransfer)

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

add_executable( styleapp
    src/styletransfer.cc
    src/main.cc)

target_link_libraries( styleapp 
    airemote stdc++ opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio )

install(TARGETS styleapp DESTINATION ${CMAKE_BINARY_DIR})

#set(TEST_FILE "${PICTURE_DIR}/rabit.jpg")
#set(TCP_REMOTE "tcp://${REMOTE_HOST}:6666")
#set(FILE_REMOTE "${MODEL_DIR}/style_transfer/picasso.om")

#add_test(test_styleapp styleapp -r${TCP_REMOTE} -i ${TEST_FILE})
#
#if (NPU_HOST)
#    set(FILE_REMOTE "file:///home/HwHiAiUser/models/styletransfer/picasso.om")
#    add_test(test_styleapp styleapp -r${FILE_REMOTE} -i ${TEST_FILE})
#endif()

