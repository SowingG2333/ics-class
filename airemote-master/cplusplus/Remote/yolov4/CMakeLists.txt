# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(cartoon_picture)

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

add_executable( yoloapp
       	src/yolov4.cc
	src/post_process.cc
        src/main.cc)

target_link_libraries( yoloapp
       airemote stdc++ opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio)

install(TARGETS yoloapp DESTINATION ${CMAKE_BINARY_DIR})

#enable_testing()

#set(TCP_REMOTE "tcp://${REMOTE_HOST}:8888")
#set(TEST_FILE "${PICTURE_DIR}/rabit.jpg")
#set(FILE_REMOTE "${MODEL_DIR}/yolov4/yolov4_416x416.om")

#add_test(test_yolov4 yoloapp -r${TCP_REMOTE} -i${TEST_FILE})

#if (NPU_HOST)
#    set(FILE_REMOTE "file:///home/HwHiAiUser/models/yolov4/yolov4.om")
#    add_test(test_yolov4_npu yoloapp -r${FILE_REMOTE} -i${TEST_FILE})
#endif()
