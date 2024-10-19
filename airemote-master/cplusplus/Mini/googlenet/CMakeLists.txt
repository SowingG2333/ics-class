# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(googlenet_imagenet_picture)

# Compile options
add_compile_options(-std=c++17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ".")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# Header path
include_directories(
    ../../include/
    /usr/local/include
)

# add host lib path
link_directories(
    /usr/local/lib
)

add_executable(gnetapp_mini
	src/googlenet.cc
    src/main.cc)

target_link_libraries(gnetapp_mini
      atlasmini stdc++ opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui 
      )

install(TARGETS gnetapp_mini DESTINATION ${CMAKE_BINARY_DIR})

enable_testing()
add_test(test_mini gnetapp_mini $ENV{HOME}/googlenet.om $ENV{HOME}/rabit.jpg)
set_tests_properties(test_mini PROPERTIES PASS_REGULAR_EXPRESSION "Class name: hare")

