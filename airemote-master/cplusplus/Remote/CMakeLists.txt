cmake_minimum_required(VERSION 3.0)
project(AiremoteExamples)

option(NPU_HOST "Flag for Atlas 200DK device" OFF)

message("top: CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("top: CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message("top: CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message("top: CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

# Compile options
add_compile_options(-std=c++17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )


add_subdirectory(googlenet)
add_subdirectory(googlenet-applet)
add_subdirectory(cartoon)
add_subdirectory(yolov4)
add_subdirectory(styletransfer)
add_subdirectory(styletransfer-applet)

enable_testing()
set(REMOTE_HOST "opi")
set(PICTURE_DIR "$ENV{HOME}/test/pictures")
set(MODEL_DIR   "/home/HwHiAiUser/models")
set(TEST_FILE   "${PICTURE_DIR}/rabit.jpg")

add_test(test_gnet gnetapp tcp://${REMOTE_HOST}:5530 ${PICTURE_DIR}/rabit.jpg)
set_tests_properties(test_gnet PROPERTIES PASS_REGULAR_EXPRESSION "Class name: hare")

add_test(test_style styleapp -rtcp://${REMOTE_HOST}:5550 -i ${PICTURE_DIR}/rabit.jpg)
#add_test(test_yolov4 yoloapp -rtcp://${REMOTE_HOST}:5565 -i ${PICTURE_DIR}/traffic.jpg)

if (NPU_HOST)
    set(FILE_REMOTE "file://${MODEL_DIR}/googlenet/googlenet.om")
    add_test(test_gnet_npu gnetapp ${FILE_REMOTE} ${PICTURE_DIR}/rabit.jpg)
    set_tests_properties(test_gnet_npu PROPERTIES PASS_REGULAR_EXPRESSION "Class name: hare")

    set(FILE_REMOTE "file://${MODEL_DIR}/style_transfer/picasso.om")
    add_test(test_style_npu styleapp -r ${FILE_REMOTE} -i ${TEST_FILE})

    set(FILE_REMOTE "file://${MODEL_DIR}/yolov4_416/yolov4_416x416.om")
    add_test(test_yolov4_npu yoloapp -r ${FILE_REMOTE} -i ${TEST_FILE})
endif()

# Ends.
