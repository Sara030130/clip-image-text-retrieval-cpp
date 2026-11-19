cmake_minimum_required(VERSION 3.20)
project(clip_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 5 REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(clip_infer clip_infer.cpp)

target_link_libraries(clip_infer PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    "-framework Accelerate"
)

target_include_directories(clip_infer PRIVATE ${OpenCV_INCLUDE_DIRS})

set_target_properties(clip_infer PROPERTIES
    BUILD_RPATH "/opt/homebrew/opt/onnxruntime/lib;/opt/homebrew/opt/opencv/lib"
    INSTALL_RPATH "/opt/homebrew/opt/onnxruntime/lib;/opt/homebrew/opt/opencv/lib"
)
