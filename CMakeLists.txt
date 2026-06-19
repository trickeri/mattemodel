cmake_minimum_required(VERSION 3.20)
project(mattemodel CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Inference backend: cuda (ONNX Runtime, default) or vulkan (ncnn).
set(BACKEND "cuda" CACHE STRING "inference backend: cuda | vulkan")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

if(BACKEND STREQUAL "vulkan")
    # ncnn from its install tree (pass -DCMAKE_PREFIX_PATH=.../ncnn/install).
    find_package(ncnn REQUIRED)
    add_executable(mattemodel-server src/server.cpp src/matte_engine_ncnn.cpp)
    target_link_libraries(mattemodel-server PRIVATE ncnn)
elseif(BACKEND STREQUAL "cuda")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ORT REQUIRED libonnxruntime)
    add_executable(mattemodel-server
        src/server.cpp src/matte_engine_ort.cpp src/matte_rvm.cpp)
    target_include_directories(mattemodel-server PRIVATE ${ORT_INCLUDE_DIRS})
    target_link_directories(mattemodel-server PRIVATE ${ORT_LIBRARY_DIRS})
    target_link_libraries(mattemodel-server PRIVATE ${ORT_LIBRARIES})
else()
    message(FATAL_ERROR "BACKEND must be 'cuda' or 'vulkan', got: ${BACKEND}")
endif()

target_include_directories(mattemodel-server PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mattemodel-server PRIVATE ${OpenCV_LIBS} Threads::Threads)
message(STATUS "mattemodel backend: ${BACKEND}")
