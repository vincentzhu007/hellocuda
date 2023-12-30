cmake_minimum_required(VERSION 3.14)
project(cuda_by_example_zgd LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

add_subdirectory(utils)
add_subdirectory(startup)
add_subdirectory(sgemm)

