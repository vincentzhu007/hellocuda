cmake_minimum_required(VERSION 3.14)
project(cuda_by_example_zgd LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

add_executable(hello_world hello_world.cc)
add_executable(hello_cuda hello_cuda.cu)
add_executable(add add.cu)
add_executable(block_and_grid block_and_grid.cu)
add_executable(cuda_info cuda_info.cu)


