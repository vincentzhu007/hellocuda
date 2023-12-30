include_directories(${CMAKE_SOURCE_DIR}/utils/)

add_executable(hello_cuda hello_cuda.cu)
add_executable(add add.cu)
add_executable(block_and_grid block_and_grid.cu)
add_executable(cuda_info cuda_info.cu)
