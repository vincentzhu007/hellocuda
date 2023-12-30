
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_executable(sgemm_cuda_0 sgemm_cuda_0.cu)
target_link_libraries(sgemm_cuda_0 PUBLIC cuda_utils)

add_executable(sgemm_cuda_1 sgemm_cuda_1.cu)
target_link_libraries(sgemm_cuda_1 PUBLIC cuda_utils)

add_executable(sgemm_cuda_2 sgemm_cuda_2.cu)
target_link_libraries(sgemm_cuda_2 PUBLIC cuda_utils)