add_library(cuda_utils OBJECT cuda_utils.cpp)
target_link_directories(cuda_utils PUBLIC ${CMAKE_CURRENT_LIST_DIR})