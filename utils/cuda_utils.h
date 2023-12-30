#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <iostream>
#include <string>

/*
 * Handle cuda error, exit process when found cuda error.
 */
#define HANDLE_ERROR(exp) \
do { \
    cudaError error = (exp); \
    if (error != cudaSuccess) { \
        std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ << \
            ": " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)


#define CEIL_DIV(x, div) (((x) + (div) - 1) / (div))

bool allclose(float *a_ptr, float *b_ptr, size_t n_elem);
void print_array(float *a_ptr, size_t n_elem);
float *read_bin(const std::string &file, size_t size);


#endif // CUDA_UTILS_H