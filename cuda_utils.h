#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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

#endif // CUDA_UTILS_H