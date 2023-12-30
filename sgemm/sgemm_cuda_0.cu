#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>
#include "cuda_utils.h"
#include "data.h"

bool allclose(float *a_ptr, float *b_ptr, size_t n_elem) {
    constexpr float kErrorUpLimit = 1e-4;
    for (int i = 0; i < n_elem; i++) {
        float abs_error = a_ptr[i] - b_ptr[i];
        if (fabs(abs_error) > kErrorUpLimit) {
            printf("allclose: a[%d]=%f, b[%d]=%f is not close.\n", i, a_ptr[i], i, b_ptr[i]);
            return false;
        }
    }
    return true;
}

void print_array(float *a_ptr, size_t n_elem) {
    constexpr size_t kPrintMax = 10;
    if (n_elem <= kPrintMax) {
        printf("[");
        for (size_t i = 0; i < n_elem; i++) {
            printf("%16.8f", a_ptr[i]);
            if (i + 1 < n_elem) { // not tail one
                printf(", ");
            }
        }
        printf("]\n");
        return;
    }

    // 超过10个，则打印首位数字，前后各5个
    printf("[");
    size_t n_print = kPrintMax / 2;
    for (size_t i = 0; i < n_print; i++) {
        printf("%16.8f, ", a_ptr[i]);
    }
    printf(" ... , ");
    for (int i = n_print; i > 0; i--) {
        printf("%16.8f", a_ptr[n_elem - i]);
        if (i > 1) { // not tail one
            printf(", ");
        }
    }
    printf("]\n");
}

float *read_bin(const std::string &file, size_t size) {
    float *buffer = (float *)malloc(size);
    std::ifstream in(file, std::ios::in | std::ios::binary);
    in.read((char *)buffer, size);
    printf("read %d bytes from %s.\n", in.gcount(), file.c_str());
    return buffer;
}

__global__ void sgemm_naive(int m, int k, int n, const float *x, const float *y, float *z) {
    int mi = blockIdx.x;
    int ni = threadIdx.x;

    double out = 0.0f;
    for (int s = 0; s < k; s++) {
        out += x[mi * k + s] * y[s * n + ni];
    }
    z[mi * n + ni] = out; 
}

int main() {
    constexpr int kDim = 512;
    int m = kDim;
    int k = kDim;
    int n = kDim;

    size_t size_x = sizeof(float) * m * k;
    size_t size_y = sizeof(float) * k * n;
    size_t size_z = sizeof(float) * m * n;

    // 设置host侧数据
    float * host_ptr_x = read_bin(kDataDir + "/sgemm_m_512_k_512_n_512_x.bin", size_x);
    float * host_ptr_y = read_bin(kDataDir + "/sgemm_m_512_k_512_n_512_y.bin", size_y);
    float * expected_z = read_bin(kDataDir + "/sgemm_m_512_k_512_n_512_z.bin", size_z);
    float * host_ptr_z = (float *)malloc(size_z);

    // 设置device侧数据
    float *device_ptr_x;
    float *device_ptr_y;
    float *device_ptr_z;

    HANDLE_ERROR(cudaMalloc(&device_ptr_x, size_x));
    HANDLE_ERROR(cudaMalloc(&device_ptr_y, size_y));
    HANDLE_ERROR(cudaMalloc(&device_ptr_z, size_z));

    HANDLE_ERROR(cudaMemcpy(device_ptr_x, host_ptr_x, size_x, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_ptr_y, host_ptr_y, size_y, cudaMemcpyHostToDevice));
    
    // 计算matmul
    dim3 block(kDim);
    dim3 grid(kDim);
    sgemm_naive<<<grid, block>>>(m, k, n, device_ptr_x, device_ptr_y, device_ptr_z);
    HANDLE_ERROR(cudaMemcpy(host_ptr_z, device_ptr_z, size_z, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(device_ptr_x));
    HANDLE_ERROR(cudaFree(device_ptr_y));
    HANDLE_ERROR(cudaFree(device_ptr_z));

    // 比较计算结果
    printf("calculated z:\n");
    print_array(host_ptr_z, m * n);
    printf("expected z:\n");
    print_array(expected_z, m * n);

    bool is_equal = allclose(host_ptr_z, expected_z, m * n);
    printf("accuracy checking result: %s.\n", (is_equal ? "PASS" : "NOT PASS!!!"));


    free(host_ptr_x);
    free(host_ptr_y);
    free(host_ptr_z);
    free(expected_z);
}