#include <iostream>
#include <cstdio>
#include <string>
#include "cuda_utils.h"
#include "data.h"

template<int BLOCK>
__global__ void sgemm(int m, int k, int n, float *x, float *y, float *z) {
  // blockIdx control subpanel matrix
  const int tx = threadIdx.x; // 小块矩阵，0~16
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_x = x + bx * BLOCK * k; // x[mi][0]
  float *begin_y = y + by * BLOCK;
  float *end_x = begin_x + k; // x[mi+1][0]

  float sum = 0.f;
  for (float *x_ptr = begin_x, *y_ptr = begin_y; x_ptr < end_x;
       x_ptr += BLOCK, y_ptr += BLOCK * n) {

    __shared__ float share_x[BLOCK][BLOCK];
    __shared__ float share_y[BLOCK][BLOCK];

    share_x[ty][tx] = x_ptr[ty * k + tx];
    share_y[ty][tx] = y_ptr[ty * n + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += share_x[ty][kk] * share_y[kk][tx];
    }
    __syncthreads();
  }

  // mi = BLOCK * bx + ty
  // ni = BLOCK * by + tx // 弄反了？
  z[(BLOCK * bx + ty) * n + BLOCK * by + tx] = sum;
}

int main() {
    constexpr int kDim = 1024;
    int m = kDim;
    int k = kDim;
    int n = kDim;

    size_t size_x = sizeof(float) * m * k;
    size_t size_y = sizeof(float) * k * n;
    size_t size_z = sizeof(float) * m * n;

    // 设置host侧数据
    std::string bin_prefix = kDataDir + "/sgemm_m_" + std::to_string(m) + "_k_" + std::to_string(k) + "_n_" + std::to_string(n);
    float * host_ptr_x = read_bin(bin_prefix +"_x.bin", size_x);
    float * host_ptr_y = read_bin(bin_prefix +"_y.bin", size_y);
    float * expected_z = read_bin(bin_prefix +"_z.bin", size_z);
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
    constexpr int kBLOCK = 16;
    dim3 block(kBLOCK, kBLOCK);
    dim3 grid(CEIL_DIV(m, kBLOCK), CEIL_DIV(n, kBLOCK));
    sgemm<kBLOCK><<<grid, block>>>(m, k, n, device_ptr_x, device_ptr_y, device_ptr_z);
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
    printf("\nAccuracy checking result: %s.\n", (is_equal ? "PASS" : "NOT PASS!!!"));

    free(host_ptr_x);
    free(host_ptr_y);
    free(host_ptr_z);
    free(expected_z);
}