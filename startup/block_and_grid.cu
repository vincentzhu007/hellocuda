#include <stdio.h>

__global__ void CheckIndex(void) {
  // cuda内置的变量：threadIdx、blockIdx、blockDim、gridDim。
  // 可用于在kernel内获取上下文信息。
  printf("Device: threadIdx:(%d, %d, %d), blockIdx:(%d, %d, %d), "
         "blockDim:(%d, %d, %d), gridDim:(%d, %d, %d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
         blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  printf("CUDA Demo: illustrate block and thread index.\n\n");

  int n_elem = 6;
  // 计算block和grid数量：1个grid包含多个block，1个block包含多个thread.
  dim3 block(3); // block内的thread分布
  dim3 grid((n_elem + block.x - 1) / block.x); // grid中的block分布

  printf("Host: block:(%d, %d, %d), grid:(%d, %d, %d)\n\n",
         block.x, block.y, block.z, grid.x, grid.y, grid.z);

  // 调用kernel函数
  CheckIndex<<<grid, block>>>();

  // 执行同步操作，等待kernel完全执行结束。注意，没有这一行，终端中不会打印kernel printf内容！
  cudaDeviceSynchronize();
}