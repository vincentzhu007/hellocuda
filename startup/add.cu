#include <iostream>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main()
{
    int a = 100;
    int b = 200;
    int c = -1;

    int *device_ptr_c;

    cudaError error = cudaMalloc(&device_ptr_c, sizeof(int));
    if (error != cudaSuccess) {
        std::cerr << "cuda malloc failed, " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

    add<<<1,1>>>(a, b, device_ptr_c);

    error = cudaMemcpy(&c, device_ptr_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "cuda memcpy failed, " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    
    std::cout << a << " + " << b << " = " << c << std::endl;

    cudaFree(device_ptr_c);

    return 0;
}
