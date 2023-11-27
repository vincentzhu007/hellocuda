#include <iostream>
#include <iomanip>
#include "cuda_utils.h"

void printCudaProp(const struct cudaDeviceProp &prop) {
    std::cout << std::left;
    std::cout << std::setw(40) << "name: " << prop.name << std::endl;
    std::cout << std::setw(40) << "capibility: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::setw(40) << "total global memory: " << prop.totalGlobalMem << std::endl;
    std::cout << std::setw(40) << "total const memory: " << prop.totalConstMem << std::endl;
    std::cout << std::setw(40) << "shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << std::setw(40) << "regs per block: " << prop.regsPerBlock << std::endl;
    std::cout << std::setw(40) << "wrap size: " << prop.warpSize << std::endl;
    std::cout << std::setw(40) << "mem pitch: " << prop.memPitch << std::endl;
    std::cout << std::setw(40) << "clock rate: " << prop.clockRate << std::endl;
    
    std::cout << std::setw(40) << "max thread per block " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::setw(40) << "max blocks per multi-processor: "
        << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << std::setw(40) << "max grid size: [" << prop.maxGridSize[0]
        << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << std::setw(40) << "max thread dim: [" << prop.maxThreadsDim[0]
        << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;

    std::cout << std::setw(40) << "device overlap: " << prop.deviceOverlap << std::endl;
    std::cout << std::setw(40) << "can map host mem: " << prop.canMapHostMemory << std::endl;
    std::cout << std::setw(40) << "can map host mem: " << prop.canMapHostMemory << std::endl;
}

int main(int argc, char const *argv[])
{   
    std::cout << "------>>> CUDA info <<<------\n" << std::endl;

    int count = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    std::cout << "cuda device count: " << count << std::endl;

    struct cudaDeviceProp prop;
    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "=== cuda device: " << i << " ===" << std::endl;
        printCudaProp(prop);
        std::cout << std::endl;
    }
    return 0;
}
