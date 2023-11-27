#include <iostream>
#include "cuda_utils.h"

int main(int argc, char const *argv[])
{   
    int count = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    std::cout << "cuda device count: " << count << std::endl;
    return 0;
}
