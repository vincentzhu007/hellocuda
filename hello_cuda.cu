#include <iostream>

__global__ void kernel() {}

int main(int argc, char const *argv[])
{
    kernel<<<1,1>>>();
    std::cout << "hello cuda" << std::endl;
    return 0;
}
