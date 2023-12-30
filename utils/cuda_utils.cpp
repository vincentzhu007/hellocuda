
#include "cuda_utils.h"
#include <cstdio>
#include <cmath>
#include <fstream>

bool allclose(float *a_ptr, float *b_ptr, size_t n_elem) {
    constexpr float kErrorUpLimit = 1e-3;
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
    printf("read %d bytes from %s.\n", (int)in.gcount(), file.c_str());
    return buffer;
}