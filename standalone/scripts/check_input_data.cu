// Standalone tool to check if input tensors are identical
// Compile: nvcc -o check_input_data check_input_data.cu
// Usage: ./check_input_data q1.bin q2.bin

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void compare_fp16_kernel(
    const __half* data1,
    const __half* data2,
    int size,
    double* max_abs_diff,
    double* sum_abs_diff,
    int* num_diff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float v1 = __half2float(data1[idx]);
    float v2 = __half2float(data2[idx]);
    float abs_diff = fabsf(v1 - v2);

    atomicMax_double(max_abs_diff, abs_diff);
    atomicAdd(sum_abs_diff, (double)abs_diff);

    if (abs_diff > 1e-5) {
        atomicAdd(num_diff, 1);
    }
}

__device__ void atomicMax_double(double* address, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                       __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

void load_tensor(const char* filename, __half** d_data, int size) {
    // Allocate host memory
    __half* h_data = (__half*)malloc(size * sizeof(__half));

    // Read from file
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(1);
    }
    fread(h_data, sizeof(__half), size, f);
    fclose(f);

    // Copy to device
    cudaMalloc(d_data, size * sizeof(__half));
    cudaMemcpy(*d_data, h_data, size * sizeof(__half), cudaMemcpyHostToDevice);

    free(h_data);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <file1.bin> <file2.bin> <size>\n", argv[0]);
        return 1;
    }

    const char* file1 = argv[1];
    const char* file2 = argv[2];
    int size = atoi(argv[3]);

    printf("Comparing tensors:\n");
    printf("  File 1: %s\n", file1);
    printf("  File 2: %s\n", file2);
    printf("  Size: %d elements\n", size);

    // Load tensors
    __half *d_data1, *d_data2;
    load_tensor(file1, &d_data1, size);
    load_tensor(file2, &d_data2, size);

    // Allocate result buffers
    double *d_max_diff, *d_sum_diff;
    int *d_num_diff;
    cudaMalloc(&d_max_diff, sizeof(double));
    cudaMalloc(&d_sum_diff, sizeof(double));
    cudaMalloc(&d_num_diff, sizeof(int));

    double init_max = 0.0, init_sum = 0.0;
    int init_num = 0;
    cudaMemcpy(d_max_diff, &init_max, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_diff, &init_sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_diff, &init_num, sizeof(int), cudaMemcpyHostToDevice);

    // Launch comparison kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compare_fp16_kernel<<<blocks, threads>>>(
        d_data1, d_data2, size, d_max_diff, d_sum_diff, d_num_diff);
    cudaDeviceSynchronize();

    // Get results
    double h_max_diff, h_sum_diff;
    int h_num_diff;
    cudaMemcpy(&h_max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_diff, d_sum_diff, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_num_diff, d_num_diff, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("\nResults:\n");
    printf("  Max absolute difference: %.6e\n", h_max_diff);
    printf("  Mean absolute difference: %.6e\n", h_sum_diff / size);
    printf("  Number of different elements (>1e-5): %d (%.2f%%)\n",
           h_num_diff, 100.0 * h_num_diff / size);

    if (h_max_diff < 1e-6) {
        printf("\n✓ Tensors are IDENTICAL (within FP16 precision)\n");
    } else if (h_max_diff < 1e-3) {
        printf("\n⚠ Tensors are SIMILAR but have small differences\n");
    } else {
        printf("\n✗ Tensors are DIFFERENT!\n");
    }

    // Cleanup
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_max_diff);
    cudaFree(d_sum_diff);
    cudaFree(d_num_diff);

    return 0;
}
