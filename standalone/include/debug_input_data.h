/******************************************************************************
 * Debug utilities to verify input data consistency
 *
 * Check if Q/K/V input tensors are identical between PyTorch and Standalone
 ******************************************************************************/

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

namespace flash {
namespace debug {

// Compute checksum of a tensor (sum of all elements)
template<typename T>
__global__ void compute_checksum_kernel(
    const T* data,
    int64_t size,
    double* checksum
) {
    __shared__ double shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    for (int64_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        local_sum += static_cast<double>(data[i]);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(checksum, shared_sum[0]);
    }
}

// Compute statistics: min, max, mean, variance
template<typename T>
__global__ void compute_stats_kernel(
    const T* data,
    int64_t size,
    double* min_val,
    double* max_val,
    double* sum,
    double* sum_sq
) {
    __shared__ double shared_min[256];
    __shared__ double shared_max[256];
    __shared__ double shared_sum[256];
    __shared__ double shared_sum_sq[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_min = INFINITY;
    double local_max = -INFINITY;
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    for (int64_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        double val = static_cast<double>(data[i]);
        local_min = fmin(local_min, val);
        local_max = fmax(local_max, val);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = fmin(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + s]);
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin_double(min_val, shared_min[0]);
        atomicMax_double(max_val, shared_max[0]);
        atomicAdd(sum, shared_sum[0]);
        atomicAdd(sum_sq, shared_sum_sq[0]);
    }
}

// Helper for atomic min/max with double
__device__ inline void atomicMin_double(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__device__ inline void atomicMax_double(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

// Print tensor statistics
template<typename T>
void print_tensor_stats(const T* d_data, int64_t size, const char* name) {
    // Allocate device memory for results
    double *d_min, *d_max, *d_sum, *d_sum_sq;
    cudaMalloc(&d_min, sizeof(double));
    cudaMalloc(&d_max, sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_sum_sq, sizeof(double));

    double init_min = INFINITY;
    double init_max = -INFINITY;
    double init_sum = 0.0;
    double init_sum_sq = 0.0;

    cudaMemcpy(d_min, &init_min, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &init_max, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &init_sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_sq, &init_sum_sq, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = std::min((int)((size + threads - 1) / threads), 1024);
    compute_stats_kernel<<<blocks, threads>>>(d_data, size, d_min, d_max, d_sum, d_sum_sq);
    cudaDeviceSynchronize();

    // Copy results back
    double h_min, h_max, h_sum, h_sum_sq;
    cudaMemcpy(&h_min, d_min, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost);

    // Compute stats
    double mean = h_sum / size;
    double variance = (h_sum_sq / size) - (mean * mean);
    double stddev = sqrt(variance);

    printf("[TENSOR STATS] %s (size=%ld):\n", name, (long)size);
    printf("  min=%.6f, max=%.6f, mean=%.6f, std=%.6f\n", h_min, h_max, mean, stddev);
    printf("  sum=%.6f, checksum_abs_sum=%.6f\n", h_sum, fabs(h_sum));

    // Cleanup
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
}

// Macro to print all input tensor stats
#define PRINT_INPUT_TENSOR_STATS(params) \
    do { \
        printf("\n========== INPUT TENSOR STATISTICS ==========\n"); \
        if (params.q_ptr) { \
            int64_t q_size = (params.cu_seqlens_q ? params.total_q : params.b * params.seqlen_q) * params.h * params.d; \
            if (params.is_bf16) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_bfloat16*>(params.q_ptr), q_size, "Q"); \
            } else if (params.is_e4m3) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_fp8_e4m3*>(params.q_ptr), q_size, "Q"); \
            } else { \
                flash::debug::print_tensor_stats(static_cast<const __half*>(params.q_ptr), q_size, "Q"); \
            } \
        } \
        if (params.k_ptr) { \
            int64_t k_size = (params.cu_seqlens_k ? params.total_k : params.b * params.seqlen_k) * params.h_k * params.d; \
            if (params.is_bf16) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_bfloat16*>(params.k_ptr), k_size, "K"); \
            } else if (params.is_e4m3) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_fp8_e4m3*>(params.k_ptr), k_size, "K"); \
            } else { \
                flash::debug::print_tensor_stats(static_cast<const __half*>(params.k_ptr), k_size, "K"); \
            } \
        } \
        if (params.v_ptr) { \
            int64_t v_size = (params.cu_seqlens_k ? params.total_k : params.b * params.seqlen_k) * params.h_k * params.dv; \
            if (params.is_bf16) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_bfloat16*>(params.v_ptr), v_size, "V"); \
            } else if (params.is_e4m3) { \
                flash::debug::print_tensor_stats(static_cast<const __nv_fp8_e4m3*>(params.v_ptr), v_size, "V"); \
            } else { \
                flash::debug::print_tensor_stats(static_cast<const __half*>(params.v_ptr), v_size, "V"); \
            } \
        } \
        printf("=============================================\n\n"); \
        fflush(stdout); \
    } while(0)

// Compare two tensors element-wise
template<typename T>
__global__ void compare_tensors_kernel(
    const T* data1,
    const T* data2,
    int64_t size,
    double* max_abs_diff,
    double* max_rel_diff,
    int64_t* num_diff
) {
    __shared__ double shared_abs_diff[256];
    __shared__ double shared_rel_diff[256];
    __shared__ int64_t shared_num_diff[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_abs_diff = 0.0;
    double local_rel_diff = 0.0;
    int64_t local_num_diff = 0;

    for (int64_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        double v1 = static_cast<double>(data1[i]);
        double v2 = static_cast<double>(data2[i]);
        double abs_diff = fabs(v1 - v2);
        double rel_diff = fabs(v1) > 1e-8 ? abs_diff / fabs(v1) : 0.0;

        local_abs_diff = fmax(local_abs_diff, abs_diff);
        local_rel_diff = fmax(local_rel_diff, rel_diff);

        if (abs_diff > 1e-6) {  // Threshold for "different"
            local_num_diff++;
        }
    }

    shared_abs_diff[tid] = local_abs_diff;
    shared_rel_diff[tid] = local_rel_diff;
    shared_num_diff[tid] = local_num_diff;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_abs_diff[tid] = fmax(shared_abs_diff[tid], shared_abs_diff[tid + s]);
            shared_rel_diff[tid] = fmax(shared_rel_diff[tid], shared_rel_diff[tid + s]);
            shared_num_diff[tid] += shared_num_diff[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax_double(max_abs_diff, shared_abs_diff[0]);
        atomicMax_double(max_rel_diff, shared_rel_diff[0]);
        atomicAdd((unsigned long long*)num_diff, (unsigned long long)shared_num_diff[0]);
    }
}

} // namespace debug
} // namespace flash

// Usage in hopper/flash_fwd_launch_template.h:
//
// Add at the beginning of run_flash_fwd():
//
//   #ifdef FLASH_DEBUG_INPUT_DATA
//   PRINT_INPUT_TENSOR_STATS(params);
//   #endif
//
// Then compile with: -DFLASH_DEBUG_INPUT_DATA
//
// Example output:
// [TENSOR STATS] Q (size=2099200):
//   min=-3.142578, max=3.140625, mean=0.000123, std=1.001234
//   sum=258.456, checksum_abs_sum=258.456
