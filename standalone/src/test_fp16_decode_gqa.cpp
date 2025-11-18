#include "../include/flash_api.h"
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

/**
 * Test case matching the PyTorch configuration:
 *
 * batch_size = 1
 * nheads = 16
 * nheads_k = 2  (GQA with 16:2 ratio)
 * d = 128
 * seqlen_q = 1 (DECODE mode)
 * seqlen_k = 2048
 * causal = True
 * dtype = torch.float16
 *
 * Q, K, V shape: [batch, seqlen, nheads, d]
 * Non-varlen (cu_seqlens = None)
 */

void print_test_info() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  FP16 Decode GQA Attention Test\n";
    std::cout << "  (Matching PyTorch flash_attn_func configuration)\n";
    std::cout << "================================================================\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    print_test_info();

    // ========================================================================
    // CONFIGURATION (matching PyTorch testcase)
    // ========================================================================

    const int batch_size = 1;
    const int seqlen_q = 1;      // DECODE mode: single query token
    const int seqlen_k = 2048;   // Context length
    const int nheads = 16;
    const int nheads_k = 2;  // GQA: 16:2 ratio
    const int head_dim = 128;
    const bool causal = true;

    std::cout << "Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Seqlen Q: " << seqlen_q << " (DECODE mode)\n";
    std::cout << "  Seqlen K: " << seqlen_k << "\n";
    std::cout << "  Num heads Q: " << nheads << "\n";
    std::cout << "  Num heads K: " << nheads_k << " (GQA " << nheads << ":" << nheads_k << ")\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  Causal: " << (causal ? "true" : "false") << "\n";
    std::cout << "  Data type: FP16\n";
    std::cout << "\n";

    // ========================================================================
    // ALLOCATE AND GENERATE DATA
    // ========================================================================

    std::cout << "Allocating memory...\n";

    // Q, K, V: [batch, seqlen, nheads, head_dim] in FP16
    size_t q_elements = batch_size * seqlen_q * nheads * head_dim;
    size_t k_elements = batch_size * seqlen_k * nheads_k * head_dim;
    size_t v_elements = batch_size * seqlen_k * nheads_k * head_dim;

    std::cout << "  Q: " << (q_elements * sizeof(__half) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  K: " << (k_elements * sizeof(__half) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  V: " << (v_elements * sizeof(__half) / 1024.0 / 1024.0) << " MB\n";

    // Generate host data
    std::vector<__half> h_q(q_elements);
    std::vector<__half> h_k(k_elements);
    std::vector<__half> h_v(v_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::cout << "Generating random data...\n";

    // Q: all ones (like PyTorch testcase)
    for (size_t i = 0; i < q_elements; i++) {
        h_q[i] = __float2half(1.0f);
    }

    // K: all ones (like PyTorch testcase)
    for (size_t i = 0; i < k_elements; i++) {
        h_k[i] = __float2half(1.0f);
    }

    // V: random (like PyTorch testcase)
    for (size_t i = 0; i < v_elements; i++) {
        h_v[i] = __float2half(dis(gen));
    }

    // ========================================================================
    // ALLOCATE DEVICE MEMORY
    // ========================================================================

    void *d_q, *d_k, *d_v, *d_out;
    float *d_lse;

    cudaMalloc(&d_q, q_elements * sizeof(__half));
    cudaMalloc(&d_k, k_elements * sizeof(__half));
    cudaMalloc(&d_v, v_elements * sizeof(__half));
    cudaMalloc(&d_out, q_elements * sizeof(__half));  // Output is FP16
    cudaMalloc(&d_lse, batch_size * seqlen_q * nheads * sizeof(float));

    std::cout << "Copying data to device...\n";

    // Copy Q, K, V
    cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_elements * sizeof(__half), cudaMemcpyHostToDevice);

    std::cout << "Done.\n\n";

    // ========================================================================
    // SETUP FLASH ATTENTION PARAMETERS
    // ========================================================================

    std::cout << "Setting up Flash Attention parameters...\n";

    flash::FlashAttentionParams params;

    // Input/output tensors
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_lse;

    // Dimensions
    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.num_heads = nheads;
    params.num_heads_k = nheads_k;
    params.head_dim = head_dim;

    // Non-varlen (cu_seqlens = nullptr)
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.total_q = 0;
    params.total_k = 0;

    // No FP8 descaling for FP16
    params.q_descale_ptr = nullptr;
    params.k_descale_ptr = nullptr;
    params.v_descale_ptr = nullptr;

    // Attention parameters
    params.is_causal = causal;

    // Softmax scale: 1/sqrt(head_dim)
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    params.softmax_scale = softmax_scale;
    std::cout << "  Softmax scale: " << softmax_scale << "\n";

    // Data type
    params.dtype = flash::DataType::FP16;
    params.mode = flash::AttentionMode::DECODE;

    std::cout << "Done.\n\n";

    // ========================================================================
    // VALIDATE PARAMETERS
    // ========================================================================

    std::cout << "Validating parameters...\n";
    int validation_result = flash::validate_params(params);
    if (validation_result != 0) {
        std::cerr << "✗ Parameter validation failed: "
                  << flash::get_error_string(validation_result) << "\n";
        return validation_result;
    }
    std::cout << "✓ Parameters validated successfully\n\n";

    // ========================================================================
    // RUN FLASH ATTENTION
    // ========================================================================

    std::cout << "Running Flash Attention forward pass...\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warm up (optional)
    for (int i = 0; i < 2; i++) {
        flash::flash_attention_forward(params, stream);
    }
    cudaStreamSynchronize(stream);

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    int result = flash::flash_attention_forward(params, stream);
    cudaEventRecord(stop, stream);

    cudaStreamSynchronize(stream);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();

    std::cout << "\n";
    std::cout << "================================================================\n";
    if (result == 0 && err == cudaSuccess) {
        std::cout << "  ✓ SUCCESS\n";
        std::cout << "================================================================\n";
        std::cout << "\n";
        std::cout << "Performance:\n";
        std::cout << "  Time: " << elapsed_ms << " ms\n";

        // Calculate FLOPS (approximate)
        // Attention FLOPS ≈ 4 * batch * seqlen_q * seqlen_k * nheads * head_dim
        double flops = 4.0 * batch_size * seqlen_q * seqlen_k * nheads * head_dim;
        double tflops = (flops / (elapsed_ms / 1000.0)) / 1e12;
        std::cout << "  Throughput: " << tflops << " TFLOPS\n";

    } else {
        std::cout << "  ✗ FAILED\n";
        std::cout << "================================================================\n";
        std::cout << "\n";
        std::cout << "Error details:\n";
        std::cout << "  API result: " << flash::get_error_string(result) << "\n";
        if (err != cudaSuccess) {
            std::cout << "  CUDA error: " << cudaGetErrorString(err) << "\n";
        }
    }
    std::cout << "\n";

    // ========================================================================
    // OPTIONAL: VERIFY OUTPUT (basic sanity check)
    // ========================================================================

    if (result == 0 && err == cudaSuccess) {
        std::cout << "Verifying output (basic sanity check)...\n";

        // Copy a small portion of output back to host
        const int check_tokens = std::min(10, seqlen_q);
        std::vector<__half> h_out(check_tokens * nheads * head_dim);
        cudaMemcpy(h_out.data(), d_out,
                   check_tokens * nheads * head_dim * sizeof(__half),
                   cudaMemcpyDeviceToHost);

        // Check for NaN/Inf
        bool has_nan_inf = false;
        for (size_t i = 0; i < h_out.size(); i++) {
            float val = __half2float(h_out[i]);
            if (std::isnan(val) || std::isinf(val)) {
                has_nan_inf = true;
                break;
            }
        }

        if (has_nan_inf) {
            std::cout << "  ⚠️  WARNING: Output contains NaN or Inf values\n";
        } else {
            std::cout << "  ✓ Output looks reasonable (no NaN/Inf in first "
                      << check_tokens << " tokens)\n";
        }

        // Print first few values
        std::cout << "  First 5 output values (head 0, token 0): ";
        for (int i = 0; i < std::min(5, head_dim); i++) {
            std::cout << __half2float(h_out[i]) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    // ========================================================================
    // CLEANUP
    // ========================================================================

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return (result == 0 && err == cudaSuccess) ? 0 : -1;
}
