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
 * max_seqlen_q = max_seqlen_k = 1680
 * batch_size = 84
 * nheads = 16
 * nheads_k = 16
 * d = 128
 * seqlen_q = 5040 (total tokens across all sequences)
 * seqlen_k = 5040
 * causal = False
 * dtype = torch.float8_e4m3fn
 *
 * Q, K, V shape: [seqlen, nheads, d] in varlen format
 * cu_seqlens shape: [batch + 1]
 * descale shape: [batch, nheads]
 */

void print_test_info() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  FP8 E4M3 Variable-length Attention Test\n";
    std::cout << "  (Matching PyTorch flash_attn_varlen_func configuration)\n";
    std::cout << "================================================================\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    print_test_info();

    // ========================================================================
    // CONFIGURATION (matching PyTorch testcase)
    // ========================================================================

    const int max_seqlen_q = 1680;
    const int max_seqlen_k = 1680;
    const int batch_size = 3;
    const int nheads = 16;
    const int nheads_k = 16;  // MHA
    const int head_dim = 128;
    const int total_q = 5040;  // Sum of all sequence lengths
    const int total_k = 5040;
    const bool causal = false;

    std::cout << "Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Max seqlen Q: " << max_seqlen_q << "\n";
    std::cout << "  Max seqlen K: " << max_seqlen_k << "\n";
    std::cout << "  Total tokens Q: " << total_q << "\n";
    std::cout << "  Total tokens K: " << total_k << "\n";
    std::cout << "  Num heads Q: " << nheads << "\n";
    std::cout << "  Num heads K: " << nheads_k << "\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  Causal: " << (causal ? "true" : "false") << "\n";
    std::cout << "  Data type: FP8 E4M3\n";
    std::cout << "\n";

    // ========================================================================
    // CUMULATIVE SEQUENCE LENGTHS (cu_seqlens)
    // ========================================================================

    // max_seqlen = 1680, 3 sequences: [0, 1680, 3360, 5040]
    std::vector<int> h_cu_seqlens_q = {0, 1680, 3360, 5040};

    // K uses the same cu_seqlens as Q in this test
    std::vector<int> h_cu_seqlens_k = h_cu_seqlens_q;

    // Verify batch size matches
    if (h_cu_seqlens_q.size() != batch_size + 1) {
        std::cerr << "Error: cu_seqlens_q size (" << h_cu_seqlens_q.size()
                  << ") != batch_size + 1 (" << batch_size + 1 << ")\n";
        return -1;
    }

    // Verify total tokens match
    if (h_cu_seqlens_q.back() != total_q) {
        std::cerr << "Error: cu_seqlens_q last element (" << h_cu_seqlens_q.back()
                  << ") != total_q (" << total_q << ")\n";
        return -1;
    }

    std::cout << "Sequence lengths per batch:\n";
    std::cout << "  First 5 sequences: ";
    for (int i = 0; i < std::min(5, batch_size); i++) {
        int seqlen = h_cu_seqlens_q[i + 1] - h_cu_seqlens_q[i];
        std::cout << seqlen;
        if (i < 4 && i < batch_size - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "  Last 5 sequences: ";
    for (int i = std::max(0, batch_size - 5); i < batch_size; i++) {
        int seqlen = h_cu_seqlens_q[i + 1] - h_cu_seqlens_q[i];
        std::cout << seqlen;
        if (i < batch_size - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    // ========================================================================
    // ALLOCATE AND GENERATE DATA
    // ========================================================================

    std::cout << "Allocating memory...\n";

    // Q, K, V: [total_tokens, nheads, head_dim] in FP8 E4M3
    size_t q_elements = total_q * nheads * head_dim;
    size_t k_elements = total_k * nheads_k * head_dim;
    size_t v_elements = total_k * nheads_k * head_dim;

    std::cout << "  Q: " << (q_elements * sizeof(__nv_fp8_e4m3) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  K: " << (k_elements * sizeof(__nv_fp8_e4m3) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  V: " << (v_elements * sizeof(__nv_fp8_e4m3) / 1024.0 / 1024.0) << " MB\n";

    // Generate host data
    std::vector<__nv_fp8_e4m3> h_q(q_elements);
    std::vector<__nv_fp8_e4m3> h_k(k_elements);
    std::vector<__nv_fp8_e4m3> h_v(v_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::cout << "Generating random data...\n";

    // Q: all ones (like PyTorch testcase)
    for (size_t i = 0; i < q_elements; i++) {
        h_q[i] = __nv_fp8_e4m3(1.0f);
    }

    // K: all ones (like PyTorch testcase)
    for (size_t i = 0; i < k_elements; i++) {
        h_k[i] = __nv_fp8_e4m3(1.0f);
    }

    // V: random (like PyTorch testcase)
    for (size_t i = 0; i < v_elements; i++) {
        h_v[i] = __nv_fp8_e4m3(dis(gen));
    }

    // ========================================================================
    // ALLOCATE DEVICE MEMORY
    // ========================================================================

    void *d_q, *d_k, *d_v, *d_out;
    float *d_lse;
    int *d_cu_seqlens_q, *d_cu_seqlens_k;
    float *d_descale_q, *d_descale_k, *d_descale_v;

    cudaMalloc(&d_q, q_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_k, k_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_v, v_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_out, q_elements * sizeof(__half));  // Output is FP16
    cudaMalloc(&d_lse, total_q * nheads * sizeof(float));
    cudaMalloc(&d_cu_seqlens_q, (batch_size + 1) * sizeof(int));
    cudaMalloc(&d_cu_seqlens_k, (batch_size + 1) * sizeof(int));

    // Descale: [batch, nheads] - per-batch, per-head scaling
    cudaMalloc(&d_descale_q, batch_size * nheads * sizeof(float));
    cudaMalloc(&d_descale_k, batch_size * nheads_k * sizeof(float));
    cudaMalloc(&d_descale_v, batch_size * nheads_k * sizeof(float));

    std::cout << "Copying data to device...\n";

    // Copy Q, K, V
    cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // Copy cu_seqlens
    cudaMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(),
               (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cu_seqlens_k, h_cu_seqlens_k.data(),
               (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize descale factors to 1.0 (like PyTorch testcase: torch.ones)
    std::vector<float> h_descale(batch_size * nheads, 1.0f);
    cudaMemcpy(d_descale_q, h_descale.data(),
               batch_size * nheads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_descale_k, h_descale.data(),
               batch_size * nheads_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_descale_v, h_descale.data(),
               batch_size * nheads_k * sizeof(float), cudaMemcpyHostToDevice);

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
    params.seqlen_q = max_seqlen_q;  // Will be ignored due to varlen
    params.seqlen_k = max_seqlen_k;  // Will be ignored due to varlen
    params.num_heads = nheads;
    params.num_heads_k = nheads_k;
    params.head_dim = head_dim;

    // ✅ Variable-length sequences
    params.cu_seqlens_q = d_cu_seqlens_q;
    params.cu_seqlens_k = d_cu_seqlens_k;
    params.total_q = total_q;
    params.total_k = total_k;

    // ✅ FP8 descaling (per-batch, per-head)
    params.q_descale_ptr = d_descale_q;
    params.k_descale_ptr = d_descale_k;
    params.v_descale_ptr = d_descale_v;

    // Descale strides: [batch, nheads]
    // Layout: descale[batch_idx * nheads + head_idx]
    params.q_descale_batch_stride = nheads;
    params.q_descale_head_stride = 1;
    params.k_descale_batch_stride = nheads_k;
    params.k_descale_head_stride = 1;
    params.v_descale_batch_stride = nheads_k;
    params.v_descale_head_stride = 1;

    // Attention parameters
    params.is_causal = causal;

    // Softmax scale: 1/sqrt(head_dim)
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    params.softmax_scale = softmax_scale;
    std::cout << "  Softmax scale: " << softmax_scale << "\n";

    // Data type
    params.dtype = flash::DataType::FP8_E4M3;
    params.mode = flash::AttentionMode::VARLEN_PREFILL;

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
        double flops = 4.0 * total_q * total_k * nheads * head_dim;
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
        const int check_tokens = std::min(10, total_q);
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
    cudaFree(d_cu_seqlens_q);
    cudaFree(d_cu_seqlens_k);
    cudaFree(d_descale_q);
    cudaFree(d_descale_k);
    cudaFree(d_descale_v);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return (result == 0 && err == cudaSuccess) ? 0 : -1;
}
