#include "../include/flash_api_v2.h"
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

void generate_random_fp16(std::vector<__half>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __float2half(dis(gen));
    }
}

void generate_random_fp8_e4m3(std::vector<__nv_fp8_e4m3>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);  // FP8 has limited range
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __nv_fp8_e4m3(dis(gen));
    }
}

void print_test_header(const char* test_name) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  " << test_name << "\n";
    std::cout << "============================================================\n";
}

//==============================================================================
// TEST CASE 1: FP16 with Causal Mask (Prefill)
//==============================================================================

void test_fp16_causal_prefill() {
    print_test_header("TEST 1: FP16 Prefill with Causal Mask");

    const int batch_size = 2;
    const int seqlen = 512;
    const int num_heads = 16;
    const int head_dim = 128;

    std::cout << "Configuration:\n";
    std::cout << "  Mode: Prefill (autoregressive)\n";
    std::cout << "  Dtype: FP16\n";
    std::cout << "  Batch: " << batch_size << "\n";
    std::cout << "  Seqlen: " << seqlen << "\n";
    std::cout << "  Heads: " << num_heads << " (MHA)\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  Causal: YES\n\n";

    // Allocate tensors
    size_t qkv_elements = batch_size * seqlen * num_heads * head_dim;
    size_t qkv_bytes = qkv_elements * sizeof(__half);

    std::vector<__half> h_q(qkv_elements), h_k(qkv_elements), h_v(qkv_elements);
    generate_random_fp16(h_q);
    generate_random_fp16(h_k);
    generate_random_fp16(h_v);

    void *d_q, *d_k, *d_v, *d_out;
    float *d_lse;
    cudaMalloc(&d_q, qkv_bytes);
    cudaMalloc(&d_k, qkv_bytes);
    cudaMalloc(&d_v, qkv_bytes);
    cudaMalloc(&d_out, qkv_bytes);
    cudaMalloc(&d_lse, batch_size * num_heads * seqlen * sizeof(float));

    cudaMemcpy(d_q, h_q.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), qkv_bytes, cudaMemcpyHostToDevice);

    // Setup Flash Attention params
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_lse;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.num_heads = num_heads;
    params.num_heads_k = num_heads;  // MHA
    params.head_dim = head_dim;

    params.is_causal = true;  // ✅ CRITICAL: Enable causal masking
    params.dtype = flash::DataType::FP16;
    params.mode = flash::AttentionMode::PREFILL;

    // Run
    std::cout << "Running Flash Attention...\n";
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int result = flash::flash_attention_forward(params, stream);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();

    if (result == 0 && err == cudaSuccess) {
        std::cout << "✓ SUCCESS\n";
    } else {
        std::cout << "✗ FAILED: " << flash::get_error_string(result) << "\n";
        if (err != cudaSuccess) {
            std::cout << "  CUDA error: " << cudaGetErrorString(err) << "\n";
        }
    }

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaStreamDestroy(stream);
}

//==============================================================================
// TEST CASE 2: FP16 Decoding (seqlen_q = 1)
//==============================================================================

void test_fp16_decode() {
    print_test_header("TEST 2: FP16 Decoding (Inference, seqlen_q=1)");

    const int batch_size = 4;
    const int seqlen_q = 1;        // Single new token
    const int seqlen_k = 2048;     // Full context (KV cache length)
    const int num_heads_q = 16;
    const int num_heads_kv = 2;    // GQA with 16:2 ratio (like Qwen2.5-VL)
    const int head_dim = 128;

    std::cout << "Configuration:\n";
    std::cout << "  Mode: Decode (inference)\n";
    std::cout << "  Dtype: FP16\n";
    std::cout << "  Batch: " << batch_size << "\n";
    std::cout << "  Seqlen Q: " << seqlen_q << " (new token)\n";
    std::cout << "  Seqlen K: " << seqlen_k << " (context)\n";
    std::cout << "  Heads: Q=" << num_heads_q << ", KV=" << num_heads_kv << " (GQA)\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  Causal: NO (already decoded)\n\n";

    // Allocate tensors
    size_t q_elements = batch_size * seqlen_q * num_heads_q * head_dim;
    size_t kv_elements = batch_size * seqlen_k * num_heads_kv * head_dim;

    std::vector<__half> h_q(q_elements), h_k(kv_elements), h_v(kv_elements);
    generate_random_fp16(h_q);
    generate_random_fp16(h_k);
    generate_random_fp16(h_v);

    void *d_q, *d_k, *d_v, *d_out;
    float *d_lse;
    cudaMalloc(&d_q, q_elements * sizeof(__half));
    cudaMalloc(&d_k, kv_elements * sizeof(__half));
    cudaMalloc(&d_v, kv_elements * sizeof(__half));
    cudaMalloc(&d_out, q_elements * sizeof(__half));
    cudaMalloc(&d_lse, batch_size * num_heads_q * seqlen_q * sizeof(float));

    cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_elements * sizeof(__half), cudaMemcpyHostToDevice);

    // Setup Flash Attention params
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_lse;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.num_heads = num_heads_q;
    params.num_heads_k = num_heads_kv;  // GQA
    params.head_dim = head_dim;

    params.is_causal = false;  // Already decoded, just attending to full context
    params.dtype = flash::DataType::FP16;
    params.mode = flash::AttentionMode::DECODE;

    // Run
    std::cout << "Running Flash Attention...\n";
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int result = flash::flash_attention_forward(params, stream);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();

    if (result == 0 && err == cudaSuccess) {
        std::cout << "✓ SUCCESS\n";
    } else {
        std::cout << "✗ FAILED: " << flash::get_error_string(result) << "\n";
        if (err != cudaSuccess) {
            std::cout << "  CUDA error: " << cudaGetErrorString(err) << "\n";
        }
    }

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaStreamDestroy(stream);
}

//==============================================================================
// TEST CASE 3: FP8 E4M3 with Variable-length + Descaling
//==============================================================================

void test_fp8_varlen_descale() {
    print_test_header("TEST 3: FP8 E4M3 with Variable-length Sequences + Descaling");

    const int batch_size = 3;
    const int num_heads = 16;
    const int head_dim = 128;

    // Variable sequence lengths
    std::vector<int> seqlens_q = {256, 512, 128};  // Different lengths per batch
    std::vector<int> seqlens_k = {256, 512, 128};

    // Compute cumulative sequence lengths
    std::vector<int> h_cu_seqlens_q(batch_size + 1);
    std::vector<int> h_cu_seqlens_k(batch_size + 1);
    h_cu_seqlens_q[0] = 0;
    h_cu_seqlens_k[0] = 0;
    for (int i = 0; i < batch_size; i++) {
        h_cu_seqlens_q[i + 1] = h_cu_seqlens_q[i] + seqlens_q[i];
        h_cu_seqlens_k[i + 1] = h_cu_seqlens_k[i] + seqlens_k[i];
    }
    int total_q = h_cu_seqlens_q[batch_size];
    int total_k = h_cu_seqlens_k[batch_size];

    std::cout << "Configuration:\n";
    std::cout << "  Mode: Varlen Prefill\n";
    std::cout << "  Dtype: FP8 E4M3\n";
    std::cout << "  Batch: " << batch_size << "\n";
    std::cout << "  Seqlens Q: [";
    for (int i = 0; i < batch_size; i++) {
        std::cout << seqlens_q[i];
        if (i < batch_size - 1) std::cout << ", ";
    }
    std::cout << "] (total=" << total_q << ")\n";
    std::cout << "  Seqlens K: [";
    for (int i = 0; i < batch_size; i++) {
        std::cout << seqlens_k[i];
        if (i < batch_size - 1) std::cout << ", ";
    }
    std::cout << "] (total=" << total_k << ")\n";
    std::cout << "  Heads: " << num_heads << " (MHA)\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  FP8 Descaling: YES (per-tensor)\n\n";

    // Allocate tensors (varlen format: contiguous across all sequences)
    size_t q_elements = total_q * num_heads * head_dim;
    size_t k_elements = total_k * num_heads * head_dim;

    std::vector<__nv_fp8_e4m3> h_q(q_elements), h_k(k_elements), h_v(k_elements);
    generate_random_fp8_e4m3(h_q);
    generate_random_fp8_e4m3(h_k);
    generate_random_fp8_e4m3(h_v);

    void *d_q, *d_k, *d_v, *d_out;
    float *d_lse;
    int *d_cu_seqlens_q, *d_cu_seqlens_k;
    float *d_q_descale, *d_k_descale, *d_v_descale;

    cudaMalloc(&d_q, q_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_k, k_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_v, k_elements * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_out, q_elements * sizeof(__half));  // Output is FP16
    cudaMalloc(&d_lse, total_q * num_heads * sizeof(float));
    cudaMalloc(&d_cu_seqlens_q, (batch_size + 1) * sizeof(int));
    cudaMalloc(&d_cu_seqlens_k, (batch_size + 1) * sizeof(int));
    cudaMalloc(&d_q_descale, sizeof(float));  // Single global scale
    cudaMalloc(&d_k_descale, sizeof(float));
    cudaMalloc(&d_v_descale, sizeof(float));

    cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), k_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cu_seqlens_q, h_cu_seqlens_q.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cu_seqlens_k, h_cu_seqlens_k.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Set descale factors (example: 0.1 for all)
    float descale_value = 0.1f;
    cudaMemcpy(d_q_descale, &descale_value, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_descale, &descale_value, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_descale, &descale_value, sizeof(float), cudaMemcpyHostToDevice);

    // Setup Flash Attention params
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_lse;

    params.batch_size = batch_size;
    params.seqlen_q = 0;  // Ignored with varlen
    params.seqlen_k = 0;  // Ignored with varlen
    params.num_heads = num_heads;
    params.num_heads_k = num_heads;
    params.head_dim = head_dim;

    // ✅ Variable-length sequences
    params.cu_seqlens_q = d_cu_seqlens_q;
    params.cu_seqlens_k = d_cu_seqlens_k;
    params.total_q = total_q;
    params.total_k = total_k;

    // ✅ FP8 descaling
    params.q_descale_ptr = d_q_descale;
    params.k_descale_ptr = d_k_descale;
    params.v_descale_ptr = d_v_descale;
    params.q_descale_batch_stride = 0;  // Global scaling
    params.q_descale_head_stride = 0;
    params.k_descale_batch_stride = 0;
    params.k_descale_head_stride = 0;
    params.v_descale_batch_stride = 0;
    params.v_descale_head_stride = 0;

    params.dtype = flash::DataType::FP8_E4M3;
    params.mode = flash::AttentionMode::VARLEN_PREFILL;

    // Run
    std::cout << "Running Flash Attention...\n";
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int result = flash::flash_attention_forward(params, stream);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();

    if (result == 0 && err == cudaSuccess) {
        std::cout << "✓ SUCCESS\n";
    } else {
        std::cout << "✗ FAILED: " << flash::get_error_string(result) << "\n";
        if (err != cudaSuccess) {
            std::cout << "  CUDA error: " << cudaGetErrorString(err) << "\n";
        }
    }

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaFree(d_cu_seqlens_q);
    cudaFree(d_cu_seqlens_k);
    cudaFree(d_q_descale);
    cudaFree(d_k_descale);
    cudaFree(d_v_descale);
    cudaStreamDestroy(stream);
}

//==============================================================================
// MAIN
//==============================================================================

int main(int argc, char** argv) {
    std::cout << "Flash Attention V2 API - Comprehensive Test Suite\n";
    std::cout << "==================================================\n";
    std::cout << "\nThis test suite demonstrates:\n";
    std::cout << "  1. FP16 with Causal Mask (Prefill)\n";
    std::cout << "  2. FP16 Decoding with GQA (seqlen_q=1)\n";
    std::cout << "  3. FP8 E4M3 with Variable-length + Descaling\n";

    // Run all tests
    test_fp16_causal_prefill();
    test_fp16_decode();
    test_fp8_varlen_descale();

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  All tests completed!\n";
    std::cout << "============================================================\n";

    return 0;
}
