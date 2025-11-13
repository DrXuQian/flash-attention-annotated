#include "../include/flash_api.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

int main(int argc, char** argv) {
    // Default configuration for Qwen2.5-VL-3B
    int batch_size = 1;
    int seqlen_q = 512;
    int seqlen_k = 512;
    int num_heads = 16;
    int head_dim = 128;

    std::cout << "Flash Attention Standalone Test" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Configuration (Qwen2.5-VL-3B):" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length Q: " << seqlen_q << std::endl;
    std::cout << "  Sequence length K/V: " << seqlen_k << std::endl;
    std::cout << "  Number of heads: " << num_heads << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Hidden size: " << (num_heads * head_dim) << std::endl;
    std::cout << std::endl;

    // Calculate tensor sizes
    size_t qkv_elements = batch_size * seqlen_q * num_heads * head_dim;
    size_t qkv_bytes = qkv_elements * sizeof(__half);

    // Allocate device memory
    void *d_q, *d_k, *d_v, *d_out;
    float *d_softmax_lse;

    cudaMalloc(&d_q, qkv_bytes);
    cudaMalloc(&d_k, qkv_bytes);
    cudaMalloc(&d_v, qkv_bytes);
    cudaMalloc(&d_out, qkv_bytes);
    cudaMalloc(&d_softmax_lse, batch_size * num_heads * seqlen_q * sizeof(float));

    // Initialize with random data (in real usage, copy from host)
    cudaMemset(d_q, 0, qkv_bytes);
    cudaMemset(d_k, 0, qkv_bytes);
    cudaMemset(d_v, 0, qkv_bytes);

    // Set up Flash Attention parameters
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_softmax_lse;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.num_heads = num_heads;
    params.num_heads_k = num_heads;  // No GQA for Qwen2.5-VL-3B
    params.head_dim = head_dim;

    params.is_causal = false;
    params.dtype = flash::DataType::FP16;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run Flash Attention
    std::cout << "Running Flash Attention forward pass..." << std::endl;
    int result = flash::flash_attention_forward(params, stream);

    if (result == 0) {
        std::cout << "✓ Flash Attention completed successfully!" << std::endl;
    } else {
        std::cout << "✗ Flash Attention failed with error: "
                  << flash::get_error_string(result) << std::endl;
    }

    // Synchronize and check for errors
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_softmax_lse);
    cudaStreamDestroy(stream);

    return result;
}