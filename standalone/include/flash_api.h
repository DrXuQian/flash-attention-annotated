#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash {

// Data type enum
enum class DataType {
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2
};

// Flash Attention parameters
struct FlashAttentionParams {
    // Input/output pointers
    void* q;        // [batch, seqlen_q, num_heads, head_dim]
    void* k;        // [batch, seqlen_k, num_heads_k, head_dim]
    void* v;        // [batch, seqlen_k, num_heads_k, head_dim]
    void* out;      // [batch, seqlen_q, num_heads, head_dim]

    // Optional outputs
    void* softmax_lse = nullptr;  // [batch, num_heads, seqlen_q]

    // Dimensions
    int batch_size;
    int seqlen_q;
    int seqlen_k;
    int num_heads;
    int num_heads_k;  // For GQA/MQA
    int head_dim;

    // Attention parameters
    float softmax_scale = 0.0f;  // 0 means use 1/sqrt(head_dim)
    bool is_causal = false;

    // Data type
    DataType dtype = DataType::FP16;

    // Window size for local attention (-1 for full attention)
    int window_size_left = -1;
    int window_size_right = -1;
};

// Main API function
int flash_attention_forward(
    FlashAttentionParams& params,
    cudaStream_t stream = nullptr
);

// Helper functions
size_t get_workspace_size(const FlashAttentionParams& params);
const char* get_error_string(int error_code);

} // namespace flash