#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash {

// Data type enum
enum class DataType {
    FP16,
    BF16,        // NOT SUPPORTED in this build
    FP8_E4M3,
    FP8_E5M2     // NOT SUPPORTED in this build
};

// Attention mode enum for better API clarity
enum class AttentionMode {
    // Standard attention with full Q, K, V matrices
    PREFILL,           // [batch, seqlen_q, num_heads, head_dim]

    // Decoding mode: seqlen_q = 1, seqlen_k = full context length
    DECODE,            // Q: [batch, 1, num_heads, head_dim], K/V: [batch, seqlen_k, num_heads_k, head_dim]

    // Variable-length sequences (uses cu_seqlens)
    VARLEN_PREFILL,    // Variable length prefill with cu_seqlens_q/k
    VARLEN_DECODE      // Variable length decode (NOT FULLY TESTED)
};

// Flash Attention V2 parameters - Extended version with full feature support
struct FlashAttentionParams {
    //==========================================================================
    // BASIC INPUT/OUTPUT TENSORS (Required for all modes)
    //==========================================================================

    void* q;        // Query:  [batch, seqlen_q, num_heads, head_dim] or varlen format
    void* k;        // Key:    [batch, seqlen_k, num_heads_k, head_dim] or varlen format
    void* v;        // Value:  [batch, seqlen_k, num_heads_k, head_dim] or varlen format
    void* out;      // Output: [batch, seqlen_q, num_heads, head_dim] or varlen format

    // Optional outputs
    float* softmax_lse = nullptr;  // [batch, num_heads, seqlen_q] - Log-sum-exp for backward pass

    //==========================================================================
    // DIMENSIONS (Required)
    //==========================================================================

    int batch_size;      // Number of sequences in batch
    int seqlen_q;        // Query sequence length (ignored if cu_seqlens_q is set)
    int seqlen_k;        // Key/Value sequence length (ignored if cu_seqlens_k is set)
    int num_heads;       // Number of query heads
    int num_heads_k;     // Number of KV heads (for GQA/MQA, must divide num_heads)
    int head_dim;        // Head dimension: 64, 96, 128, 192, 256

    //==========================================================================
    // VARIABLE-LENGTH SEQUENCES (Optional - for VARLEN modes)
    //==========================================================================

    // ✅ SUPPORTED: Variable-length sequences
    int* cu_seqlens_q = nullptr;  // [batch + 1] cumulative sequence lengths for Q
    int* cu_seqlens_k = nullptr;  // [batch + 1] cumulative sequence lengths for K/V

    // When cu_seqlens_q/k are set:
    // - seqlen_q/k are ignored for input/output shapes
    // - max_seqlen_q/k and total_q/k must be set
    int max_seqlen_q = 0;  // Maximum sequence length in Q (required for varlen)
    int max_seqlen_k = 0;  // Maximum sequence length in K/V (required for varlen)
    int total_q = 0;       // Total number of query tokens across all sequences
    int total_k = 0;       // Total number of key/value tokens across all sequences

    // ✅ SUPPORTED: Actual sequence lengths (for padding)
    int* seqused_q = nullptr;  // [batch] actual length of each Q sequence (if < seqlen_q)
    int* seqused_k = nullptr;  // [batch] actual length of each K/V sequence (if < seqlen_k)

    // ❌ NOT SUPPORTED: Left padding for K/V
    int* leftpad_k = nullptr;  // Left padding - NOT IMPLEMENTED

    //==========================================================================
    // FP8 QUANTIZATION (Optional - only for FP8_E4M3 dtype)
    //==========================================================================

    // ✅ SUPPORTED: FP8 E4M3 with descaling
    // Descale factors for dequantizing FP8 inputs to FP32 before computation
    // If provided, values are: float_value = fp8_value * descale_factor
    float* q_descale_ptr = nullptr;  // [batch, num_heads] or [1] for global scaling
    float* k_descale_ptr = nullptr;  // [batch, num_heads_k] or [1] for global scaling
    float* v_descale_ptr = nullptr;  // [batch, num_heads_k] or [1] for global scaling

    // Descale strides (set to 0 if using global scaling)
    int64_t q_descale_batch_stride = 0;
    int64_t q_descale_head_stride = 0;
    int64_t k_descale_batch_stride = 0;
    int64_t k_descale_head_stride = 0;
    int64_t v_descale_batch_stride = 0;
    int64_t v_descale_head_stride = 0;

    //==========================================================================
    // ATTENTION MASKING
    //==========================================================================

    // ✅ SUPPORTED: Causal masking (autoregressive attention)
    bool is_causal = false;

    // ✅ SUPPORTED: Local/sliding window attention
    int window_size_left = -1;   // -1 means full attention on left side
    int window_size_right = -1;  // -1 means full attention on right side
    // Note: Cannot be both causal and local at the same time

    // ❌ NOT SUPPORTED: Custom attention mask
    void* attn_mask = nullptr;   // NOT IMPLEMENTED

    //==========================================================================
    // ATTENTION SCALING
    //==========================================================================

    float softmax_scale = 0.0f;  // 0 means use 1/sqrt(head_dim)

    // ❌ NOT SUPPORTED: Softcapping (for Gemini/Gemma models)
    float softcap = 0.0f;        // NOT IMPLEMENTED (would require Has_softcap=true kernels)

    //==========================================================================
    // DATA TYPE
    //==========================================================================

    DataType dtype = DataType::FP16;

    //==========================================================================
    // GROUPED QUERY ATTENTION (GQA)
    //==========================================================================

    // ✅ SUPPORTED: GQA/MQA (num_heads_k < num_heads)
    // Requirement: num_heads % num_heads_k == 0

    // ❌ NOT SUPPORTED: PackGQA optimization
    // PackGQA is a memory layout optimization where Q heads are packed by KV head groups
    // Would require: PackGQA=true kernel instantiations
    bool pack_gqa = false;  // MUST BE FALSE

    //==========================================================================
    // PAGED KV CACHE (For serving/inference)
    //==========================================================================

    // ❌ NOT SUPPORTED: Paged KV cache
    void* page_table = nullptr;           // NOT IMPLEMENTED
    int page_size = 0;                    // NOT IMPLEMENTED
    int* kv_batch_idx = nullptr;          // NOT IMPLEMENTED

    // ❌ NOT SUPPORTED: Appending new KV
    void* k_new = nullptr;                // NOT IMPLEMENTED
    void* v_new = nullptr;                // NOT IMPLEMENTED
    int seqlen_knew = 0;                  // NOT IMPLEMENTED

    //==========================================================================
    // ROTARY POSITION EMBEDDINGS (RoPE)
    //==========================================================================

    // ❌ NOT SUPPORTED: Rotary embeddings
    void* rotary_cos = nullptr;           // NOT IMPLEMENTED
    void* rotary_sin = nullptr;           // NOT IMPLEMENTED
    int rotary_dim = 0;                   // NOT IMPLEMENTED
    bool is_rotary_interleaved = false;   // NOT IMPLEMENTED

    //==========================================================================
    // DROPOUT
    //==========================================================================

    // ❌ NOT SUPPORTED: Attention dropout
    float p_dropout = 0.0f;               // MUST BE 0.0
    uint64_t* rng_state = nullptr;        // NOT IMPLEMENTED

    //==========================================================================
    // SPLIT-KV (For very long sequences)
    //==========================================================================

    // ❌ NOT SUPPORTED: Split-KV along sequence dimension
    int num_splits = 1;                   // MUST BE 1
    void* out_accum = nullptr;            // NOT IMPLEMENTED
    float* softmax_lse_accum = nullptr;   // NOT IMPLEMENTED

    //==========================================================================
    // ATTENTION MODE (Helper for API)
    //==========================================================================

    AttentionMode mode = AttentionMode::PREFILL;
};

//==============================================================================
// MAIN API FUNCTIONS
//==============================================================================

/**
 * Flash Attention Forward Pass
 *
 * Supported configurations:
 *
 * 1. FP16 Prefill with Causal Mask:
 *    - dtype = FP16
 *    - is_causal = true
 *    - cu_seqlens_q/k = nullptr
 *    - All descale pointers = nullptr
 *
 * 2. FP16 Decoding (seqlen_q = 1):
 *    - dtype = FP16
 *    - seqlen_q = 1
 *    - seqlen_k = context_length
 *    - Typically: is_causal = false (already decoded up to this point)
 *
 * 3. FP8 E4M3 with Variable-length + Descaling:
 *    - dtype = FP8_E4M3
 *    - cu_seqlens_q != nullptr, cu_seqlens_k != nullptr
 *    - q_descale_ptr != nullptr, k_descale_ptr != nullptr, v_descale_ptr != nullptr
 *    - total_q and total_k must be set
 *
 * 4. FP16 Variable-length Prefill:
 *    - dtype = FP16
 *    - cu_seqlens_q != nullptr, cu_seqlens_k != nullptr
 *    - total_q and total_k must be set
 *
 * Returns:
 *   0 on success
 *   Negative error code on failure (use get_error_string() to decode)
 */
int flash_attention_forward(
    FlashAttentionParams& params,
    cudaStream_t stream = nullptr
);

/**
 * Get workspace size needed (returns 0 for FA3 forward pass)
 */
size_t get_workspace_size(const FlashAttentionParams& params);

/**
 * Get human-readable error string
 */
const char* get_error_string(int error_code);

/**
 * Validate parameters before calling forward pass
 * Returns error code (0 = valid)
 */
int validate_params(const FlashAttentionParams& params);

} // namespace flash
