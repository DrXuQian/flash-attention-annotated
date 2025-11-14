/******************************************************************************
 * Flash Attention Standalone API Implementation
 * This file contains only the API wrapper, no kernel instantiations.
 * Kernels are compiled separately to avoid register limit issues.
 ******************************************************************************/

#include "../include/flash_api.h"
#include "../hopper/flash.h"
// DO NOT include flash_fwd_launch_template.h - it causes multiple template instantiations
// #include "../hopper/flash_fwd_launch_template.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <iostream>

namespace flash {

// Note: run_mha_fwd_ template is already declared in hopper/flash.h (line 219)
// We do NOT redeclare it here to avoid "more than one instance" conflicts
// The template instantiations are provided by separate kernel files

// Helper to select and call the right kernel
template<typename DType>
static int dispatch_headdim(
    Flash_fwd_params& flash_params,
    int head_dim,
    cudaStream_t stream
) {
    switch (head_dim) {
        case 64:
            run_mha_fwd_<90, DType, 64, 64, false, false, false, false>(
                flash_params, stream);
            return 0;
        case 96:
            run_mha_fwd_<90, DType, 96, 96, false, false, false, false>(
                flash_params, stream);
            return 0;
        case 128:
            run_mha_fwd_<90, DType, 128, 128, false, false, false, false>(
                flash_params, stream);
            return 0;
        case 192:
            run_mha_fwd_<90, DType, 192, 192, false, false, false, false>(
                flash_params, stream);
            return 0;
        case 256:
            run_mha_fwd_<90, DType, 256, 256, false, false, false, false>(
                flash_params, stream);
            return 0;
        default:
            std::cerr << "Unsupported head_dim: " << head_dim
                      << ". Supported values: 64, 96, 128, 192, 256" << std::endl;
            return -1;
    }
}

int flash_attention_forward(
    FlashAttentionParams& params,
    cudaStream_t stream
) {
    // Validate parameters
    if (!params.q || !params.k || !params.v || !params.out) {
        std::cerr << "Error: null pointers in input/output" << std::endl;
        return -1;
    }

    // Validate head_dim
    if (params.head_dim != 64 && params.head_dim != 96 && params.head_dim != 128 &&
        params.head_dim != 192 && params.head_dim != 256) {
        std::cerr << "Error: unsupported head_dim " << params.head_dim
                  << ". Supported values: 64, 96, 128, 192, 256" << std::endl;
        return -2;
    }

    // Create Flash_fwd_params and initialize to zero
    Flash_fwd_params flash_params = {};

    // Set architecture (SM90a for Hopper)
    flash_params.arch = 90;

    // Set data type flags
    flash_params.is_bf16 = (params.dtype == DataType::BF16);
    flash_params.is_e4m3 = (params.dtype == DataType::FP8_E4M3);
    flash_params.is_fp32 = false;

    // Split-KV parameters (no split for now)
    flash_params.num_splits = 1;
    flash_params.num_splits_dynamic_ptr = nullptr;
    flash_params.pack_gqa = false;

    // Set pointers
    flash_params.q_ptr = params.q;
    flash_params.k_ptr = params.k;
    flash_params.v_ptr = params.v;
    flash_params.o_ptr = params.out;
    flash_params.softmax_lse_ptr = params.softmax_lse;

    // Set dimensions
    flash_params.b = params.batch_size;
    flash_params.h = params.num_heads;
    flash_params.h_k = params.num_heads_k;
    flash_params.d = params.head_dim;
    flash_params.seqlen_q = params.seqlen_q;
    flash_params.seqlen_k = params.seqlen_k;

    // Set rounded dimensions (round up to multiples for alignment)
    // For Hopper, round to 128 for seqlen and to 8/16 for head_dim
    flash_params.seqlen_q_rounded = ((params.seqlen_q + 128 - 1) / 128) * 128;
    flash_params.seqlen_k_rounded = ((params.seqlen_k + 128 - 1) / 128) * 128;
    flash_params.d_rounded = ((params.head_dim + 8 - 1) / 8) * 8;

    // V dimension (same as d for our case)
    flash_params.dv = params.head_dim;
    flash_params.dv_rounded = flash_params.d_rounded;

    // Total sequences (for varlen, but we use fixed length)
    flash_params.total_q = params.batch_size * params.seqlen_q;
    flash_params.total_k = params.batch_size * params.seqlen_k;

    // Variable length sequence parameters (set to nullptr for fixed length)
    flash_params.cu_seqlens_q = nullptr;
    flash_params.cu_seqlens_k = nullptr;
    flash_params.seqused_q = nullptr;
    flash_params.seqused_k = nullptr;

    // Set strides (assuming row-major layout)
    flash_params.q_batch_stride = params.seqlen_q * params.num_heads * params.head_dim;
    flash_params.k_batch_stride = params.seqlen_k * params.num_heads_k * params.head_dim;
    flash_params.v_batch_stride = params.seqlen_k * params.num_heads_k * params.head_dim;
    flash_params.o_batch_stride = params.seqlen_q * params.num_heads * params.head_dim;

    flash_params.q_row_stride = params.num_heads * params.head_dim;
    flash_params.k_row_stride = params.num_heads_k * params.head_dim;
    flash_params.v_row_stride = params.num_heads_k * params.head_dim;
    flash_params.o_row_stride = params.num_heads * params.head_dim;

    flash_params.q_head_stride = params.head_dim;
    flash_params.k_head_stride = params.head_dim;
    flash_params.v_head_stride = params.head_dim;
    flash_params.o_head_stride = params.head_dim;
    flash_params.v_dim_stride = 1;  // Contiguous in last dimension

    // Set scale
    float scale = params.softmax_scale;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(params.head_dim));
    }
    flash_params.scale_softmax = scale;
    flash_params.softcap = 0.0f;  // No softcap

    // Set causal and local attention
    flash_params.is_causal = params.is_causal;
    flash_params.window_size_left = params.window_size_left < 0 ? -1 : params.window_size_left;
    flash_params.window_size_right = params.window_size_right < 0 ? -1 : params.window_size_right;
    flash_params.is_local = (params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal;
    flash_params.attention_chunk = 0;

    // Dropout parameters (no dropout for now)
    flash_params.p_dropout = 1.0f;  // Keep probability = 1.0 (no dropout)
    flash_params.p_dropout_in_uint8_t = 255;
    flash_params.rp_dropout = 1.0f;

    // FP8 descale pointers (nullptr for FP16)
    flash_params.q_descale_ptr = nullptr;
    flash_params.k_descale_ptr = nullptr;
    flash_params.v_descale_ptr = nullptr;

    // KV cache and paged attention (not used)
    flash_params.knew_ptr = nullptr;
    flash_params.vnew_ptr = nullptr;
    flash_params.kv_batch_idx = nullptr;
    flash_params.page_table = nullptr;
    flash_params.pagedkv_tma = false;

    // Rotary embedding (not used)
    flash_params.rotary_cos_ptr = nullptr;
    flash_params.rotary_sin_ptr = nullptr;
    flash_params.rotary_dim = 0;
    flash_params.is_rotary_interleaved = false;

    // RNG state (for dropout, not used)
    flash_params.rng_state = nullptr;

    // Accumulator pointers (for split-kv, not used)
    flash_params.oaccum_ptr = nullptr;
    flash_params.softmax_lseaccum_ptr = nullptr;

    // Default stream
    if (stream == nullptr) {
        stream = cudaStreamDefault;
    }

    // Dispatch based on data type
    int result = 0;
    switch (params.dtype) {
        case DataType::FP16:
            result = dispatch_headdim<cutlass::half_t>(
                flash_params, params.head_dim, stream);
            break;
        case DataType::BF16:
            std::cerr << "Error: BF16 is not supported in this build" << std::endl;
            return -4;
            break;
        case DataType::FP8_E4M3:
            // FP8 only supports hdim=128 for now
            if (params.head_dim != 128) {
                std::cerr << "Error: FP8 only supports head_dim=128" << std::endl;
                return -3;
            }
            run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, false, false, false>(
                flash_params, stream);
            break;
        default:
            std::cerr << "Error: unsupported data type" << std::endl;
            return -4;
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -5;
    }

    return result;
}

size_t get_workspace_size(const FlashAttentionParams& params) {
    // Flash Attention v3 doesn't require workspace for forward pass
    return 0;
}

const char* get_error_string(int error_code) {
    switch (error_code) {
        case 0: return "Success";
        case -1: return "Null pointer in input/output";
        case -2: return "Unsupported head dimension";
        case -3: return "FP8 only supports head_dim=128";
        case -4: return "Unsupported data type";
        case -5: return "CUDA runtime error";
        default: return "Unknown error";
    }
}

} // namespace flash