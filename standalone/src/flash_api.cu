/******************************************************************************
 * Flash Attention Standalone API Implementation
 * This file contains only the API wrapper, no kernel instantiations.
 * Kernels are compiled separately to avoid register limit issues.
 ******************************************************************************/

#include "../include/flash_api.h"
#include "../hopper/flash.h"
#include "../hopper/flash_fwd_launch_template.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <iostream>

namespace flash {

// Forward declarations of kernel templates
// These are instantiated in separate kernel_*.cu files
// Template parameters: Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA
template<int Arch, typename T, int kHeadDim, int kHeadDimV,
         bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

// Helper to select and call the right kernel
template<typename DType>
static int dispatch_headdim(
    Flash_fwd_params& flash_params,
    int head_dim,
    cudaStream_t stream
) {
    switch (head_dim) {
        case 128:
            run_mha_fwd_<90, DType, 128, 128, false, false, false, false>(
                flash_params, stream);
            return 0;
        case 256:
            run_mha_fwd_<90, DType, 256, 256, false, false, false, false>(
                flash_params, stream);
            return 0;
        default:
            std::cerr << "Unsupported head_dim: " << head_dim
                      << ". Supported values: 128, 256" << std::endl;
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

    if (params.head_dim != 128 && params.head_dim != 256) {
        std::cerr << "Error: unsupported head_dim " << params.head_dim
                  << ". Only 128 and 256 are supported." << std::endl;
        return -2;
    }

    // Create Flash_fwd_params
    Flash_fwd_params flash_params;

    // Set pointers based on data type
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

    // Set scale
    float scale = params.softmax_scale;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(params.head_dim));
    }
    flash_params.scale_softmax = scale;
    // Note: scale_softmax_log2 was removed in newer versions
    // The kernel now computes this internally if needed

    // Set causal
    flash_params.is_causal = params.is_causal;

    // Set window sizes
    flash_params.window_size_left = params.window_size_left;
    flash_params.window_size_right = params.window_size_right;

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
            result = dispatch_headdim<cutlass::bfloat16_t>(
                flash_params, params.head_dim, stream);
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