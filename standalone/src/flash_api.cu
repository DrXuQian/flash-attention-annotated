/******************************************************************************
 * Flash Attention Standalone API V2 - Enhanced Implementation
 *
 * Supports:
 * - FP16 with causal mask
 * - FP16 decoding (seqlen_q=1)
 * - FP8 E4M3 with variable-length sequences and descaling
 * - FP16 variable-length sequences
 * - GQA/MQA
 *
 * This file contains the API wrapper. Kernels are compiled separately.
 ******************************************************************************/

#include "../include/flash_api.h"
#include "../hopper/flash.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <cmath>

namespace flash {

//==============================================================================
// PARAMETER VALIDATION
//==============================================================================

int validate_params(const FlashAttentionParams& params) {
    // Null pointer checks
    if (!params.q || !params.k || !params.v || !params.out) {
        std::cerr << "Error: null pointers in Q/K/V/out" << std::endl;
        return -1;
    }

    // Head dimension validation
    if (params.head_dim != 64 && params.head_dim != 96 && params.head_dim != 128 &&
        params.head_dim != 192 && params.head_dim != 256) {
        std::cerr << "Error: unsupported head_dim " << params.head_dim
                  << ". Supported: 64, 96, 128, 192, 256" << std::endl;
        return -2;
    }

    // FP8 constraints
    if (params.dtype == DataType::FP8_E4M3) {
        if (params.head_dim != 64 && params.head_dim != 96 && params.head_dim != 128) {
            std::cerr << "Error: FP8 only supports head_dim=64, 96, or 128" << std::endl;
            return -3;
        }
    }

    // GQA validation
    if (params.num_heads_k > params.num_heads) {
        std::cerr << "Error: num_heads_k (" << params.num_heads_k
                  << ") cannot exceed num_heads (" << params.num_heads << ")" << std::endl;
        return -4;
    }
    if (params.num_heads % params.num_heads_k != 0) {
        std::cerr << "Error: num_heads (" << params.num_heads
                  << ") must be divisible by num_heads_k (" << params.num_heads_k << ")" << std::endl;
        return -5;
    }

    // Varlen validation
    bool is_varlen = (params.cu_seqlens_q != nullptr) || (params.cu_seqlens_k != nullptr);
    if (is_varlen) {
        if (params.cu_seqlens_q == nullptr || params.cu_seqlens_k == nullptr) {
            std::cerr << "Error: cu_seqlens_q and cu_seqlens_k must both be set or both be null" << std::endl;
            return -6;
        }
        if (params.total_q <= 0 || params.total_k <= 0) {
            std::cerr << "Error: total_q and total_k must be set when using varlen" << std::endl;
            return -7;
        }
    }

    // Unsupported features
    if (params.pack_gqa) {
        std::cerr << "Error: PackGQA is not supported (requires different kernel instantiations)" << std::endl;
        return -10;
    }
    if (params.softcap != 0.0f) {
        std::cerr << "Error: Softcap is not supported (requires Has_softcap=true kernels)" << std::endl;
        return -11;
    }
    if (params.p_dropout > 0.0f) {
        std::cerr << "Error: Dropout is not supported" << std::endl;
        return -12;
    }
    if (params.num_splits > 1) {
        std::cerr << "Error: Split-KV is not supported" << std::endl;
        return -13;
    }
    if (params.page_table != nullptr) {
        std::cerr << "Error: Paged KV cache is not supported" << std::endl;
        return -14;
    }
    if (params.rotary_cos != nullptr || params.rotary_sin != nullptr) {
        std::cerr << "Error: Rotary embeddings are not supported" << std::endl;
        return -15;
    }
    if (params.is_causal && (params.window_size_left >= 0 || params.window_size_right >= 0)) {
        std::cerr << "Error: Cannot be both causal and local attention" << std::endl;
        return -16;
    }

    return 0;  // Valid
}

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

// Helper to initialize Flash_fwd_params from FlashAttentionParams
static void initialize_flash_params(
    Flash_fwd_params& flash_params,
    const FlashAttentionParams& params
) {
    // Zero-initialize
    flash_params = {};

    // Architecture
    flash_params.arch = 90;  // SM90a (Hopper)

    // Data type flags
    flash_params.is_bf16 = (params.dtype == DataType::BF16);
    flash_params.is_e4m3 = (params.dtype == DataType::FP8_E4M3);
    flash_params.is_fp32 = false;

    // Pointers
    flash_params.q_ptr = params.q;
    flash_params.k_ptr = params.k;
    flash_params.v_ptr = params.v;
    flash_params.o_ptr = params.out;
    flash_params.softmax_lse_ptr = params.softmax_lse;

    // Dimensions
    bool is_varlen = (params.cu_seqlens_q != nullptr);

    flash_params.b = params.batch_size;
    flash_params.h = params.num_heads;
    flash_params.h_k = params.num_heads_k;
    flash_params.d = params.head_dim;
    flash_params.seqlen_q = is_varlen ? 0 : params.seqlen_q;  // Ignored if varlen
    flash_params.seqlen_k = is_varlen ? 0 : params.seqlen_k;
    flash_params.seqlen_knew = 0;

    // Total sequences (critical for varlen)
    flash_params.total_q = is_varlen ? params.total_q : (params.batch_size * params.seqlen_q);
    flash_params.total_k = is_varlen ? params.total_k : (params.batch_size * params.seqlen_k);
    flash_params.total_knew = 0;

    // Rounded dimensions
    flash_params.seqlen_q_rounded = ((params.seqlen_q + 128 - 1) / 128) * 128;
    flash_params.seqlen_k_rounded = ((params.seqlen_k + 128 - 1) / 128) * 128;
    flash_params.d_rounded = ((params.head_dim + 8 - 1) / 8) * 8;
    flash_params.dv = params.head_dim;
    flash_params.dv_rounded = flash_params.d_rounded;
    flash_params.rotary_dim = 0;

    // Strides (row-major: [batch, seqlen, heads, headdim])
    if (!is_varlen) {
        flash_params.q_batch_stride = params.seqlen_q * params.num_heads * params.head_dim;
        flash_params.k_batch_stride = params.seqlen_k * params.num_heads_k * params.head_dim;
        flash_params.v_batch_stride = params.seqlen_k * params.num_heads_k * params.head_dim;
        flash_params.o_batch_stride = params.seqlen_q * params.num_heads * params.head_dim;
    } else {
        // Varlen: batch stride is 0 (data is contiguous across all sequences)
        flash_params.q_batch_stride = 0;
        flash_params.k_batch_stride = 0;
        flash_params.v_batch_stride = 0;
        flash_params.o_batch_stride = 0;
    }

    flash_params.q_row_stride = params.num_heads * params.head_dim;
    flash_params.k_row_stride = params.num_heads_k * params.head_dim;
    flash_params.v_row_stride = params.num_heads_k * params.head_dim;
    flash_params.o_row_stride = params.num_heads * params.head_dim;

    flash_params.q_head_stride = params.head_dim;
    flash_params.k_head_stride = params.head_dim;
    flash_params.v_head_stride = params.head_dim;
    flash_params.o_head_stride = params.head_dim;
    flash_params.v_dim_stride = 1;  // Column-major within head

    // Softmax scale
    float scale = params.softmax_scale;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(params.head_dim));
    }
    flash_params.scale_softmax = scale;
    flash_params.softcap = params.softcap;

    // Masking
    flash_params.is_causal = params.is_causal;
    flash_params.window_size_left = params.window_size_left < 0 ? -1 : params.window_size_left;
    flash_params.window_size_right = params.window_size_right < 0 ? -1 : params.window_size_right;
    flash_params.is_local = (params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal;
    flash_params.attention_chunk = 0;

    // Variable-length sequences
    flash_params.cu_seqlens_q = params.cu_seqlens_q;
    flash_params.cu_seqlens_k = params.cu_seqlens_k;
    flash_params.cu_seqlens_knew = nullptr;
    flash_params.seqused_q = params.seqused_q;
    flash_params.seqused_k = params.seqused_k;
    flash_params.leftpad_k = params.leftpad_k;

    // FP8 descaling
    flash_params.q_descale_ptr = params.q_descale_ptr;
    flash_params.k_descale_ptr = params.k_descale_ptr;
    flash_params.v_descale_ptr = params.v_descale_ptr;
    flash_params.q_descale_batch_stride = params.q_descale_batch_stride;
    flash_params.q_descale_head_stride = params.q_descale_head_stride;
    flash_params.k_descale_batch_stride = params.k_descale_batch_stride;
    flash_params.k_descale_head_stride = params.k_descale_head_stride;
    flash_params.v_descale_batch_stride = params.v_descale_batch_stride;
    flash_params.v_descale_head_stride = params.v_descale_head_stride;

    // Dropout (not supported)
    flash_params.p_dropout = 1.0f;  // Keep probability = 1.0
    flash_params.p_dropout_in_uint8_t = 255;
    flash_params.rp_dropout = 1.0f;
    flash_params.rng_state = nullptr;

    // KV cache and paged attention (not supported)
    flash_params.knew_ptr = nullptr;
    flash_params.vnew_ptr = nullptr;
    flash_params.kv_batch_idx = nullptr;
    flash_params.page_table = nullptr;
    flash_params.page_table_batch_stride = 0;
    flash_params.page_size = 0;
    flash_params.num_pages = 0;
    flash_params.pagedkv_tma = false;

    // Rotary embeddings (not supported)
    flash_params.rotary_cos_ptr = nullptr;
    flash_params.rotary_sin_ptr = nullptr;
    flash_params.seqlens_rotary = nullptr;
    flash_params.is_rotary_interleaved = false;

    // Split-KV (not supported)
    flash_params.num_splits = 1;
    flash_params.oaccum_ptr = nullptr;
    flash_params.softmax_lseaccum_ptr = nullptr;
    flash_params.oaccum_split_stride = 0;
    flash_params.oaccum_batch_stride = 0;
    flash_params.oaccum_row_stride = 0;
    flash_params.oaccum_head_stride = 0;
    flash_params.lseaccum_split_stride = 0;
    flash_params.lseaccum_batch_stride = 0;
    flash_params.lseaccum_head_stride = 0;

    // PackGQA (not supported)
    flash_params.pack_gqa = false;

    // QV fusion (Hopper optimization, auto-detected by kernel)
    flash_params.qv_ptr = nullptr;
    flash_params.qv_batch_stride = 0;
    flash_params.qv_row_stride = 0;
    flash_params.qv_head_stride = 0;

    // Scheduler metadata (CRITICAL!)
    flash_params.tile_count_semaphore = nullptr;
    flash_params.num_m_blocks_ptr = nullptr;
    flash_params.num_splits_dynamic_ptr = nullptr;
    flash_params.varlen_batch_idx_ptr = nullptr;
    flash_params.num_nheads_in_l2_ptr = nullptr;
    flash_params.skip_scheduler_metadata_computation = !is_varlen;  // Only compute for varlen
    flash_params.varlen_sort_batches = false;
    flash_params.tile_count_semaphore_offset = 0;
    flash_params.head_swizzle = false;
    flash_params.prepare_varlen_pdl = false;

    // Device properties
    cudaDeviceProp device_prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&device_prop, device);
    flash_params.num_sm = device_prop.multiProcessorCount;
    flash_params.b_k = params.batch_size;
}

//==============================================================================
// KERNEL DISPATCH
//==============================================================================

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
        // case 192:
        //     run_mha_fwd_<90, DType, 192, 192, false, false, false, false>(
        //         flash_params, stream);
        //     return 0;
        // case 256:
        //     run_mha_fwd_<90, DType, 256, 256, false, false, false, false>(
        //         flash_params, stream);
        //     return 0;
        default:
            std::cerr << "Unsupported head_dim: " << head_dim << std::endl;
            return -1;
    }
}

//==============================================================================
// MAIN API IMPLEMENTATION
//==============================================================================

int flash_attention_forward(
    FlashAttentionParams& params,
    cudaStream_t stream
) {
    // Validate parameters
    int validation_result = validate_params(params);
    if (validation_result != 0) {
        return validation_result;
    }

    // Initialize Flash_fwd_params
    Flash_fwd_params flash_params;
    initialize_flash_params(flash_params, params);

    // Set stream
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
            return -20;

        case DataType::FP8_E4M3:
            result = dispatch_headdim<cutlass::float_e4m3_t>(
                flash_params, params.head_dim, stream);
            break;

        case DataType::FP8_E5M2:
            std::cerr << "Error: FP8 E5M2 is not supported in this build" << std::endl;
            return -21;

        default:
            std::cerr << "Error: unsupported data type" << std::endl;
            return -22;
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
        return -100;
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
        case -1: return "Null pointer in Q/K/V/out";
        case -2: return "Unsupported head dimension";
        case -3: return "FP8 only supports head_dim=64, 96, or 128";
        case -4: return "num_heads_k cannot exceed num_heads";
        case -5: return "num_heads must be divisible by num_heads_k";
        case -6: return "cu_seqlens_q and cu_seqlens_k must both be set or both be null";
        case -7: return "total_q and total_k must be set when using varlen";
        case -10: return "PackGQA is not supported";
        case -11: return "Softcap is not supported";
        case -12: return "Dropout is not supported";
        case -13: return "Split-KV is not supported";
        case -14: return "Paged KV cache is not supported";
        case -15: return "Rotary embeddings are not supported";
        case -16: return "Cannot be both causal and local attention";
        case -20: return "BF16 is not supported in this build";
        case -21: return "FP8 E5M2 is not supported in this build";
        case -22: return "Unsupported data type";
        case -100: return "CUDA runtime error";
        default: return "Unknown error";
    }
}

} // namespace flash
