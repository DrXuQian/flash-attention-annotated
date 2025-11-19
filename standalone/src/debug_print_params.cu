#include "../hopper/flash.h"
#include <iostream>
#include <iomanip>

namespace flash {

void print_flash_fwd_params(const Flash_fwd_params& p, const char* label) {
    std::cout << "\n========== " << label << " ==========\n";

    // Pointers (just print if non-null)
    std::cout << "Pointers:\n";
    std::cout << "  q_ptr: " << (p.q_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  k_ptr: " << (p.k_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  v_ptr: " << (p.v_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  o_ptr: " << (p.o_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  softmax_lse_ptr: " << (p.softmax_lse_ptr ? "SET" : "NULL") << "\n";

    // Dimensions
    std::cout << "\nDimensions:\n";
    std::cout << "  b: " << p.b << "\n";
    std::cout << "  h: " << p.h << "\n";
    std::cout << "  h_k: " << p.h_k << "\n";
    std::cout << "  d: " << p.d << "\n";
    std::cout << "  dv: " << p.dv << "\n";
    std::cout << "  seqlen_q: " << p.seqlen_q << "\n";
    std::cout << "  seqlen_k: " << p.seqlen_k << "\n";
    std::cout << "  seqlen_q_rounded: " << p.seqlen_q_rounded << "\n";
    std::cout << "  seqlen_k_rounded: " << p.seqlen_k_rounded << "\n";
    std::cout << "  d_rounded: " << p.d_rounded << "\n";
    std::cout << "  dv_rounded: " << p.dv_rounded << "\n";
    std::cout << "  total_q: " << p.total_q << "\n";
    std::cout << "  total_k: " << p.total_k << "\n";

    // Strides
    std::cout << "\nStrides:\n";
    std::cout << "  q_batch_stride: " << p.q_batch_stride << "\n";
    std::cout << "  q_row_stride: " << p.q_row_stride << "\n";
    std::cout << "  q_head_stride: " << p.q_head_stride << "\n";
    std::cout << "  k_batch_stride: " << p.k_batch_stride << "\n";
    std::cout << "  k_row_stride: " << p.k_row_stride << "\n";
    std::cout << "  k_head_stride: " << p.k_head_stride << "\n";
    std::cout << "  v_batch_stride: " << p.v_batch_stride << "\n";
    std::cout << "  v_row_stride: " << p.v_row_stride << "\n";
    std::cout << "  v_head_stride: " << p.v_head_stride << "\n";
    std::cout << "  v_dim_stride: " << p.v_dim_stride << "\n";
    std::cout << "  o_batch_stride: " << p.o_batch_stride << "\n";
    std::cout << "  o_row_stride: " << p.o_row_stride << "\n";
    std::cout << "  o_head_stride: " << p.o_head_stride << "\n";

    // Flags
    std::cout << "\nFlags:\n";
    std::cout << "  is_bf16: " << p.is_bf16 << "\n";
    std::cout << "  is_e4m3: " << p.is_e4m3 << "\n";
    std::cout << "  is_causal: " << p.is_causal << "\n";
    std::cout << "  is_local: " << p.is_local << "\n";
    std::cout << "  window_size_left: " << p.window_size_left << "\n";
    std::cout << "  window_size_right: " << p.window_size_right << "\n";
    std::cout << "  pack_gqa: " << p.pack_gqa << "\n";

    // Varlen
    std::cout << "\nVarlen:\n";
    std::cout << "  cu_seqlens_q: " << (p.cu_seqlens_q ? "SET" : "NULL") << "\n";
    std::cout << "  cu_seqlens_k: " << (p.cu_seqlens_k ? "SET" : "NULL") << "\n";
    std::cout << "  seqused_q: " << (p.seqused_q ? "SET" : "NULL") << "\n";
    std::cout << "  seqused_k: " << (p.seqused_k ? "SET" : "NULL") << "\n";
    std::cout << "  leftpad_k: " << (p.leftpad_k ? "SET" : "NULL") << "\n";

    // Scheduler
    std::cout << "\nScheduler:\n";
    std::cout << "  num_splits: " << p.num_splits << "\n";
    std::cout << "  num_splits_dynamic_ptr: " << p.num_splits_dynamic_ptr << "\n";
    std::cout << "  skip_scheduler_metadata_computation: " << p.skip_scheduler_metadata_computation << "\n";
    std::cout << "  prepare_varlen_pdl: " << p.prepare_varlen_pdl << "\n";
    std::cout << "  tile_count_semaphore: " << (p.tile_count_semaphore ? "SET" : "NULL") << "\n";
    std::cout << "  num_m_blocks_ptr: " << (p.num_m_blocks_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  varlen_batch_idx_ptr: " << (p.varlen_batch_idx_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  varlen_sort_batches: " << p.varlen_sort_batches << "\n";
    std::cout << "  head_swizzle: " << p.head_swizzle << "\n";

    // Scale & softcap
    std::cout << "\nScale:\n";
    std::cout << "  scale_softmax: " << p.scale_softmax << "\n";
    std::cout << "  softcap: " << p.softcap << "\n";

    // Dropout
    std::cout << "\nDropout:\n";
    std::cout << "  p_dropout: " << p.p_dropout << "\n";
    std::cout << "  p_dropout_in_uint8_t: " << (int)p.p_dropout_in_uint8_t << "\n";
    std::cout << "  rp_dropout: " << p.rp_dropout << "\n";

    // FP8 descale
    std::cout << "\nFP8 Descale:\n";
    std::cout << "  q_descale_ptr: " << (p.q_descale_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  k_descale_ptr: " << (p.k_descale_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  v_descale_ptr: " << (p.v_descale_ptr ? "SET" : "NULL") << "\n";
    if (p.q_descale_ptr) {
        std::cout << "  q_descale_batch_stride: " << p.q_descale_batch_stride << "\n";
        std::cout << "  q_descale_head_stride: " << p.q_descale_head_stride << "\n";
    }

    // Architecture
    std::cout << "\nArchitecture:\n";
    std::cout << "  arch: " << p.arch << "\n";
    std::cout << "  num_sm: " << p.num_sm << "\n";

    // Rotary
    std::cout << "\nRotary:\n";
    std::cout << "  rotary_dim: " << p.rotary_dim << "\n";
    std::cout << "  rotary_cos_ptr: " << (p.rotary_cos_ptr ? "SET" : "NULL") << "\n";
    std::cout << "  is_rotary_interleaved: " << p.is_rotary_interleaved << "\n";

    // PagedKV
    std::cout << "\nPagedKV:\n";
    std::cout << "  page_table: " << (p.page_table ? "SET" : "NULL") << "\n";
    std::cout << "  page_size: " << p.page_size << "\n";
    std::cout << "  pagedkv_tma: " << p.pagedkv_tma << "\n";

    std::cout << "========================================\n\n";
}

} // namespace flash
