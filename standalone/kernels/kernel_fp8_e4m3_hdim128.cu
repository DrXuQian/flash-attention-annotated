// Auto-generated kernel instantiation file
// Configuration: dtype=fp8_e4m3, hdim=128
// This file is compiled separately to avoid register limit conflicts

#include "../hopper/flash_fwd_launch_template.h"

// Single kernel instantiation per file
// Template parameters: Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA
template void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, false, false, false>
    (Flash_fwd_params &params, cudaStream_t stream);
