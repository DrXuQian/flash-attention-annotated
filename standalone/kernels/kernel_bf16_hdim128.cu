// Auto-generated kernel instantiation file
// Configuration: dtype=bf16, hdim=128
// This file is compiled separately to avoid register limit conflicts

#include "../hopper/flash_fwd_launch_template.h"

// Single kernel instantiation per file
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>
    (Flash_fwd_params &params, cudaStream_t stream);
