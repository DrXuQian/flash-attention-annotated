// Auto-generated kernel instantiation file
// Configuration: dtype=fp16, hdim=256
// This file is compiled separately to avoid register limit conflicts

#include "../hopper/flash_fwd_launch_template.h"

// Single kernel instantiation per file
template void run_mha_fwd_<90, cutlass::half_t, 256, 256, false, false, false, false>
    (Flash_fwd_params &params, cudaStream_t stream);
