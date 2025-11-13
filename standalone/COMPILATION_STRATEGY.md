# Flash Attention Standalone Compilation Strategy

## Goal
Create a standalone executable for Flash Attention on Hopper (SM90) architecture without PyTorch dependencies.

## Key Requirements
1. Support Qwen2.5-VL-3B configuration (16 heads, 128 head_dim)
2. Support FP16 and FP8 data types
3. Avoid register limit issues from previous attempts
4. Match the compilation behavior of `python setup.py install`

## Register Limit Problem Analysis

### Root Cause
- CUTLASS uses `__launch_bounds__` to set max registers per thread
- When multiple kernel templates are in the same compilation unit, they must share the same register limit
- Flash Attention's Hopper kernels use `setmaxnreg` for dynamic register allocation
- `setmaxnreg` gets ignored when kernels are in the same compilation unit

### Solution: Complete Separation
Each kernel configuration must be compiled in its own translation unit:
- One .cu file per (dtype, hdim) combination
- Each file contains exactly one template instantiation
- Link all object files at the final stage

## Directory Structure
```
standalone/
├── CMakeLists.txt                  # Main build configuration
├── src/
│   ├── flash_api.cu                # API wrapper (no kernel instantiation)
│   ├── flash_api.h                 # API header
│   └── main.cpp                    # Test executable
├── kernels/                        # Generated kernel files
│   ├── kernel_fp16_hdim128.cu      # FP16, hdim=128
│   ├── kernel_fp8_hdim128.cu       # FP8 E4M3, hdim=128
│   └── ...                         # Other configurations
├── include/                        # Standalone headers
│   └── flash_fwd_params.h          # Parameter structures
└── scripts/
    └── generate_kernels.py         # Generate kernel files
```

## Compilation Flags
Based on setup.py analysis:
```
-O3
-std=c++17
-gencode arch=compute_90a,code=sm_90a  # Use sm_90a not sm_90
-U__CUDA_NO_HALF_OPERATORS__
-U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__
-U__CUDA_NO_BFLOAT16_CONVERSIONS__
--expt-relaxed-constexpr
--expt-extended-lambda
--use_fast_math
-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
-DCUTLASS_ENABLE_GDC_FOR_SM90
```

## Build Process
1. Generate kernel instantiation files (one per configuration)
2. Compile each kernel file to an object file
3. Compile API wrapper separately
4. Link all objects into final executable

## Verification
- Each kernel compiles independently
- No `setmaxnreg` warnings
- No register limit errors
- CMake configuration succeeds