# Flash Attention Standalone Executable

Standalone Flash Attention executable for Hopper (SM90) architecture without PyTorch dependencies.

## Features
- Flash Attention v3 for Hopper (H100/H800)
- Support for FP16, BF16, and FP8 (E4M3) data types
- Optimized for Qwen2.5-VL-3B configuration (16 heads, 128 head_dim)
- Complete separation compilation to avoid register limit issues
- No PyTorch/ATen dependencies

## Architecture
The compilation uses complete kernel separation to avoid register pressure issues:
- Each kernel configuration (dtype, hdim) is compiled in its own translation unit
- Allows `setmaxnreg` dynamic register allocation to work properly
- Matches the compilation behavior of `python setup.py install`

## Directory Structure
```
standalone/
├── CMakeLists.txt           # Main build configuration
├── src/
│   ├── flash_api.cu         # API wrapper (no kernel instantiation)
│   └── main.cpp             # Test executable
├── kernels/                 # Separated kernel files
│   ├── kernel_fp16_hdim128.cu
│   ├── kernel_fp16_hdim256.cu
│   ├── kernel_bf16_hdim128.cu
│   ├── kernel_bf16_hdim256.cu
│   └── kernel_fp8_e4m3_hdim128.cu
├── include/
│   └── flash_api.h          # Public API header
└── generate_kernels.py      # Kernel generation script
```

## Requirements
- CUDA 11.8+ (for Hopper support)
- CMake 3.18+
- C++17 compiler
- Hopper GPU (SM90/SM90a)

## Build Instructions

### 1. Generate kernel files
```bash
cd standalone
python3 generate_kernels.py
```

### 2. Configure with CMake
```bash
mkdir build
cd build
cmake ..
```

### 3. Build
```bash
make -j4
```

### 4. Run test
```bash
./flash_attention_exec
```

## API Usage

```cpp
#include "flash_api.h"

// Set up parameters
flash::FlashAttentionParams params;
params.q = d_q;          // Device pointer to Q tensor
params.k = d_k;          // Device pointer to K tensor
params.v = d_v;          // Device pointer to V tensor
params.out = d_out;      // Device pointer to output tensor

params.batch_size = 1;
params.seqlen_q = 512;
params.seqlen_k = 512;
params.num_heads = 16;
params.num_heads_k = 16;
params.head_dim = 128;

params.dtype = flash::DataType::FP16;
params.is_causal = false;

// Run Flash Attention
int result = flash::flash_attention_forward(params, stream);
```

## Supported Configurations

| Data Type | Head Dimensions | Status |
|-----------|----------------|--------|
| FP16      | 128, 256       | ✓      |
| BF16      | 128, 256       | ✓      |
| FP8 E4M3  | 128            | ✓      |

## Performance Notes

- The separate compilation strategy ensures optimal register usage
- Each kernel can use dynamic register reconfiguration (`setmaxnreg`)
- No register pressure conflicts between different kernel configurations

## Troubleshooting

### Register limit errors
If you see "too many resources requested for launch", ensure you're using the separate compilation approach as implemented here.

### setmaxnreg warnings
The warning "setmaxnreg ignored to maintain compatibility" indicates kernels are not properly separated. This implementation avoids this issue.

## License
Same as Flash Attention main repository.