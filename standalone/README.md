# Flash Attention Standalone

Standalone Flash Attention v3 for Hopper architecture (H100/H800/RTX 5090), extracted for easy integration without PyTorch dependencies.

## Features

- **Hopper Architecture**: Optimized for SM90a (H100/H800/RTX 5090)
- **No PyTorch Required**: Pure CUDA/C++ implementation
- **Official Kernels**: Uses kernel files from `hopper/instantiations/`
- **FP16 and FP8**: Supports half-precision and FP8 E4M3 formats
- **Qwen2.5-VL Ready**: Pre-configured for head_dim=128

## Requirements

- CUDA 11.8 or later
- CMake 3.18+
- NVIDIA GPU with Hopper architecture (compute capability 9.0)
- GCC/G++ with C++17 support

## Quick Start

```bash
# Build
./build.sh

# Run with default settings (FP16, batch=1, seq=512, heads=16, dim=128)
./build/flash_attention_exec

# Show help
./build/flash_attention_exec -h

# Run with FP8 E4M3
./build/flash_attention_exec -d fp8

# Custom configuration
./build/flash_attention_exec -b 2 -q 1024 -k 1024 -n 32

# Enable causal attention
./build/flash_attention_exec -c
```

## Command Line Options

```
Usage: flash_attention_exec [OPTIONS]

Options:
  -h, --help              Show help message
  -d, --dtype TYPE        Data type: fp16, fp8 (default: fp16)
  -b, --batch SIZE        Batch size (default: 1)
  -q, --seqlen-q LENGTH   Query sequence length (default: 512)
  -k, --seqlen-k LENGTH   Key/Value sequence length (default: 512)
  -n, --num-heads HEADS   Number of attention heads (default: 16)
  -m, --head-dim DIM      Head dimension: 128 or 256 (default: 128)
  -c, --causal            Enable causal masking (default: false)

Examples:
  ./build/flash_attention_exec                   # Defaults
  ./build/flash_attention_exec -d fp8            # FP8 mode
  ./build/flash_attention_exec -b 2 -q 1024      # Batch=2, seqlen=1024
  ./build/flash_attention_exec -c                # Causal attention
```

## Build Details

The build script:
1. Checks for kernel files in `../hopper/instantiations/`
2. Generates kernels if needed using `hopper/generate_kernels.py`
3. Compiles with CMake using separated compilation strategy
4. Links all components into `flash_attention_exec`

### Kernel Files Used

- `flash_fwd_hdim128_fp16_sm90.cu` - FP16, head_dim=128
- `flash_fwd_hdim128_e4m3_sm90.cu` - FP8 E4M3, head_dim=128
- `flash_fwd_hdim256_fp16_sm90.cu` - FP16, head_dim=256

### Compilation Strategy

Each kernel is compiled as a separate object library to avoid register limit conflicts. This matches the behavior of `python setup.py install` and allows proper use of Hopper's `setmaxnreg` dynamic register allocation.

## Directory Structure

```
standalone/
├── CMakeLists.txt              # Build configuration
├── build.sh                    # Build script
├── src/
│   ├── flash_api.cu            # API wrapper
│   └── main.cpp                # Example usage
└── include/
    └── flash_attention_api.h   # C++ API header
```

## API Usage

```cpp
#include "flash_attention_api.h"

// Configure parameters
Flash_fwd_params params;
params.batch_size = 1;
params.num_heads = 16;
params.head_dim = 128;
params.seqlen_q = 1024;
params.seqlen_k = 1024;
// ... set Q, K, V, output pointers ...

// Run Flash Attention
int status = flash_attention_forward(
    params.q_ptr,
    params.k_ptr,
    params.v_ptr,
    params.o_ptr,
    params.batch_size,
    params.seqlen_q,
    params.seqlen_k,
    params.num_heads,
    params.num_heads_k,
    params.head_dim,
    /*is_causal=*/true,
    /*dtype=*/0,  // 0=FP16, 2=FP8_E4M3
    /*scale=*/1.0f / sqrtf(128.0f)
);
```

## Integration

To use in your project:

```cmake
# Link the standalone library
add_executable(your_app main.cpp)
target_include_directories(your_app PRIVATE ${CMAKE_SOURCE_DIR}/standalone/include)
target_link_libraries(your_app PRIVATE flash_attention_lib)
```

## Supported Configurations

| Data Type | Head Dimensions | Architecture |
|-----------|-----------------|--------------|
| FP16      | 128, 256        | SM90a        |
| FP8 E4M3  | 128             | SM90a        |

**Note**: BF16 support removed per project requirements.

## Testing

The test program generates random data for both Q, K, and V tensors:

- **FP16 mode**: Random half-precision floating point values in range [-1, 1]
- **FP8 mode**: Random FP8 E4M3 values in range [-1, 1]

Example output:
```
Flash Attention Standalone Test
================================
Configuration (Qwen2.5-VL-3B):
  Data type: FP16
  Batch size: 1
  Sequence length Q: 512
  Sequence length K/V: 512
  Number of heads: 16
  Head dimension: 128

Generating random test data...
  Elements per tensor: 1048576
  Bytes per tensor: 2097152 (2.0 MB)
  Generating FP16 data...
  ✓ Test data generated and copied to device

Running Flash Attention forward pass...
✓ Flash Attention completed successfully!
```

## Troubleshooting

**Kernel files not found:**
```bash
cd ../hopper
python3 generate_kernels.py
cd ../standalone
./build.sh
```

**Register limit errors:**
This implementation uses separated compilation to prevent register conflicts. Each kernel is compiled in its own translation unit.

**Undefined reference to prepare_varlen_num_blocks:**
This is now fixed - `flash_prepare_scheduler.cu` is automatically compiled and linked.

## Technical Notes

- Uses official Flash Attention v3 Hopper kernels
- Separate compilation prevents register allocation conflicts
- Includes variable-length sequence support via scheduler
- Optimized for Qwen2.5-VL-3B (16 heads, head_dim=128)

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention v3 (Hopper)](https://arxiv.org/abs/2407.08608)
- [CUTLASS Library](https://github.com/NVIDIA/cutlass)
