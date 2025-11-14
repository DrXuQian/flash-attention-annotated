# Build Instructions for Flash Attention Standalone

## Quick Start (Standard Build)

```bash
cd standalone
mkdir build
cd build
cmake ..
make -j4
```

## Fully Separated Build (No setmaxnreg Warnings)

### Option 1: Replace CMakeLists.txt

```bash
cd standalone
cp CMakeLists_fully_separated.txt CMakeLists.txt
mkdir build
cd build
cmake ..
make -j4
```

### Option 2: Use the automated script

```bash
cd standalone
chmod +x use_fully_separated.sh
./use_fully_separated.sh
cd build_separated
make -j4
```

### Option 3: All-in-one build script

```bash
cd standalone
chmod +x build_fully_separated.sh
./build_fully_separated.sh
```

## Common CMake Errors and Fixes

### Error: "ignoring extra path from command line"

This happens when trying to specify a different CMakeLists file incorrectly.

❌ Wrong:
```bash
cmake ../CMakeLists_fully_separated.txt ..
```

✅ Correct:
```bash
cp CMakeLists_fully_separated.txt CMakeLists.txt
cmake ..
```

### Error: "scale_softmax_log2 member not found"

Make sure you have the latest fixes from the repository:
```bash
git pull
```

The file `src/flash_api.cu` should NOT contain `scale_softmax_log2`.

### Warning: "setmaxnreg ignored"

This is a performance warning, not an error. To eliminate it, use the fully separated build (Option 1, 2, or 3 above).

## Build Outputs

After successful build:

- **Standard build**: Single static library
  - `build/libflash_kernels.a`
  - `build/flash_attention_exec`

- **Fully separated build**: Multiple shared libraries
  - `build_separated/libkernel_fp16_hdim128.so`
  - `build_separated/libkernel_fp16_hdim256.so`
  - `build_separated/libkernel_bf16_hdim128.so`
  - `build_separated/libkernel_bf16_hdim256.so`
  - `build_separated/libkernel_fp8_e4m3_hdim128.so`
  - `build_separated/libflash_api.so`
  - `build_separated/flash_attention_exec`

## Testing

Run the test executable:
```bash
./flash_attention_exec
```

## Clean Build

To start fresh:
```bash
rm -rf build build_separated
# Then follow any of the build options above
```

## Troubleshooting

1. **CUDA not found**: Make sure CUDA 11.8+ is installed and in PATH
2. **Compilation hangs**: Use `-j1` instead of `-j4` for sequential build
3. **Out of memory**: Reduce parallel jobs or build kernels one by one
4. **Architecture mismatch**: Adjust `-gencode arch=compute_90a,code=sm_90a` in CMakeLists.txt