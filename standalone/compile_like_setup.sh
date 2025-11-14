#!/bin/bash
# Compile Flash Attention for Hopper exactly like setup.py does
# Key insight: setup.py uses existing kernel files from hopper/instantiations/

set -e

echo "========================================"
echo "Flash Attention Hopper Compilation"
echo "Mimicking setup.py behavior"
echo "========================================"
echo ""

# Check if we're in the standalone directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Must run from standalone directory"
    exit 1
fi

# Go to parent directory (flash-attention root)
cd ..

# First, generate Hopper kernel instantiation files if needed
if [ ! -d "hopper/instantiations" ]; then
    echo "Generating Hopper kernel instantiation files..."
    cd hopper
    python3 generate_kernels.py
    cd ..
fi

# Back to standalone
cd standalone

# Clean build directory
rm -rf build_like_setup
mkdir build_like_setup
cd build_like_setup

echo "Collecting Hopper kernel files..."
# List of kernel files we need (for Qwen2.5-VL: hdim=128)
KERNEL_FILES=(
    "../../hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu"
    "../../hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu"
    "../../hopper/instantiations/flash_fwd_hdim128_e4m3_sm90.cu"
    "../../hopper/instantiations/flash_fwd_hdim128_fp16_split_sm90.cu"
    "../../hopper/instantiations/flash_fwd_hdim128_bf16_split_sm90.cu"
    "../../hopper/instantiations/flash_fwd_hdim128_e4m3_split_sm90.cu"
)

# Check which files exist
EXISTING_FILES=()
for file in "${KERNEL_FILES[@]}"; do
    if [ -f "$file" ]; then
        EXISTING_FILES+=("$file")
        echo "  ✓ Found: $(basename $file)"
    else
        echo "  ✗ Missing: $(basename $file)"
    fi
done

if [ ${#EXISTING_FILES[@]} -eq 0 ]; then
    echo ""
    echo "No kernel files found. Generating minimal set..."

    # Create minimal kernel files
    mkdir -p kernels

    cat > kernels/flash_fwd_hdim128_fp16_sm90.cu << 'EOF'
#include "flash_fwd_launch_template.h"

// Minimal instantiation for hdim=128, fp16
template void run_mha_fwd_<90, cutlass::half_t, 128, 128, false, false, false, false>(
    Flash_fwd_params &params, cudaStream_t stream);
EOF

    EXISTING_FILES=("kernels/flash_fwd_hdim128_fp16_sm90.cu")
fi

echo ""
echo "Compiling ${#EXISTING_FILES[@]} kernel files..."

# CUDA compilation flags (exact match with setup.py)
NVCC_FLAGS=(
    -O3
    -std=c++17
    -gencode arch=compute_90a,code=sm_90a  # Note: using 90a not 90
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DCUTLASS_ENABLE_GDC_FOR_SM90
    -DNDEBUG
    -DFLASHATTENTION_DISABLE_BACKWARD
    -Xcompiler=-fPIC
)

# Include paths
INCLUDE_PATHS=(
    -I../../hopper
    -I../../csrc/flash_attn/src
    -I../../csrc/cutlass/include
    -I../../
)

# Compile each kernel separately (key to avoiding register issues!)
echo ""
for kernel_file in "${EXISTING_FILES[@]}"; do
    basename=$(basename "$kernel_file" .cu)
    echo "Compiling $basename..."

    # Compile to object file
    nvcc "${NVCC_FLAGS[@]}" "${INCLUDE_PATHS[@]}" \
         -dc "$kernel_file" \
         -o "${basename}.o" 2>&1 | \
         grep -E "(error|warning.*register|ptxas.*error|setmaxnreg)" || true

    if [ -f "${basename}.o" ]; then
        echo "  ✓ Compiled successfully"
    else
        echo "  ✗ Compilation failed"
    fi
done

# Compile API wrapper if it exists
if [ -f "../src/flash_api.cu" ]; then
    echo ""
    echo "Compiling API wrapper..."
    nvcc "${NVCC_FLAGS[@]}" "${INCLUDE_PATHS[@]}" \
         -dc "../src/flash_api.cu" \
         -o "flash_api.o"
    echo "  ✓ API wrapper compiled"
fi

# Link all objects into a shared library
echo ""
echo "Linking shared library..."
nvcc -shared *.o -o libflash_attn_hopper.so

if [ -f "libflash_attn_hopper.so" ]; then
    echo "  ✓ Library created: $(pwd)/libflash_attn_hopper.so"

    # Check for setmaxnreg in the binary
    echo ""
    echo "Checking for setmaxnreg usage:"
    cuobjdump -sass libflash_attn_hopper.so 2>/dev/null | grep -i setmaxnreg | head -3 || echo "  No setmaxnreg instructions found"
else
    echo "  ✗ Linking failed"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo ""
echo "Key differences from standard CMake approach:"
echo "1. Each kernel compiled separately (like setup.py with ninja)"
echo "2. Uses arch=compute_90a,code=sm_90a (not just sm_90)"
echo "3. Kernel files from hopper/instantiations/ directory"
echo "4. No static linking - all dynamic"
echo "========================================"