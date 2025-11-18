#!/bin/bash
# Build Flash Attention standalone for Hopper architecture
# Uses official kernel files from hopper/instantiations/

set -e

echo "========================================"
echo "Flash Attention Standalone Build"
echo "========================================"
echo ""

# Check if hopper kernel files exist
HOPPER_DIR="../hopper/instantiations"
if [ ! -d "$HOPPER_DIR" ]; then
    echo "Error: hopper/instantiations directory not found"
    echo "Generating kernel files..."
    cd ../hopper
    python3 generate_kernels.py
    cd ../standalone
fi

# Check for specific kernel files we need
REQUIRED_KERNELS=(
    "flash_fwd_hdim64_fp16_sm90.cu"
    "flash_fwd_hdim64_e4m3_sm90.cu"
    "flash_fwd_hdim96_fp16_sm90.cu"
    "flash_fwd_hdim96_e4m3_sm90.cu"
    "flash_fwd_hdim128_fp16_sm90.cu"
    "flash_fwd_hdim128_e4m3_sm90.cu"
    "flash_fwd_hdim192_fp16_sm90.cu"
    "flash_fwd_hdim256_fp16_sm90.cu"
)

echo "Checking for required kernel files..."
MISSING_COUNT=0
for kernel in "${REQUIRED_KERNELS[@]}"; do
    if [ -f "$HOPPER_DIR/$kernel" ]; then
        echo "  ✓ $kernel"
    else
        echo "  ✗ $kernel (missing)"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo ""
    echo "Generating missing kernel files..."
    cd ../hopper
    python3 generate_kernels.py
    cd ../standalone
fi

# Clean and create build directory
echo ""
echo "Setting up build directory..."
rm -rf build
mkdir build
cd build

echo ""
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      ..

echo ""
echo "Building (this may take a while)..."
make -j4

echo ""
echo "========================================"
echo "Build Results:"
echo "========================================"

# Check all executables
EXECUTABLES=(
    "flash_attention_exec"
    "test_fp8_varlen_1680"
    "test_fp8_varlen_64"
    "test_fp16_causal_gqa"
    "test_fp16_decode_gqa"
)

BUILD_SUCCESS=true
for exe in "${EXECUTABLES[@]}"; do
    if [ -f "$exe" ]; then
        echo "  ✓ $exe"
    else
        echo "  ✗ $exe (missing)"
        BUILD_SUCCESS=false
    fi
done

echo ""
if [ "$BUILD_SUCCESS" = true ]; then
    echo "✓ All executables built successfully!"
    echo ""
    echo "Build directory: $(pwd)"
else
    echo "✗ Some executables failed to build"
    exit 1
fi
echo "========================================"