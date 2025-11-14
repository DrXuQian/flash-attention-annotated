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
    "flash_fwd_hdim128_fp16_sm90.cu"
    "flash_fwd_hdim128_e4m3_sm90.cu"
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
if [ -f "flash_attention_exec" ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Executable: $(pwd)/flash_attention_exec"
else
    echo "✗ Build failed"
    exit 1
fi
echo "========================================"