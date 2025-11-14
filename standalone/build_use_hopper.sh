#!/bin/bash
# Build using official Hopper kernel files from hopper/instantiations/
# This is the correct approach that matches python setup.py

set -e

echo "========================================"
echo "Flash Attention Build (Official Kernels)"
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
rm -rf build_hopper
mkdir build_hopper
cd build_hopper

# Use the Hopper-based CMakeLists
cp ../CMakeLists_use_hopper.txt ../CMakeLists.txt.tmp
mv ../CMakeLists.txt ../CMakeLists.txt.backup 2>/dev/null || true
mv ../CMakeLists.txt.tmp ../CMakeLists.txt

echo ""
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      ..

# Restore original CMakeLists
mv ../CMakeLists.txt ../CMakeLists_use_hopper.used
mv ../CMakeLists.txt.backup ../CMakeLists.txt 2>/dev/null || true

echo ""
echo "Building (this may take a while)..."
make -j4

echo ""
echo "========================================"
if [ -f "flash_attention_exec" ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Executable: $(pwd)/flash_attention_exec"
    echo ""
    echo "This build uses official Hopper kernel files,"
    echo "matching the behavior of python setup.py install"
else
    echo "✗ Build failed"
    exit 1
fi
echo "========================================"