#!/bin/bash
# Build script with fully separated compilation to avoid setmaxnreg warnings
# Each kernel is compiled into its own shared library

set -e

echo "================================================"
echo "Flash Attention Fully Separated Build"
echo "================================================"
echo ""

# Clean previous build
rm -rf build_separated
mkdir build_separated
cd build_separated

# Use the fully separated CMakeLists
echo "Configuring with complete kernel isolation..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      ../CMakeLists_fully_separated.txt ..

echo ""
echo "Building kernel libraries separately..."
echo ""

# Build each kernel as a separate target
# This ensures they never share compilation units
for kernel in kernel_fp16_hdim128 kernel_fp16_hdim256 kernel_bf16_hdim128 kernel_bf16_hdim256 kernel_fp8_e4m3_hdim128; do
    echo "Building $kernel..."
    make $kernel -j1  # Use -j1 to ensure sequential build
    echo "✓ $kernel built successfully"
    echo ""
done

echo "Building API library..."
make flash_api -j1
echo "✓ API library built"
echo ""

echo "Building executable..."
make flash_attention_exec -j1
echo "✓ Executable built"
echo ""

echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo ""
echo "Executables and libraries:"
ls -lh *.so flash_attention_exec 2>/dev/null || true
echo ""
echo "To run:"
echo "  cd build_separated"
echo "  ./flash_attention_exec"
echo ""
echo "Note: Each kernel is in its own .so file,"
echo "completely eliminating setmaxnreg conflicts."