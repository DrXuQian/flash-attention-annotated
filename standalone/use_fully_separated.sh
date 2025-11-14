#!/bin/bash
# Script to use the fully separated CMake configuration

set -e

echo "Setting up fully separated build configuration..."

# Backup original CMakeLists.txt if it exists
if [ -f CMakeLists.txt ]; then
    cp CMakeLists.txt CMakeLists.txt.backup
    echo "✓ Backed up original CMakeLists.txt"
fi

# Copy the fully separated version
cp CMakeLists_fully_separated.txt CMakeLists.txt
echo "✓ Installed fully separated CMakeLists.txt"

# Clean and create build directory
rm -rf build_separated
mkdir build_separated
cd build_separated

echo ""
echo "Configuring CMake..."
cmake ..

echo ""
echo "Configuration complete!"
echo ""
echo "Now you can build with:"
echo "  cd build_separated"
echo "  make -j4"
echo ""
echo "Or use individual targets:"
echo "  make kernel_fp16_hdim128"
echo "  make kernel_fp16_hdim256"
echo "  make flash_api"
echo "  make flash_attention_exec"