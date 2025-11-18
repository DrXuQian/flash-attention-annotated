#!/bin/bash

# Build and Test Script for Flash Attention V2 API
# This script builds and runs the FP8 varlen test matching PyTorch testcase

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  Flash Attention V2 API - Build and Test"
echo "================================================================"
echo ""

# ============================================================================
# BUILD
# ============================================================================

echo "Step 1: Creating build directory..."
mkdir -p build_v2
cd build_v2

echo ""
echo "Step 2: Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_v2.txt

echo ""
echo "Step 3: Building executables..."
echo "  (This may take several minutes...)"
cmake --build . --target test_fp8_varlen -j$(nproc)

echo ""
echo "✓ Build completed successfully!"
echo ""

# ============================================================================
# RUN TESTS
# ============================================================================

echo "================================================================"
echo "  Running Test: FP8 Varlen (PyTorch matching)"
echo "================================================================"
echo ""

if [ -f ./test_fp8_varlen ]; then
    ./test_fp8_varlen
    TEST_RESULT=$?
    echo ""
    if [ $TEST_RESULT -eq 0 ]; then
        echo "================================================================"
        echo "  ✓ ALL TESTS PASSED"
        echo "================================================================"
    else
        echo "================================================================"
        echo "  ✗ TEST FAILED (exit code: $TEST_RESULT)"
        echo "================================================================"
        exit $TEST_RESULT
    fi
else
    echo "Error: test_fp8_varlen executable not found!"
    exit 1
fi

echo ""
echo "To run the comprehensive test suite (3 cases):"
echo "  ./build_v2/flash_attention_test_v2"
echo ""
