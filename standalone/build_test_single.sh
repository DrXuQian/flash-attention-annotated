#!/bin/bash
# Test building each kernel individually to identify which one has register issues

set -e

echo "Testing individual kernel compilation..."
echo "======================================="

# Create test directory
rm -rf test_build
mkdir test_build
cd test_build

# Test each kernel one by one
kernels=(
    "kernel_fp16_hdim128"
    "kernel_fp16_hdim256"
    "kernel_bf16_hdim128"
    "kernel_bf16_hdim256"
    "kernel_fp8_e4m3_hdim128"
)

for kernel in "${kernels[@]}"; do
    echo ""
    echo "Testing $kernel..."
    echo "-----------------"

    # Create a minimal CMakeLists.txt for this kernel only
    cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.18)
project(test_${kernel} CUDA)

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

include_directories(
    \${CMAKE_CURRENT_SOURCE_DIR}/../include
    \${CMAKE_CURRENT_SOURCE_DIR}/../../
    \${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/flash_attn/src
    \${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/cutlass/include
    \${CMAKE_CURRENT_SOURCE_DIR}/../../hopper
)

add_compile_definitions(
    FLASHATTENTION_STANDALONE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASH_NAMESPACE=flash
)

set(CUDA_FLAGS
    -O3
    -std=c++17
    -gencode arch=compute_90a,code=sm_90a
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
    -Xcompiler=-fPIC
    # Try with explicit register limit
    -maxrregcount=255
)

add_library(${kernel} OBJECT ../kernels/${kernel}.cu)

target_compile_options(${kernel} PRIVATE
    \$<\$<COMPILE_LANGUAGE:CUDA>:\${CUDA_FLAGS}>
)

set_target_properties(${kernel} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
EOF

    # Try to configure and build
    if cmake . > /dev/null 2>&1; then
        echo "  ✓ CMake configuration successful"

        # Try to compile with verbose output to see register usage
        if make VERBOSE=1 2>&1 | grep -E "(error|warning.*register|ptxas.*error)" | head -5; then
            echo "  ✗ Compilation failed or has warnings"
        else
            echo "  ✓ Compilation successful"

            # Check actual register usage
            if [ -f "CMakeFiles/${kernel}.dir/kernels/${kernel}.cu.o" ]; then
                echo "  Register usage:"
                cuobjdump --dump-resource-usage "CMakeFiles/${kernel}.dir/kernels/${kernel}.cu.o" 2>/dev/null | grep -E "registers|smem" | head -3 || true
            fi
        fi
    else
        echo "  ✗ CMake configuration failed"
    fi

    # Clean for next test
    rm -rf CMakeFiles CMakeCache.txt
done

echo ""
echo "======================================="
echo "Test complete. Check results above."