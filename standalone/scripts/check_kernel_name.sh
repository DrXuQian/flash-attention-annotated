#!/bin/bash
# 检查实际执行的kernel名称
# Usage: ./check_kernel_name.sh <executable>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <executable>"
    echo "Example: $0 ./build/test_fp16_causal_gqa"
    exit 1
fi

EXECUTABLE=$1

echo "============================================"
echo "Profiling kernels from: $EXECUTABLE"
echo "============================================"

# Use cuda-gdb to catch kernel launches
cuda-gdb -batch \
    -ex "set cuda api_failures ignore" \
    -ex "set cuda memcheck off" \
    -ex "break cudaLaunchKernel" \
    -ex "commands
        silent
        printf \"[KERNEL] %s\\n\", (char*)func
        continue
        end" \
    -ex "run" \
    -ex "quit" \
    $EXECUTABLE 2>&1 | grep "\[KERNEL\]"

echo "============================================"
echo "Alternatively, use nsys for cleaner output:"
echo "  nsys profile --stats=true -o profile $EXECUTABLE"
echo "  nsys stats --report cuda_gpu_kern_sum profile.nsys-rep"
echo "============================================"
