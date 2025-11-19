#!/bin/bash
# 直接对比PyTorch和Standalone实际执行的kernel
#
# 这个脚本会：
# 1. 运行PyTorch版本并记录kernel
# 2. 运行Standalone版本并记录kernel
# 3. 对比两者的差异

set -e

PYTORCH_TEST=${1:-"test_pytorch.py"}
STANDALONE_TEST=${2:-"./build/test_fp16_causal_gqa"}

echo "==============================================="
echo "Comparing kernels between:"
echo "  PyTorch:    $PYTORCH_TEST"
echo "  Standalone: $STANDALONE_TEST"
echo "==============================================="

# Create temp directory
TMPDIR=$(mktemp -d)
echo "Using temp directory: $TMPDIR"

# Profile PyTorch
echo ""
echo "[1/4] Profiling PyTorch version..."
nsys profile \
    --trace=cuda \
    --stats=true \
    --force-overwrite true \
    -o "$TMPDIR/pytorch" \
    python $PYTORCH_TEST > /dev/null 2>&1

echo "[2/4] Extracting PyTorch kernels..."
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    "$TMPDIR/pytorch.nsys-rep" | \
    grep -E "^[0-9]" | \
    awk -F',' '{print $NF}' | \
    sed 's/"//g' | \
    sort > "$TMPDIR/pytorch_kernels.txt"

PYTORCH_COUNT=$(wc -l < "$TMPDIR/pytorch_kernels.txt")
echo "  Found $PYTORCH_COUNT unique kernels"

# Profile Standalone
echo ""
echo "[3/4] Profiling Standalone version..."
nsys profile \
    --trace=cuda \
    --stats=true \
    --force-overwrite true \
    -o "$TMPDIR/standalone" \
    $STANDALONE_TEST > /dev/null 2>&1

echo "[4/4] Extracting Standalone kernels..."
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    "$TMPDIR/standalone.nsys-rep" | \
    grep -E "^[0-9]" | \
    awk -F',' '{print $NF}' | \
    sed 's/"//g' | \
    sort > "$TMPDIR/standalone_kernels.txt"

STANDALONE_COUNT=$(wc -l < "$TMPDIR/standalone_kernels.txt")
echo "  Found $STANDALONE_COUNT unique kernels"

# Compare
echo ""
echo "==============================================="
echo "COMPARISON RESULTS"
echo "==============================================="

echo ""
echo "🔵 Kernels ONLY in PyTorch:"
comm -23 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt"

echo ""
echo "🔵 Kernels ONLY in Standalone:"
comm -13 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt"

echo ""
echo "✅ Common kernels:"
comm -12 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt"

echo ""
echo "==============================================="
echo "SUMMARY"
echo "==============================================="
COMMON=$(comm -12 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt" | wc -l)
PYTORCH_ONLY=$(comm -23 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt" | wc -l)
STANDALONE_ONLY=$(comm -13 "$TMPDIR/pytorch_kernels.txt" "$TMPDIR/standalone_kernels.txt" | wc -l)

echo "  PyTorch kernels:      $PYTORCH_COUNT"
echo "  Standalone kernels:   $STANDALONE_COUNT"
echo "  Common kernels:       $COMMON"
echo "  PyTorch-only:         $PYTORCH_ONLY"
echo "  Standalone-only:      $STANDALONE_ONLY"

if [ $COMMON -eq $PYTORCH_COUNT ] && [ $COMMON -eq $STANDALONE_COUNT ]; then
    echo ""
    echo "✅ PERFECT MATCH! All kernels are identical."
    echo "   Performance difference must be from:"
    echo "   - Compilation flags"
    echo "   - GPU state (clocks, temperature)"
    echo "   - Measurement overhead"
else
    echo ""
    echo "❌ MISMATCH DETECTED!"
    echo "   Check the kernel lists above to see what's different."
fi

echo ""
echo "Detailed reports saved to:"
echo "  $TMPDIR/pytorch.nsys-rep"
echo "  $TMPDIR/standalone.nsys-rep"
echo ""
echo "To view detailed timeline:"
echo "  nsys-ui $TMPDIR/pytorch.nsys-rep"
echo "  nsys-ui $TMPDIR/standalone.nsys-rep"
echo ""

# Don't cleanup so user can inspect
echo "Temp files kept at: $TMPDIR"
