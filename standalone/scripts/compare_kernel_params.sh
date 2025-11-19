#!/bin/bash
# 对比PyTorch和Standalone的kernel参数
# 这个脚本会：
# 1. 临时patch hopper代码添加debug打印
# 2. 重新编译PyTorch和Standalone
# 3. 运行并对比输出
# 4. 恢复原始代码

set -e

FLASH_ROOT=${FLASH_ROOT:-"/home/qianxu/flash-attention"}
STANDALONE_ROOT="$FLASH_ROOT/standalone"

echo "============================================"
echo "Comparing Kernel Parameters"
echo "============================================"

# Check if patch file exists
PATCH_FILE="$STANDALONE_ROOT/scripts/add_debug_print.patch"
if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found: $PATCH_FILE"
    exit 1
fi

# Backup original file
TEMPLATE_FILE="$FLASH_ROOT/hopper/flash_fwd_launch_template.h"
BACKUP_FILE="$TEMPLATE_FILE.backup"

echo "Step 1: Backing up original file..."
cp "$TEMPLATE_FILE" "$BACKUP_FILE"

# Apply patch
echo "Step 2: Applying debug patch..."
cd "$FLASH_ROOT"
if patch -p1 < "$PATCH_FILE"; then
    echo "  ✓ Patch applied successfully"
else
    echo "  ✗ Patch failed, trying reverse patch first..."
    patch -R -p1 < "$PATCH_FILE" || true
    patch -p1 < "$PATCH_FILE"
fi

# Function to restore original file
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -f "$BACKUP_FILE" ]; then
        echo "  Restoring original file..."
        mv "$BACKUP_FILE" "$TEMPLATE_FILE"
        echo "  ✓ Restored"
    fi
}

# Register cleanup on exit
trap cleanup EXIT

# Rebuild standalone
echo ""
echo "Step 3: Rebuilding standalone..."
cd "$STANDALONE_ROOT/build"
make -j$(nproc) 2>&1 | tail -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ✗ Build failed"
    exit 1
fi
echo "  ✓ Build successful"

# Run standalone and capture output
echo ""
echo "Step 4: Running standalone version..."
STANDALONE_OUT=$(mktemp)
./test_fp16_causal_gqa 2>&1 | tee "$STANDALONE_OUT"

# Extract kernel info
echo ""
echo "Step 5: Analyzing standalone kernel parameters..."
STANDALONE_KERNEL=$(mktemp)
grep -A 20 "KERNEL INSTANTIATION" "$STANDALONE_OUT" > "$STANDALONE_KERNEL" || true
grep -A 10 "RUNTIME PARAMS" "$STANDALONE_OUT" >> "$STANDALONE_KERNEL" || true

# Optional: Do the same for PyTorch if test script provided
if [ -n "$1" ]; then
    PYTORCH_TEST="$1"
    echo ""
    echo "Step 6: Running PyTorch version: $PYTORCH_TEST"

    # Rebuild PyTorch extension
    cd "$FLASH_ROOT"
    echo "  Rebuilding PyTorch extension..."
    python setup.py build_ext --inplace 2>&1 | tail -20

    PYTORCH_OUT=$(mktemp)
    python "$PYTORCH_TEST" 2>&1 | tee "$PYTORCH_OUT"

    echo ""
    echo "Step 7: Analyzing PyTorch kernel parameters..."
    PYTORCH_KERNEL=$(mktemp)
    grep -A 20 "KERNEL INSTANTIATION" "$PYTORCH_OUT" > "$PYTORCH_KERNEL" || true
    grep -A 10 "RUNTIME PARAMS" "$PYTORCH_OUT" >> "$PYTORCH_KERNEL" || true

    # Compare
    echo ""
    echo "============================================"
    echo "COMPARISON"
    echo "============================================"
    echo ""
    echo "=== PYTORCH ==="
    cat "$PYTORCH_KERNEL"
    echo ""
    echo "=== STANDALONE ==="
    cat "$STANDALONE_KERNEL"
    echo ""

    # Show diff
    echo "============================================"
    echo "DIFF (if any):"
    echo "============================================"
    diff -u "$PYTORCH_KERNEL" "$STANDALONE_KERNEL" || true

    # Cleanup temp files
    rm -f "$PYTORCH_OUT" "$PYTORCH_KERNEL"
else
    echo ""
    echo "============================================"
    echo "STANDALONE KERNEL INFO"
    echo "============================================"
    cat "$STANDALONE_KERNEL"
    echo ""
    echo "To compare with PyTorch, run:"
    echo "  $0 <pytorch_test_script.py>"
fi

rm -f "$STANDALONE_OUT" "$STANDALONE_KERNEL"

echo ""
echo "Done!"
