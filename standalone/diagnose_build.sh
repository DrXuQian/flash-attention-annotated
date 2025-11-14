#!/bin/bash
# Diagnose compilation issues

echo "========================================"
echo "Flash Attention Build Diagnostics"
echo "========================================"
echo ""

# Check 1: Hopper kernel files
echo "1. Checking Hopper kernel files..."
HOPPER_DIR="../hopper/instantiations"
if [ -d "$HOPPER_DIR" ]; then
    echo "  ✓ Directory exists"
    echo "  FP16 kernels:"
    ls -1 "$HOPPER_DIR"/flash_fwd_hdim128*fp16*.cu 2>/dev/null || echo "    (none found)"
    echo "  FP8 kernels:"
    ls -1 "$HOPPER_DIR"/flash_fwd_hdim128*e4m3*.cu 2>/dev/null || echo "    (none found)"
else
    echo "  ✗ Directory not found"
fi

echo ""
echo "2. Checking flash_api.cu includes..."
echo "  Includes in flash_api.cu:"
grep "^#include" src/flash_api.cu | head -10

echo ""
echo "3. Checking for template definitions in flash_api.cu..."
if grep -q "flash_fwd_launch_template.h" src/flash_api.cu; then
    echo "  ✗ WARNING: flash_api.cu includes flash_fwd_launch_template.h"
    echo "  This will cause 'more than one instance' errors!"
else
    echo "  ✓ Good: flash_api.cu does not include template definitions"
fi

echo ""
echo "4. Test compile flash_api.cu alone..."
cd src
nvcc -c flash_api.cu \
    -I../../hopper \
    -I../../csrc/flash_attn/src \
    -I../../csrc/cutlass/include \
    -I../include \
    -O3 -std=c++17 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -DFLASHATTENTION_STANDALONE \
    -DFLASHATTENTION_DISABLE_BACKWARD \
    -o /tmp/flash_api_test.o 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✓ flash_api.cu compiles successfully"
    rm -f /tmp/flash_api_test.o
else
    echo "  ✗ flash_api.cu has compilation errors"
fi

cd ..

echo ""
echo "5. Check for kernel instantiations in standalone/kernels..."
if [ -d "kernels" ]; then
    echo "  Files in kernels/:"
    ls -1 kernels/*.cu 2>/dev/null
    echo ""
    echo "  Checking if they include template header:"
    for f in kernels/*.cu; do
        if [ -f "$f" ]; then
            echo "  $f:"
            grep "#include.*flash_fwd_launch_template" "$f" || echo "    (no template include)"
        fi
    done
else
    echo "  (no kernels directory)"
fi

echo ""
echo "========================================"
echo "Diagnosis complete"
echo "========================================"