#!/usr/bin/env python3
"""
Generate separated kernel instantiation files for Flash Attention.
Each kernel configuration gets its own compilation unit to avoid register conflicts.
"""

import os
import argparse
from pathlib import Path

# Template for kernel instantiation files
KERNEL_TEMPLATE = """// Auto-generated kernel instantiation file
// Configuration: dtype={dtype_name}, hdim={hdim}
// This file is compiled separately to avoid register limit conflicts

#include "../hopper/flash_fwd_launch_template.h"

// Single kernel instantiation per file
// Template parameters: Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA
template void run_mha_fwd_<{sm_arch}, {dtype}, {hdim}, {hdim}, false, false, false, false>
    (Flash_fwd_params &params, cudaStream_t stream);
"""

def generate_kernel_file(output_dir, dtype_name, dtype_cutlass, hdim, sm_arch=90):
    """Generate a single kernel instantiation file."""

    filename = f"kernel_{dtype_name}_hdim{hdim}.cu"
    filepath = os.path.join(output_dir, filename)

    content = KERNEL_TEMPLATE.format(
        dtype_name=dtype_name,
        dtype=dtype_cutlass,
        hdim=hdim,
        sm_arch=sm_arch
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Generated: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate Flash Attention kernel files')
    parser.add_argument('--output-dir', default='kernels', help='Output directory for kernel files')
    parser.add_argument('--sm-arch', type=int, default=90, help='SM architecture (90 for Hopper)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Kernel configurations for Qwen2.5-VL-3B
    # Only FP16 and FP8, no BF16
    configs = [
        # FP16 configurations
        ('fp16', 'cutlass::half_t', 128),      # Primary config for Qwen2.5-VL
        ('fp16', 'cutlass::half_t', 256),      # Alternative config

        # FP8 E4M3 configurations (for inference)
        ('fp8_e4m3', 'cutlass::float_e4m3_t', 128),
    ]

    generated_files = []

    print(f"Generating kernel files in {args.output_dir}/")
    print("=" * 60)

    for dtype_name, dtype_cutlass, hdim in configs:
        filename = generate_kernel_file(
            args.output_dir,
            dtype_name,
            dtype_cutlass,
            hdim,
            args.sm_arch
        )
        generated_files.append(filename)

    print("=" * 60)
    print(f"Generated {len(generated_files)} kernel files")

    # Generate CMakeLists snippet
    cmake_snippet = "\n".join([
        f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/kernels/{f}"
        for f in generated_files
    ])

    print("\nCMakeLists.txt snippet for kernel sources:")
    print("-" * 40)
    print(cmake_snippet)

    return generated_files

if __name__ == '__main__':
    main()