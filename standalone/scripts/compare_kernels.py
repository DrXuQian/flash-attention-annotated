#!/usr/bin/env python3
"""
Compare CUDA kernels executed by PyTorch vs Standalone
Usage:
  1. Profile PyTorch version: nsys profile -o pytorch python test_pytorch.py
  2. Profile Standalone version: nsys profile -o standalone ./test_standalone
  3. Run this script: python compare_kernels.py pytorch.nsys-rep standalone.nsys-rep
"""

import sys
import subprocess
import re
from collections import defaultdict

def extract_kernels(nsys_file):
    """Extract kernel names and stats from nsys report"""
    # Use nsys stats to get kernel information
    cmd = ['nsys', 'stats', '--report', 'cuda_gpu_kern_sum', nsys_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running nsys stats: {e}")
        return {}

    kernels = {}
    in_table = False

    for line in output.split('\n'):
        line = line.strip()

        # Look for the table start
        if 'Time (%)' in line and 'Total Time' in line:
            in_table = True
            continue

        if in_table and line and not line.startswith('-'):
            # Parse kernel line
            parts = line.split()
            if len(parts) >= 6:
                # Format: Time(%) Total_Time Instances Avg Min Max Name
                try:
                    time_pct = float(parts[0])
                    total_time = parts[1]
                    instances = int(parts[2])
                    avg_time = parts[3]
                    kernel_name = ' '.join(parts[6:])  # Rest is kernel name

                    # Clean up kernel name (remove template details for matching)
                    clean_name = re.sub(r'<.*?>', '<...>', kernel_name)

                    kernels[clean_name] = {
                        'name': kernel_name,
                        'time_pct': time_pct,
                        'total_time': total_time,
                        'instances': instances,
                        'avg_time': avg_time
                    }
                except (ValueError, IndexError):
                    continue

    return kernels

def compare_kernel_lists(pytorch_kernels, standalone_kernels):
    """Compare two kernel lists and report differences"""

    print("=" * 80)
    print("KERNEL COMPARISON")
    print("=" * 80)

    pytorch_names = set(pytorch_kernels.keys())
    standalone_names = set(standalone_kernels.keys())

    # Kernels only in PyTorch
    only_pytorch = pytorch_names - standalone_names
    if only_pytorch:
        print("\n🔴 Kernels ONLY in PyTorch version:")
        for name in sorted(only_pytorch):
            k = pytorch_kernels[name]
            print(f"  - {k['name']}")
            print(f"    Time: {k['time_pct']}% ({k['total_time']}), Instances: {k['instances']}")

    # Kernels only in Standalone
    only_standalone = standalone_names - pytorch_names
    if only_standalone:
        print("\n🔴 Kernels ONLY in Standalone version:")
        for name in sorted(only_standalone):
            k = standalone_kernels[name]
            print(f"  - {k['name']}")
            print(f"    Time: {k['time_pct']}% ({k['total_time']}), Instances: {k['instances']}")

    # Common kernels
    common = pytorch_names & standalone_names
    if common:
        print("\n✅ Common kernels:")
        for name in sorted(common):
            pt = pytorch_kernels[name]
            sa = standalone_kernels[name]
            print(f"\n  Kernel: {name}")
            print(f"    PyTorch:    {pt['time_pct']:6.2f}% | {pt['instances']:4d} calls | {pt['avg_time']:>10s} avg")
            print(f"    Standalone: {sa['time_pct']:6.2f}% | {sa['instances']:4d} calls | {sa['avg_time']:>10s} avg")

            # Highlight differences
            if abs(pt['time_pct'] - sa['time_pct']) > 1.0:
                print(f"    ⚠️  TIME DIFFERENCE: {abs(pt['time_pct'] - sa['time_pct']):.2f}%")
            if pt['instances'] != sa['instances']:
                print(f"    ⚠️  CALL COUNT DIFFERENCE: {pt['instances']} vs {sa['instances']}")

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  PyTorch kernels:    {len(pytorch_names)}")
    print(f"  Standalone kernels: {len(standalone_names)}")
    print(f"  Common kernels:     {len(common)}")
    print(f"  PyTorch-only:       {len(only_pytorch)}")
    print(f"  Standalone-only:    {len(only_standalone)}")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    pytorch_file = sys.argv[1]
    standalone_file = sys.argv[2]

    print(f"Analyzing PyTorch kernels from: {pytorch_file}")
    pytorch_kernels = extract_kernels(pytorch_file)
    print(f"  Found {len(pytorch_kernels)} kernels\n")

    print(f"Analyzing Standalone kernels from: {standalone_file}")
    standalone_kernels = extract_kernels(standalone_file)
    print(f"  Found {len(standalone_kernels)} kernels\n")

    compare_kernel_lists(pytorch_kernels, standalone_kernels)
