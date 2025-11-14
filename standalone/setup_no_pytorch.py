#!/usr/bin/env python3
"""
Setup script for Flash Attention without PyTorch dependency.
Mimics the compilation behavior of the original setup.py but without PyTorch.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get the directory of this script
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CUDA_ARCHITECTURES=90',
        ]

        build_args = ['--config', 'Release', '--', '-j4']

        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Run CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

class NinjaBuild(build_ext):
    """Build using Ninja, similar to PyTorch's approach"""

    def build_extensions(self):
        # Ensure Ninja is available
        try:
            subprocess.check_call(['ninja', '--version'])
        except:
            raise RuntimeError("Ninja build system not found. Install with: pip install ninja")

        # Create build directory
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Collect all Hopper kernel files
        hopper_kernels = glob.glob(str(Path(parent_dir) / "hopper" / "instantiations" / "*.cu"))

        # Filter for the kernels we need (Qwen2.5-VL configuration)
        needed_kernels = [
            "flash_fwd_hdim128_fp16_sm90.cu",
            "flash_fwd_hdim128_bf16_sm90.cu",
            "flash_fwd_hdim128_e4m3_sm90.cu",
        ]

        hopper_kernels = [k for k in hopper_kernels
                         if any(n in k for n in needed_kernels)]

        if not hopper_kernels:
            # Generate kernels if they don't exist
            generate_script = Path(parent_dir) / "hopper" / "generate_kernels.py"
            if generate_script.exists():
                subprocess.check_call([sys.executable, str(generate_script)])
                hopper_kernels = glob.glob(str(Path(parent_dir) / "hopper" / "instantiations" / "*.cu"))
                hopper_kernels = [k for k in hopper_kernels
                                 if any(n in k for n in needed_kernels)]

        # CUDA compilation flags (matching original setup.py)
        nvcc_flags = [
            "-O3",
            "-std=c++17",
            "-gencode", "arch=compute_90a,code=sm_90a",  # Note: using 90a
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90",
            "-DNDEBUG",
        ]

        # Include directories
        include_dirs = [
            str(Path(parent_dir) / "hopper"),
            str(Path(parent_dir) / "csrc" / "flash_attn" / "src"),
            str(Path(parent_dir) / "csrc" / "cutlass" / "include"),
        ]

        # Write ninja build file
        ninja_file = build_dir / "build.ninja"
        with open(ninja_file, 'w') as f:
            # Write rules
            f.write("rule nvcc\n")
            f.write(f"  command = nvcc -c $in -o $out {' '.join(nvcc_flags)} ")
            f.write(f"{' '.join(f'-I{d}' for d in include_dirs)}\n")
            f.write("  description = Compiling $in\n\n")

            f.write("rule link\n")
            f.write("  command = nvcc -shared $in -o $out\n")
            f.write("  description = Linking $out\n\n")

            # Write build statements for each kernel
            obj_files = []
            for kernel in hopper_kernels:
                kernel_name = Path(kernel).stem
                obj_file = f"{kernel_name}.o"
                obj_files.append(obj_file)

                f.write(f"build {obj_file}: nvcc {kernel}\n")

            # API wrapper
            api_src = Path(__file__).parent / "src" / "flash_api.cu"
            if api_src.exists():
                f.write(f"build flash_api.o: nvcc {api_src}\n")
                obj_files.append("flash_api.o")

            # Link all objects into shared library
            f.write(f"build libflash_attn_hopper.so: link {' '.join(obj_files)}\n")
            f.write("default libflash_attn_hopper.so\n")

        # Run ninja
        subprocess.check_call(['ninja', '-C', str(build_dir)])

        # Copy the library to the output directory
        lib_file = build_dir / "libflash_attn_hopper.so"
        if lib_file.exists():
            import shutil
            output_dir = Path(self.build_lib)
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lib_file, output_dir)
            print(f"Built library: {output_dir / 'libflash_attn_hopper.so'}")

def main():
    # Check for CUDA
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(cuda_home):
        raise RuntimeError("CUDA not found. Please set CUDA_HOME environment variable.")

    # Get CUDA version
    try:
        nvcc_output = subprocess.check_output([f"{cuda_home}/bin/nvcc", "-V"],
                                             universal_newlines=True)
        print(f"CUDA info:\n{nvcc_output}")
    except:
        raise RuntimeError("nvcc not found. Please ensure CUDA is properly installed.")

    setup(
        name='flash_attn_hopper_standalone',
        version='0.1.0',
        author='Flash Attention Standalone',
        description='Flash Attention for Hopper without PyTorch',
        ext_modules=[CMakeExtension('flash_attn_hopper')],
        cmdclass={
            'build_ext': NinjaBuild,  # Use Ninja build like the original
        },
        zip_safe=False,
    )

if __name__ == '__main__':
    main()