# Flash Attention Standalone Compilation Fixes

## Error 1: scale_softmax_log2 member not found
**Error Message:**
```
error: class "Flash_fwd_params" has no member "scale_softmax_log2"
```

**Fix:**
Remove the line that sets `scale_softmax_log2`. This member was removed in newer versions of Flash Attention. The kernel now computes this internally if needed.

```cpp
// Remove this line:
// flash_params.scale_softmax_log2 = scale * M_LOG2E;

// Keep only:
flash_params.scale_softmax = scale;
```

## Error 2: Multiple template instances match
**Error Message:**
```
error: more than one instance of function template "flash::run_mha_fwd_" matches
```

**Fix:**
Update the forward declaration to match the actual template signature:

```cpp
// Correct template parameters:
template<int Arch, typename T, int kHeadDim, int kHeadDimV,
         bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
```

## Warning: setmaxnreg ignored
**Warning Message:**
```
ptxas info: (C7504) Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility across compilation units.
```

**Root Cause:**
Multiple kernels are being linked together, forcing them to share register limits.

**Solutions:**

### Solution 1: Object Libraries (Current CMakeLists.txt)
Each kernel is compiled as an object library, then linked together. This provides partial separation but may still show warnings.

### Solution 2: Shared Libraries (CMakeLists_fully_separated.txt)
Each kernel is compiled as a separate shared library (.so file). This provides complete isolation:

```bash
# Use the fully separated build
cp CMakeLists_fully_separated.txt CMakeLists.txt
mkdir build && cd build
cmake ..
make
```

### Solution 3: Manual Compilation (build_fully_separated.sh)
Use the provided script that builds each kernel separately:

```bash
./build_fully_separated.sh
```

## Quick Start

For a clean build without warnings:

```bash
cd standalone

# Option 1: Standard build (may show setmaxnreg warning)
mkdir build && cd build
cmake ..
make

# Option 2: Fully separated build (no warnings)
./build_fully_separated.sh
```

## Verification

After fixing, verify with:

```bash
# Check for scale_softmax_log2 (should return nothing)
grep -n "scale_softmax_log2" src/flash_api.cu

# Check template declaration
grep -A2 "template.*run_mha_fwd_" src/flash_api.cu

# Test compilation
mkdir test_build && cd test_build
cmake ..
make kernel_fp16_hdim128  # Test one kernel
```

## Summary

The standalone Flash Attention implementation now:
1. ✅ Removes deprecated `scale_softmax_log2` member
2. ✅ Uses correct template signatures
3. ✅ Provides multiple build options to avoid register conflicts
4. ✅ Compiles successfully with CMake
5. ⚠️ May show setmaxnreg warning with standard build (use fully_separated for production)