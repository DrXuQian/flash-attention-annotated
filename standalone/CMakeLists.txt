cmake_minimum_required(VERSION 3.18)
project(flash_attention_standalone LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS "11.8")
    message(FATAL_ERROR "CUDA 11.8+ is required for Hopper (SM90) support")
endif()

# Include directories - use parent directory structure
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_compile_definitions(
    FLASHATTENTION_STANDALONE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASH_NAMESPACE=flash
)

# CUDA flags matching setup.py
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
)

# ============================================================================
# Use official Hopper kernel instantiation files
# ============================================================================

# Kernel files from hopper/instantiations directory
set(HOPPER_KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim64_fp16_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim64_e4m3_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim96_fp16_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim96_e4m3_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim128_e4m3_sm90.cu
    # Note: 192 and 256 are commented out in flash_api.cu to avoid unnecessary linking
    # ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim192_fp16_sm90.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim192_e4m3_sm90.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim256_fp16_sm90.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/instantiations/flash_fwd_hdim256_e4m3_sm90.cu
)

# Check which files exist
set(EXISTING_KERNELS)
foreach(kernel_file ${HOPPER_KERNEL_FILES})
    if(EXISTS ${kernel_file})
        list(APPEND EXISTING_KERNELS ${kernel_file})
        message(STATUS "Found kernel: ${kernel_file}")
    else()
        message(WARNING "Kernel file not found: ${kernel_file}")
    endif()
endforeach()

if(NOT EXISTING_KERNELS)
    message(FATAL_ERROR "No kernel files found in hopper/instantiations/. Run hopper/generate_kernels.py first.")
endif()

# Compile each kernel as a separate object library
set(KERNEL_OBJECTS)
foreach(kernel_file ${EXISTING_KERNELS})
    get_filename_component(kernel_name ${kernel_file} NAME_WE)

    add_library(${kernel_name} OBJECT ${kernel_file})

    target_compile_options(${kernel_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    )

    set_target_properties(${kernel_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    list(APPEND KERNEL_OBJECTS $<TARGET_OBJECTS:${kernel_name}>)
    message(STATUS "Added kernel object: ${kernel_name}")
endforeach()

# ============================================================================
# API wrapper and scheduler - compiled separately WITHOUT kernel instantiations
# ============================================================================

add_library(flash_api OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_api.cu
)

target_compile_options(flash_api PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

set_target_properties(flash_api PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Scheduler for variable-length sequences
add_library(flash_scheduler OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/../hopper/flash_prepare_scheduler.cu
)

target_compile_options(flash_scheduler PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

set_target_properties(flash_scheduler PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Main executable - links all object files together
# ============================================================================

add_executable(flash_attention_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    $<TARGET_OBJECTS:flash_api>
    $<TARGET_OBJECTS:flash_scheduler>
    ${KERNEL_OBJECTS}
)

target_link_libraries(flash_attention_exec PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(flash_attention_exec PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test executable for FP8 varlen with max_seqlen=1680 (3 sequences)
add_executable(test_fp8_varlen_1680
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fp8_varlen_1680.cpp
    $<TARGET_OBJECTS:flash_api>
    $<TARGET_OBJECTS:flash_scheduler>
    ${KERNEL_OBJECTS}
)

target_link_libraries(test_fp8_varlen_1680 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(test_fp8_varlen_1680 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test executable for FP8 varlen with max_seqlen=64
add_executable(test_fp8_varlen_64
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fp8_varlen_64.cpp
    $<TARGET_OBJECTS:flash_api>
    $<TARGET_OBJECTS:flash_scheduler>
    ${KERNEL_OBJECTS}
)

target_link_libraries(test_fp8_varlen_64 PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(test_fp8_varlen_64 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test executable for FP16 causal GQA
add_executable(test_fp16_causal_gqa
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fp16_causal_gqa.cpp
    $<TARGET_OBJECTS:flash_api>
    $<TARGET_OBJECTS:flash_scheduler>
    ${KERNEL_OBJECTS}
)

target_link_libraries(test_fp16_causal_gqa PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(test_fp16_causal_gqa PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test executable for FP16 decode GQA
add_executable(test_fp16_decode_gqa
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fp16_decode_gqa.cpp
    $<TARGET_OBJECTS:flash_api>
    $<TARGET_OBJECTS:flash_scheduler>
    ${KERNEL_OBJECTS}
)

target_link_libraries(test_fp16_decode_gqa PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(test_fp16_decode_gqa PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ============================================================================
# Print configuration
# ============================================================================

message(STATUS "")
message(STATUS "Flash Attention Standalone")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Architecture: SM90a (Hopper)")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - flash_attention_exec: Main test (3 cases)")
message(STATUS "  - test_fp8_varlen_1680: FP8 varlen test (max_seqlen=1680, 3 seqs)")
message(STATUS "  - test_fp8_varlen_64: FP8 varlen test (max_seqlen=64, 84 seqs)")
message(STATUS "  - test_fp16_causal_gqa: FP16 causal GQA prefill (seqlen=1285, 16:2 GQA)")
message(STATUS "  - test_fp16_decode_gqa: FP16 decode GQA (seqlen_q=1, seqlen_k=2048, 16:2 GQA)")
message(STATUS "")