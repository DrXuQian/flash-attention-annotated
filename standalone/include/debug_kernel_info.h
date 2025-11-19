/******************************************************************************
 * Debug macros to print kernel template parameters
 *
 * Add this at the beginning of run_flash_fwd() to see exactly what kernel
 * is being instantiated.
 ******************************************************************************/

#pragma once
#include <cstdio>
#include <typeinfo>
#include <cxxabi.h>

namespace flash {
namespace debug {

// Demangle C++ type names
inline const char* demangle(const char* name) {
    int status = -1;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        static thread_local char buffer[1024];
        snprintf(buffer, sizeof(buffer), "%s", demangled);
        free(demangled);
        return buffer;
    }
    return name;
}

// Macro to print template parameters at compile time and runtime
#define PRINT_KERNEL_TEMPLATE_PARAMS() \
    do { \
        static bool printed = false; \
        if (!printed) { \
            printed = true; \
            printf("\n"); \
            printf("================================================================================\n"); \
            printf("KERNEL INSTANTIATION: %s\n", __PRETTY_FUNCTION__); \
            printf("================================================================================\n"); \
            printf("Template Parameters:\n"); \
            printf("  Arch:           %d\n", Arch); \
            printf("  kHeadDim:       %d\n", kHeadDim); \
            printf("  kHeadDimV:      %d\n", kHeadDimV); \
            printf("  ClusterM:       %d\n", ClusterM); \
            printf("  Element:        %s\n", flash::debug::demangle(typeid(Element).name())); \
            printf("  ElementOut:     %s\n", flash::debug::demangle(typeid(ElementOut).name())); \
            printf("  Is_causal:      %d\n", (int)Is_causal); \
            printf("  Is_local:       %d\n", (int)Is_local); \
            printf("  Has_softcap:    %d\n", (int)Has_softcap); \
            printf("  Varlen:         %d\n", (int)Varlen); \
            printf("  PagedKVNonTMA:  %d\n", (int)PagedKVNonTMA); \
            printf("  AppendKV:       %d\n", (int)AppendKV); \
            printf("  HasQv:          %d\n", (int)HasQv); \
            printf("  PackGQA:        %d\n", (int)PackGQA); \
            printf("  Split:          %d\n", (int)Split); \
            printf("  V_colmajor:     %d\n", (int)V_colmajor); \
            printf("\n"); \
            printf("Computed Constants:\n"); \
            printf("  kBlockM:        %d\n", kBlockM); \
            printf("  kBlockN:        %d\n", kBlockN); \
            printf("  kStages:        %d\n", kStages); \
            printf("  kNWarps:        %d\n", kNWarps); \
            printf("  MmaPV_is_RS:    %d\n", (int)MmaPV_is_RS); \
            printf("  IntraWGOverlap: %d\n", (int)IntraWGOverlap); \
            printf("  UsePersistent:  %d\n", (int)UsePersistentScheduler); \
            printf("\n"); \
            printf("Runtime Parameters (from Flash_fwd_params):\n"); \
            printf("  is_varlen_q:    %d\n", (int)is_varlen_q); \
            printf("  is_varlen_k:    %d\n", (int)is_varlen_k); \
            printf("  seqlen_q:       %d\n", seqlen_q); \
            printf("  batch_q:        %d\n", batch_q); \
            printf("  batch_k:        %d\n", batch_k); \
            printf("  params.b:       %d\n", params.b); \
            printf("  params.h:       %d\n", params.h); \
            printf("  params.h_k:     %d\n", params.h_k); \
            printf("  params.d:       %d\n", params.d); \
            printf("  params.seqlen_q:%d\n", params.seqlen_q); \
            printf("  params.seqlen_k:%d\n", params.seqlen_k); \
            printf("  params.total_q: %ld\n", (long)params.total_q); \
            printf("  params.total_k: %ld\n", (long)params.total_k); \
            printf("================================================================================\n"); \
            printf("\n"); \
            fflush(stdout); \
        } \
    } while(0)

// Simpler version that just prints the key info
#define PRINT_KERNEL_INFO_SHORT() \
    do { \
        static bool printed = false; \
        if (!printed) { \
            printed = true; \
            printf("[KERNEL] Arch=%d Element=%s hdim=%d/%d causal=%d varlen=%d packgqa=%d split=%d bM=%d bN=%d\n", \
                Arch, \
                (sizeof(Element) == 2 ? "FP16" : sizeof(Element) == 1 ? "FP8" : "?"), \
                kHeadDim, kHeadDimV, \
                (int)Is_causal, (int)Varlen, (int)PackGQA, (int)Split, \
                kBlockM, kBlockN); \
            fflush(stdout); \
        } \
    } while(0)

} // namespace debug
} // namespace flash


// Usage in hopper/flash_fwd_launch_template.h:
//
// Add at the beginning of run_flash_fwd():
//
//   void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
//       #ifdef FLASH_DEBUG_KERNEL_INFO
//       PRINT_KERNEL_TEMPLATE_PARAMS();  // Full details
//       // or
//       PRINT_KERNEL_INFO_SHORT();       // Brief info
//       #endif
//
//       ... rest of function ...
//   }
//
// Then compile with: -DFLASH_DEBUG_KERNEL_INFO
