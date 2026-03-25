#pragma once
/**
 * nvtx_ranges.cuh
 *
 * NVTX (NVIDIA Tools Extension) annotation helpers for FHE profiling.
 *
 * NVTX markers appear as labeled, colored bands in Nsight Systems timeline.
 * This makes it trivial to identify FHE regions without guessing from kernel names.
 *
 * Usage:
 *   #include "nvtx_ranges.cuh"
 *
 *   {
 *       NvtxRange r("KeySwitch");          // auto-pops when r goes out of scope
 *       keyswitching_input_broadcast(...);
 *   }
 *
 *   // Or manually:
 *   NVTX_PUSH("Bootstrap-CtoS");
 *   coefftoslot_3(...);
 *   NVTX_POP();
 *
 * Colors: each FHE operation gets a distinct color in the timeline.
 */

#ifdef USE_NVTX
#  include <nvtx3/nvToolsExt.h>
#else
// No-op stubs when NVTX is not available (CPU-only builds, CI)
namespace nvtxDetail { inline void push(const char*) {} inline void pop() {} }
#endif

// ---------------------------------------------------------------------------
// Color palette (ARGB format)
// ---------------------------------------------------------------------------
namespace fhe_nvtx {
    static constexpr uint32_t COLOR_NTT          = 0xFF4472C4;  // blue
    static constexpr uint32_t COLOR_KEYSWITCH     = 0xFFED7D31;  // orange
    static constexpr uint32_t COLOR_BOOTSTRAP     = 0xFF70AD47;  // green
    static constexpr uint32_t COLOR_MATMUL        = 0xFFFFC000;  // yellow
    static constexpr uint32_t COLOR_GELU          = 0xFF9E480E;  // dark orange
    static constexpr uint32_t COLOR_SOFTMAX       = 0xFF833C11;  // brown
    static constexpr uint32_t COLOR_LAYERNORM     = 0xFF264478;  // dark blue
    static constexpr uint32_t COLOR_NCCL          = 0xFFFF0000;  // red (comm = hot)
    static constexpr uint32_t COLOR_ENCODE        = 0xFF7030A0;  // purple
    static constexpr uint32_t COLOR_MODSWITCH     = 0xFF00B050;  // bright green
}

// ---------------------------------------------------------------------------
// Low-level push/pop
// ---------------------------------------------------------------------------

inline void nvtx_push(const char *name, uint32_t color = 0xFF888888) {
#ifdef USE_NVTX
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
#else
    (void)name; (void)color;
#endif
}

inline void nvtx_pop() {
#ifdef USE_NVTX
    nvtxRangePop();
#endif
}

// ---------------------------------------------------------------------------
// RAII scoped range
// ---------------------------------------------------------------------------

struct NvtxRange {
    explicit NvtxRange(const char *name, uint32_t color = 0xFF888888) {
        nvtx_push(name, color);
    }
    ~NvtxRange() { nvtx_pop(); }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

// ---------------------------------------------------------------------------
// Convenience macros for named FHE regions
// ---------------------------------------------------------------------------

#define NVTX_NTT(label)        NvtxRange _nvtx_ntt(label,         fhe_nvtx::COLOR_NTT)
#define NVTX_KEYSWITCH(label)  NvtxRange _nvtx_ks(label,          fhe_nvtx::COLOR_KEYSWITCH)
#define NVTX_BOOTSTRAP(label)  NvtxRange _nvtx_bs(label,          fhe_nvtx::COLOR_BOOTSTRAP)
#define NVTX_MATMUL(label)     NvtxRange _nvtx_mm(label,          fhe_nvtx::COLOR_MATMUL)
#define NVTX_GELU(label)       NvtxRange _nvtx_gelu(label,        fhe_nvtx::COLOR_GELU)
#define NVTX_SOFTMAX(label)    NvtxRange _nvtx_sm(label,          fhe_nvtx::COLOR_SOFTMAX)
#define NVTX_LAYERNORM(label)  NvtxRange _nvtx_ln(label,          fhe_nvtx::COLOR_LAYERNORM)
#define NVTX_NCCL(label)       NvtxRange _nvtx_nccl(label,        fhe_nvtx::COLOR_NCCL)
#define NVTX_ENCODE(label)     NvtxRange _nvtx_enc(label,         fhe_nvtx::COLOR_ENCODE)
#define NVTX_MODSWITCH(label)  NvtxRange _nvtx_ms(label,          fhe_nvtx::COLOR_MODSWITCH)

// Generic push/pop macros (still useful for one-liners)
#define NVTX_PUSH(name)  nvtx_push(name)
#define NVTX_POP()       nvtx_pop()
