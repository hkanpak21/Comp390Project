/**
 * nvtx_tracer.cuh
 *
 * Lightweight NVTX (NVIDIA Tools Extension) wrappers for timeline annotation.
 * Ranges show up as named blocks in Nsight Systems — critical for understanding
 * GPU-stream overlaps that timing printfs can't reveal.
 *
 * Usage:
 *   {
 *       NVTX_SCOPE("rotate_vector");         // range starts
 *       // ... work ...
 *   }                                        // range auto-ends
 *
 * Dynamic names (e.g. include a step number) via NVTX_SCOPE_FMT:
 *   NVTX_SCOPE_FMT("rotate step=%d", steps);
 *
 * Overhead is ~100 ns per push/pop when not profiled; ~300 ns when nsys is
 * attached. Zero cost in Release builds that define NEXUS_NO_NVTX.
 */
#pragma once

#ifndef NEXUS_NO_NVTX
#include <nvtx3/nvToolsExt.h>
#include <cstdio>

namespace nexus_trace {

struct ScopedRange {
    ScopedRange(const char *name)               { nvtxRangePushA(name); }
    ~ScopedRange()                              { nvtxRangePop(); }
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange &operator=(const ScopedRange&) = delete;
};

// For dynamic messages — up to 128 chars, formatted once on construction.
struct ScopedRangeFmt {
    template<typename... Args>
    ScopedRangeFmt(const char *fmt, Args... args) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        nvtxRangePushA(buf);
    }
    ~ScopedRangeFmt() { nvtxRangePop(); }
    ScopedRangeFmt(const ScopedRangeFmt&) = delete;
    ScopedRangeFmt &operator=(const ScopedRangeFmt&) = delete;
};

} // namespace nexus_trace

#define NEXUS_NVTX_CAT2(a,b) a##b
#define NEXUS_NVTX_CAT(a,b)  NEXUS_NVTX_CAT2(a,b)
#define NVTX_SCOPE(name) \
    ::nexus_trace::ScopedRange NEXUS_NVTX_CAT(_nvtx_s_, __LINE__)(name)
#define NVTX_SCOPE_FMT(fmt, ...) \
    ::nexus_trace::ScopedRangeFmt NEXUS_NVTX_CAT(_nvtx_fs_, __LINE__)(fmt, __VA_ARGS__)

#else  // NEXUS_NO_NVTX

#define NVTX_SCOPE(name)       (void)0
#define NVTX_SCOPE_FMT(...)    (void)0

#endif
