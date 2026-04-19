// Global mutex for serializing all PJRT operations that touch MLX.
// MLX's Metal backend is not thread-safe, so concurrent calls from jaxlib
// (e.g. test_concurrent_jit) cause SIGABRT.
//
// Uses std::recursive_mutex because WhileLoopPrimitive callbacks re-enter
// Execute() while the outer Execute() still holds the lock.
#pragma once

#include <mutex>

inline std::recursive_mutex& GetPjrtGlobalMutex() {
    static std::recursive_mutex mutex;
    return mutex;
}
