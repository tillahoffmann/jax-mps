// Global mutex for serializing all PJRT operations that touch MLX.
// MLX's Metal backend is not thread-safe, so concurrent calls from jaxlib
// (e.g. test_concurrent_jit) cause SIGABRT.
#pragma once

#include <mutex>

inline std::mutex& GetPjrtGlobalMutex() {
    static std::mutex mutex;
    return mutex;
}
