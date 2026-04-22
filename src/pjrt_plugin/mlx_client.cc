// MLX client implementation

#include "pjrt_plugin/mlx_client.h"

#include <mlx/mlx.h>

#include <cstdlib>
#include <exception>
#include <string>

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/mlx_device.h"
#include "pjrt_plugin/mlx_executable.h"
#include "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

MlxClient::MlxClient() {
    // Create a single device
    devices_.push_back(std::make_unique<MlxDevice>(0));

    // Set MLX to use GPU (Metal) device
    mlx::core::set_default_device(mlx::core::Device::gpu);

    // Optional cap on MLX's internal buffer cache. MLX's own default sizes the
    // cache at ~1.5x the recommended working set (tens of GB on Apple Silicon),
    // which matches what training/inference workloads actually want — a hard
    // 1 GiB cap caused 30–70% regressions on Max-class GPUs (#134) because each
    // training iteration evicts and reallocates MTLBuffers.
    //
    // The pathological case the cap was originally added for (#139) is the
    // upstream JAX test suite: thousands of unrelated computations in one
    // process, where freed MTLBuffers stay resident and eventually swap. That
    // scenario sets JAX_MPS_CACHE_LIMIT_BYTES explicitly via
    // scripts/run_jax_tests.py.
    if (const char* env = std::getenv("JAX_MPS_CACHE_LIMIT_BYTES")) {
        // std::stoull accepts a leading '-' (wraps to a huge unsigned) and
        // silently stops at trailing junk like "1024abc". Validate
        // explicitly so a typo doesn't accidentally disable the cap.
        std::string s(env);
        size_t pos = 0;
        size_t cache_limit = 0;
        bool ok = !s.empty() && s.find('-') == std::string::npos;
        if (ok) {
            try {
                cache_limit = std::stoull(s, &pos);
            } catch (const std::exception&) {
                ok = false;
            }
        }
        if (!ok || pos != s.size()) {
            // Caller meant to set a cap and got it wrong. Fall back to 1 GiB
            // rather than the MLX default — a typo shouldn't silently
            // re-enable unbounded residency growth in the scenarios that
            // motivated this knob (long multi-computation processes).
            cache_limit = 1ULL << 30;
            MPS_LOG_WARN("Invalid JAX_MPS_CACHE_LIMIT_BYTES=%s, falling back to %zu\n", env,
                         cache_limit);
        }
        mlx::core::set_cache_limit(cache_limit);
        MPS_LOG_DEBUG("MlxClient cache_limit set to %zu\n", cache_limit);
    }

    MPS_LOG_DEBUG("MlxClient initialized with MLX GPU backend\n");
}

MlxClient::~MlxClient() = default;

int MlxClient::device_count() const {
    return static_cast<int>(devices_.size());
}

MlxDevice* MlxClient::device(int index) {
    if (index >= 0 && index < static_cast<int>(devices_.size())) {
        return devices_[index].get();
    }
    return nullptr;
}

void* MlxClient::metal_device() const {
    // MLX manages its own Metal device internally
    // Return non-null to indicate we have a valid device
    return reinterpret_cast<void*>(1);
}

std::unique_ptr<MlxExecutable> MlxClient::CompileStableHLO(mps::ParsedModule parsed_module,
                                                           void* options) {
    MPS_LOG_DEBUG("Compiling StableHLO module\n");
    return MlxExecutable::Create(std::move(parsed_module));
}

std::unique_ptr<MlxBuffer> MlxClient::BufferFromHostBuffer(const void* data, int dtype,
                                                           const std::vector<int64_t>& dims,
                                                           const std::vector<int64_t>& byte_strides,
                                                           MlxDevice* device) {
    MPS_LOG_DEBUG("Creating buffer from host: dtype=%d, ndims=%zu\n", dtype, dims.size());
    return MlxBuffer::FromHostBuffer(data, dtype, dims, byte_strides);
}

}  // namespace jax_mps
