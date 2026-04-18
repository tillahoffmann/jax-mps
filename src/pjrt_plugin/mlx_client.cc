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

    // Cap MLX's internal buffer cache. By default MLX sizes the cache at
    // ~1.5x the recommended working set (tens of GB on Apple Silicon),
    // which is fine for a long-running training script but causes the cache
    // to grow without bound across thousands of unrelated JAX
    // computations. The cached MTLBuffers stay in the residency set and,
    // once the cache exceeds physical memory, are swapped to disk; we have
    // observed 23 GB of swapped IOAccelerator memory after a full upstream
    // JAX test run, alongside intermittent command-buffer hangs.
    //
    // 1 GiB is plenty to absorb hot allocations within a single
    // computation while letting MLX evict and release MTLBuffers between
    // unrelated computations. Override with JAX_MPS_CACHE_LIMIT_BYTES.
    size_t cache_limit = 1ULL << 30;
    if (const char* env = std::getenv("JAX_MPS_CACHE_LIMIT_BYTES")) {
        // std::stoull accepts a leading '-' (wraps to a huge unsigned) and
        // silently stops at trailing junk like "1024abc". Validate
        // explicitly so a typo doesn't accidentally disable the cap.
        std::string s(env);
        size_t pos = 0;
        bool ok = !s.empty() && s.find('-') == std::string::npos;
        if (ok) {
            try {
                cache_limit = std::stoull(s, &pos);
            } catch (const std::exception&) {
                ok = false;
            }
        }
        if (!ok || pos != s.size()) {
            MPS_LOG_WARN("Invalid JAX_MPS_CACHE_LIMIT_BYTES=%s, using default\n", env);
            cache_limit = 1ULL << 30;
        }
    }
    mlx::core::set_cache_limit(cache_limit);

    MPS_LOG_DEBUG("MlxClient initialized with MLX GPU backend (cache_limit=%zu)\n", cache_limit);
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
