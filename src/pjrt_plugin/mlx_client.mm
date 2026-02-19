// MLX client implementation

#include "pjrt_plugin/mlx_client.h"

#include <mlx/mlx.h>

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
