// MLX device implementation

#include "pjrt_plugin/mlx_device.h"

namespace jax_mps {

MlxDevice::MlxDevice(int id) : id_(id) {}

int MlxDevice::id() const {
    return id_;
}

int MlxDevice::local_hardware_id() const {
    return id_;
}

}  // namespace jax_mps
