#import "pjrt_plugin/mps_device.h"

#import <Metal/Metal.h>

#import "pjrt_plugin/mps_client.h"

namespace jax_mps {

MpsDevice::MpsDevice(MpsClient* client, int id, const std::string& name)
    : client_(client), id_(id), device_kind_("GPU") {
    debug_string_ = "MPS:" + std::to_string(id) + " " + name;
}

MpsDevice::~MpsDevice() = default;

}  // namespace jax_mps
