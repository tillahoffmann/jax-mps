#import "pjrt_plugin/mps_client.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

MpsClient::MpsClient()
    : platform_name_("mps"), platform_version_("0.1.0"), metal_device_(nullptr) {}

MpsClient::~MpsClient() {
    devices_.clear();
    if (metal_device_) {
        CFRelease((__bridge CFTypeRef)metal_device_);
        metal_device_ = nullptr;
    }
}

std::unique_ptr<MpsClient> MpsClient::Create() {
    auto client = std::unique_ptr<MpsClient>(new MpsClient());
    if (!client->Initialize()) {
        return nullptr;
    }
    return client;
}

bool MpsClient::Initialize() {
    // Get the default Metal device (GPU)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"Failed to create Metal device");
        return false;
    }

    metal_device_ = (__bridge_retained void*)device;

    // Create device wrapper
    NSString* name = [device name];
    auto mps_device = std::make_unique<MpsDevice>(this,
                                                  0,  // device id
                                                  [name UTF8String]);
    devices_.push_back(std::move(mps_device));

    NSLog(@"Initialized MPS client with device: %@", name);
    return true;
}

int MpsClient::device_count() const {
    return static_cast<int>(devices_.size());
}

int MpsClient::addressable_device_count() const {
    return device_count();  // All devices addressable in single process
}

MpsDevice* MpsClient::device(int index) {
    if (index < 0 || index >= devices_.size()) {
        return nullptr;
    }
    return devices_[index].get();
}

MpsDevice* MpsClient::addressable_device(int index) {
    return device(index);
}

MpsDevice* MpsClient::LookupDevice(int device_id) {
    for (auto& dev : devices_) {
        if (dev->id() == device_id) {
            return dev.get();
        }
    }
    return nullptr;
}

std::unique_ptr<MpsBuffer> MpsClient::BufferFromHostBuffer(const void* data, int dtype,
                                                           const std::vector<int64_t>& dims,
                                                           MpsDevice* device) {
    if (!metal_device_) {
        NSLog(@"ERROR: BufferFromHostBuffer called with no Metal device");
        return nullptr;
    }
    if (!data) {
        NSLog(@"ERROR: BufferFromHostBuffer called with null data pointer");
        return nullptr;
    }

    // Calculate buffer size
    int64_t element_count = 1;
    for (int64_t dim : dims) {
        element_count *= dim;
    }
    size_t byte_size = element_count * DtypeByteSize(dtype);

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)metal_device_;

    // Create Metal buffer with shared storage mode (efficient on Apple Silicon)
    id<MTLBuffer> buffer = [mtl_device newBufferWithBytes:data
                                                   length:byte_size
                                                  options:MTLResourceStorageModeShared];
    if (!buffer) {
        NSLog(@"Failed to create Metal buffer of size %zu", byte_size);
        return nullptr;
    }

    return std::make_unique<MpsBuffer>(device ? device : devices_[0].get(), (__bridge void*)buffer,
                                       dtype, dims);
}

std::unique_ptr<MpsExecutable> MpsClient::CompileStableHLO(mps::ParsedModule module,
                                                           MpsDevice* device) {
    // Create executable from ParsedModule (takes ownership)
    auto exec = std::make_unique<MpsExecutable>(this, std::move(module));
    if (!exec->IsValid()) {
        NSLog(@"Failed to compile StableHLO module: %s", exec->error().c_str());
        return nullptr;
    }

    return exec;
}

}  // namespace jax_mps
