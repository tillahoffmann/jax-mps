#import "pjrt_plugin/mps_client.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "pjrt_plugin/logging.h"
#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

MpsClient::MpsClient()
    : platform_name_("mps"),
      platform_version_("0.1.0"),
      metal_device_(nullptr),
      command_queue_(nullptr) {}

MpsClient::~MpsClient() {
    devices_.clear();
    if (command_queue_) {
        CFRelease((__bridge CFTypeRef)command_queue_);
        command_queue_ = nullptr;
    }
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

    // Create a shared command queue (avoids context leak warnings)
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        NSLog(@"Failed to create Metal command queue");
        return false;
    }
    command_queue_ = (__bridge_retained void*)queue;

    // Create device wrapper
    NSString* name = [device name];
    auto mps_device = std::make_unique<MpsDevice>(this,
                                                  0,  // device id
                                                  [name UTF8String]);
    devices_.push_back(std::move(mps_device));

    MPS_LOG_INFO("Initialized MPS client with device: %s\n", [name UTF8String]);
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

// Helper to check if strides represent a contiguous (dense row-major) layout
static bool IsContiguous(const std::vector<int64_t>& dims, const std::vector<int64_t>& byte_strides,
                         size_t element_size) {
    if (byte_strides.empty()) {
        return true;  // No strides provided, assumed contiguous
    }
    if (byte_strides.size() != dims.size()) {
        return false;  // Mismatched sizes, treat as non-contiguous
    }

    // For row-major (C-contiguous), stride[i] = element_size * product(dims[i+1:])
    int64_t expected_stride = static_cast<int64_t>(element_size);
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        if (byte_strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= dims[i];
    }
    return true;
}

// Copy non-contiguous strided data to a contiguous buffer
static void CopyStridedToContiguous(const void* src, void* dst, const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& byte_strides, size_t element_size) {
    size_t ndim = dims.size();
    if (ndim == 0) {
        memcpy(dst, src, element_size);
        return;
    }

    // Compute total elements and contiguous strides
    std::vector<int64_t> contiguous_strides(ndim);
    int64_t stride = static_cast<int64_t>(element_size);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        contiguous_strides[i] = stride;
        stride *= dims[i];
    }

    // Iterate over all elements using multi-dimensional index
    std::vector<int64_t> indices(ndim, 0);
    int64_t total_elements = 1;
    for (auto d : dims) {
        total_elements *= d;
    }

    const char* src_base = static_cast<const char*>(src);
    char* dst_base = static_cast<char*>(dst);

    for (int64_t elem = 0; elem < total_elements; ++elem) {
        // Compute source offset from strides
        int64_t src_offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            src_offset += indices[d] * byte_strides[d];
        }

        // Compute destination offset (contiguous)
        int64_t dst_offset = elem * static_cast<int64_t>(element_size);

        // Copy single element
        memcpy(dst_base + dst_offset, src_base + src_offset, element_size);

        // Increment multi-dimensional index
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < dims[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
}

std::unique_ptr<MpsBuffer> MpsClient::BufferFromHostBuffer(const void* data, int dtype,
                                                           const std::vector<int64_t>& dims,
                                                           const std::vector<int64_t>& byte_strides,
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
    size_t element_size = DtypeByteSize(dtype);
    size_t byte_size = element_count * element_size;

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)metal_device_;
    id<MTLBuffer> buffer = nil;

    if (IsContiguous(dims, byte_strides, element_size)) {
        // Data is contiguous, create buffer directly from source
        buffer = [mtl_device newBufferWithBytes:data
                                         length:byte_size
                                        options:MTLResourceStorageModeShared];
    } else {
        // Data is non-contiguous (e.g., from numpy transpose)
        // Allocate buffer and copy with stride handling
        MPS_LOG_DEBUG("BufferFromHostBuffer: copying non-contiguous array to contiguous buffer\n");
        buffer = [mtl_device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
        if (buffer) {
            CopyStridedToContiguous(data, [buffer contents], dims, byte_strides, element_size);
        }
    }

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
