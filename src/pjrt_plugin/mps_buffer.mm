#import "pjrt_plugin/mps_buffer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"

namespace jax_mps {

// PJRT dtype enum values (from pjrt_c_api.h)
enum PjrtDtype {
    PJRT_INVALID = 0,
    PJRT_PRED = 1,
    PJRT_S8 = 2,
    PJRT_S16 = 3,
    PJRT_S32 = 4,
    PJRT_S64 = 5,
    PJRT_U8 = 6,
    PJRT_U16 = 7,
    PJRT_U32 = 8,
    PJRT_U64 = 9,
    PJRT_F16 = 10,
    PJRT_F32 = 11,
    PJRT_F64 = 12,
    PJRT_BF16 = 16,
    PJRT_C64 = 15,
    PJRT_C128 = 18,
};

size_t DtypeByteSize(int dtype) {
    switch (dtype) {
        case PJRT_PRED:
        case PJRT_S8:
        case PJRT_U8:
            return 1;
        case PJRT_S16:
        case PJRT_U16:
        case PJRT_F16:
        case PJRT_BF16:
            return 2;
        case PJRT_S32:
        case PJRT_U32:
        case PJRT_F32:
            return 4;
        case PJRT_S64:
        case PJRT_U64:
        case PJRT_F64:
        case PJRT_C64:
            return 8;
        case PJRT_C128:
            return 16;
        default:
            // Unknown dtype - return 0 to make failures obvious
            // Caller should validate dtype before calling this
            NSLog(@"ERROR: Unknown dtype %d in DtypeByteSize", dtype);
            return 0;
    }
}

MpsBuffer::MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype,
                     const std::vector<int64_t>& dims)
    : device_(device), metal_buffer_(metal_buffer), dtype_(dtype), dims_(dims) {
    // Retain the Metal buffer
    if (metal_buffer_) {
        CFRetain((__bridge CFTypeRef)metal_buffer_);
    }
}

MpsBuffer::~MpsBuffer() {
    Delete();
}

int64_t MpsBuffer::element_count() const {
    int64_t count = 1;
    for (int64_t dim : dims_) {
        count *= dim;
    }
    return count;
}

size_t MpsBuffer::byte_size() const {
    return element_count() * DtypeByteSize(dtype_);
}

void MpsBuffer::ToHostBuffer(void* dst, std::function<void()> on_done) {
    if (is_deleted_ || !metal_buffer_) {
        if (on_done)
            on_done();
        return;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer_;

    // For shared memory (Apple Silicon), we can read directly
    void* contents = [buffer contents];
    if (contents && dst) {
        memcpy(dst, contents, byte_size());
    }

    if (on_done)
        on_done();
}

void MpsBuffer::Delete() {
    if (!is_deleted_ && metal_buffer_) {
        CFRelease((__bridge CFTypeRef)metal_buffer_);
        metal_buffer_ = nullptr;
    }
    is_deleted_ = true;
}

}  // namespace jax_mps
