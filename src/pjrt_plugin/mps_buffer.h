#pragma once

#include <cstdint>
#include <vector>
#include <functional>

namespace jax_mps {

class MpsDevice;

// Represents a buffer on Metal GPU
class MpsBuffer {
public:
    MpsBuffer(MpsDevice* device,
              void* metal_buffer,  // id<MTLBuffer>
              int dtype,
              const std::vector<int64_t>& dims);
    ~MpsBuffer();

    // Buffer info
    MpsDevice* device() const { return device_; }
    int dtype() const { return dtype_; }
    const std::vector<int64_t>& dimensions() const { return dims_; }
    int64_t element_count() const;
    size_t byte_size() const;

    // Data access
    void* metal_buffer() const { return metal_buffer_; }

    // Copy to host
    void ToHostBuffer(void* dst, std::function<void()> on_done);

    // Check if buffer is deleted
    bool IsDeleted() const { return is_deleted_; }
    void Delete();

private:
    MpsDevice* device_;
    void* metal_buffer_;  // id<MTLBuffer>
    int dtype_;
    std::vector<int64_t> dims_;
    bool is_deleted_ = false;
};

// Size of each dtype in bytes
size_t DtypeByteSize(int dtype);

}  // namespace jax_mps
