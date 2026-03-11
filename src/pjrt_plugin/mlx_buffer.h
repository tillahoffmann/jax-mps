#pragma once

#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "pjrt_plugin/logging.h"

namespace jax_mps {

// Convert PJRT dtype to MLX dtype
mlx::core::Dtype PjrtDtypeToMlx(int dtype);

// Convert MLX dtype to PJRT dtype
int MlxDtypeToPjrt(mlx::core::Dtype dtype);

// Get element size in bytes for an MLX dtype
inline size_t GetMlxDtypeSize(mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::float32:
        case mlx::core::int32:
        case mlx::core::uint32:
            return 4;
        case mlx::core::float16:
        case mlx::core::bfloat16:
        case mlx::core::int16:
        case mlx::core::uint16:
            return 2;
        case mlx::core::int64:
        case mlx::core::uint64:
        case mlx::core::complex64:
            return 8;
        case mlx::core::int8:
        case mlx::core::uint8:
        case mlx::core::bool_:
            return 1;
        default:
            MPS_LOG_ERROR("GetMlxDtypeSize: unknown dtype, defaulting to 4 bytes\n");
            return 4;
    }
}

class MlxBuffer final {
public:
    ~MlxBuffer() = default;

    // Factory method to create buffer from host data
    static std::unique_ptr<MlxBuffer> FromHostBuffer(const void* data, int dtype,
                                                     const std::vector<int64_t>& dims,
                                                     const std::vector<int64_t>& byte_strides);

    // Factory method to create buffer from existing MLX array
    static std::unique_ptr<MlxBuffer> FromArray(mlx::core::array arr);

    int dtype() const;
    const std::vector<int64_t>& dimensions() const;
    size_t byte_size() const;
    void Delete();
    bool IsDeleted() const;

    // Copy array data to host buffer. Returns true on success, false on error.
    bool ToHostBuffer(void* dst);

    // Access the underlying MLX array
    mlx::core::array& array() {
        return array_;
    }
    const mlx::core::array& array() const {
        return array_;
    }

private:
    // Private constructor that takes an initialized array
    explicit MlxBuffer(mlx::core::array arr);

    mlx::core::array array_;
    std::vector<int64_t> dimensions_;
    int dtype_ = 0;
    size_t byte_size_ = 0;
    bool deleted_ = false;
};

}  // namespace jax_mps
