// MLX buffer implementation

#include "pjrt_plugin/mlx_buffer.h"

#include <xla/pjrt/c/pjrt_c_api.h>

#include <cstring>
#include <stdexcept>

#include "pjrt_plugin/logging.h"

namespace jax_mps {

mlx::core::Dtype PjrtDtypeToMlx(int dtype) {
    switch (dtype) {
        case PJRT_Buffer_Type_F32:
            return mlx::core::float32;
        case PJRT_Buffer_Type_F16:
            return mlx::core::float16;
        case PJRT_Buffer_Type_BF16:
            return mlx::core::bfloat16;
        case PJRT_Buffer_Type_S32:
            return mlx::core::int32;
        case PJRT_Buffer_Type_S64:
            return mlx::core::int64;
        case PJRT_Buffer_Type_S16:
            return mlx::core::int16;
        case PJRT_Buffer_Type_S8:
            return mlx::core::int8;
        case PJRT_Buffer_Type_U32:
            return mlx::core::uint32;
        case PJRT_Buffer_Type_U64:
            return mlx::core::uint64;
        case PJRT_Buffer_Type_U16:
            return mlx::core::uint16;
        case PJRT_Buffer_Type_U8:
            return mlx::core::uint8;
        case PJRT_Buffer_Type_PRED:
            return mlx::core::bool_;
        case PJRT_Buffer_Type_F64:
            throw std::runtime_error("MLX does not support float64 (F64). Use "
                                     "jax.config.update('jax_enable_x64', False) "
                                     "or ensure your arrays are float32.");
        case PJRT_Buffer_Type_C64:
            return mlx::core::complex64;
        case PJRT_Buffer_Type_C128:
            MPS_LOG_WARN("MLX does not support complex128, downcast to complex64\n");
            return mlx::core::complex64;
        default:
            throw std::runtime_error("Unsupported PJRT dtype: " + std::to_string(dtype));
    }
}

int MlxDtypeToPjrt(mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::float32:
            return PJRT_Buffer_Type_F32;
        case mlx::core::float16:
            return PJRT_Buffer_Type_F16;
        case mlx::core::bfloat16:
            return PJRT_Buffer_Type_BF16;
        case mlx::core::int32:
            return PJRT_Buffer_Type_S32;
        case mlx::core::int64:
            return PJRT_Buffer_Type_S64;
        case mlx::core::int16:
            return PJRT_Buffer_Type_S16;
        case mlx::core::int8:
            return PJRT_Buffer_Type_S8;
        case mlx::core::uint32:
            return PJRT_Buffer_Type_U32;
        case mlx::core::uint64:
            return PJRT_Buffer_Type_U64;
        case mlx::core::uint16:
            return PJRT_Buffer_Type_U16;
        case mlx::core::uint8:
            return PJRT_Buffer_Type_U8;
        case mlx::core::bool_:
            return PJRT_Buffer_Type_PRED;
        case mlx::core::complex64:
            return PJRT_Buffer_Type_C64;
        default:
            MPS_LOG_ERROR("Unsupported MLX dtype\n");
            return PJRT_Buffer_Type_F32;
    }
}

MlxBuffer::MlxBuffer(mlx::core::array arr) : array_(std::move(arr)), deleted_(false) {
    dtype_ = MlxDtypeToPjrt(array_.dtype());

    // Copy shape
    dimensions_.clear();
    dimensions_.reserve(array_.ndim());
    size_t num_elements = 1;
    for (int i = 0; i < array_.ndim(); ++i) {
        dimensions_.push_back(array_.shape(i));
        num_elements *= array_.shape(i);
    }

    byte_size_ = num_elements * GetMlxDtypeSize(array_.dtype());
}

std::unique_ptr<MlxBuffer> MlxBuffer::FromHostBuffer(const void* data, int dtype,
                                                     const std::vector<int64_t>& dims,
                                                     const std::vector<int64_t>& byte_strides) {
    mlx::core::Dtype mlx_dtype = PjrtDtypeToMlx(dtype);

    // Convert int64_t dims to MLX Shape
    mlx::core::Shape mlx_shape;
    for (auto d : dims) {
        mlx_shape.push_back(static_cast<int>(d));
    }

    size_t elem_size = GetMlxDtypeSize(mlx_dtype);

    // Handle zero-sized tensors - create empty array without reading data
    bool is_zero_sized = false;
    for (auto d : mlx_shape) {
        if (d == 0) {
            is_zero_sized = true;
            break;
        }
    }
    if (is_zero_sized) {
        auto arr = mlx::core::zeros(mlx_shape, mlx_dtype);
        auto buffer = std::unique_ptr<MlxBuffer>(new MlxBuffer(std::move(arr)));
        MPS_LOG_DEBUG("Created zero-sized MlxBuffer: dtype=%d, shape=[", dtype);
        for (size_t i = 0; i < dims.size(); ++i) {
            MPS_LOG_DEBUG("%lld%s", dims[i], i < dims.size() - 1 ? ", " : "");
        }
        MPS_LOG_DEBUG("], byte_size=%zu\n", buffer->byte_size_);
        return buffer;
    }

    // Check if we have contiguous data (no strides or default strides)
    bool is_contiguous = byte_strides.empty();
    if (!is_contiguous) {
        // Check if strides match default row-major layout
        is_contiguous = true;
        size_t expected_stride = elem_size;
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            if (byte_strides[i] != static_cast<int64_t>(expected_stride)) {
                is_contiguous = false;
                break;
            }
            expected_stride *= dims[i];
        }
    }

    if (!is_contiguous) {
        // Non-contiguous data requires copying with stride handling
        // For now, log an error and return nullptr
        MPS_LOG_ERROR("Non-contiguous strides not supported - data must be contiguous\n");
        return nullptr;
    }

    // Create MLX array from host data - cast to correct type based on dtype
    mlx::core::array arr(0.0F);  // Placeholder, will be reassigned

    // The iterator-based constructor expects the pointer type to match the value type
    // Use a switch to cast to the correct pointer type
    switch (mlx_dtype) {
        case mlx::core::float32:
            arr = mlx::core::array(static_cast<const float*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::float16:
            // MLX uses its own float16 type, but it's binary compatible with uint16
            arr = mlx::core::array(static_cast<const mlx::core::float16_t*>(data), mlx_shape,
                                   mlx_dtype);
            break;
        case mlx::core::bfloat16:
            arr = mlx::core::array(static_cast<const mlx::core::bfloat16_t*>(data), mlx_shape,
                                   mlx_dtype);
            break;
        case mlx::core::int32:
            arr = mlx::core::array(static_cast<const int32_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::int64:
            arr = mlx::core::array(static_cast<const int64_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::int16:
            arr = mlx::core::array(static_cast<const int16_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::int8:
            arr = mlx::core::array(static_cast<const int8_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::uint32:
            arr = mlx::core::array(static_cast<const uint32_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::uint64:
            arr = mlx::core::array(static_cast<const uint64_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::uint16:
            arr = mlx::core::array(static_cast<const uint16_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::uint8:
            arr = mlx::core::array(static_cast<const uint8_t*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::bool_:
            arr = mlx::core::array(static_cast<const bool*>(data), mlx_shape, mlx_dtype);
            break;
        case mlx::core::complex64:
            arr = mlx::core::array(static_cast<const mlx::core::complex64_t*>(data), mlx_shape,
                                   mlx_dtype);
            break;
        default:
            // Fallback - treat as float32
            arr = mlx::core::array(static_cast<const float*>(data), mlx_shape, mlx_dtype);
            break;
    }

    auto buffer = std::unique_ptr<MlxBuffer>(new MlxBuffer(std::move(arr)));

    MPS_LOG_DEBUG("Created MlxBuffer: dtype=%d, shape=[", dtype);
    for (size_t i = 0; i < dims.size(); ++i) {
        MPS_LOG_DEBUG("%lld%s", dims[i], i < dims.size() - 1 ? ", " : "");
    }
    MPS_LOG_DEBUG("], byte_size=%zu\n", buffer->byte_size_);

    return buffer;
}

std::unique_ptr<MlxBuffer> MlxBuffer::FromArray(mlx::core::array arr) {
    auto buffer = std::unique_ptr<MlxBuffer>(new MlxBuffer(std::move(arr)));

    MPS_LOG_DEBUG("Created MlxBuffer from array: dtype=%d, ndim=%zu, byte_size=%zu\n",
                  buffer->dtype_, buffer->dimensions_.size(), buffer->byte_size_);

    return buffer;
}

int MlxBuffer::dtype() const {
    return dtype_;
}

const std::vector<int64_t>& MlxBuffer::dimensions() const {
    return dimensions_;
}

size_t MlxBuffer::byte_size() const {
    return byte_size_;
}

void MlxBuffer::Delete() {
    deleted_ = true;
}

bool MlxBuffer::IsDeleted() const {
    return deleted_;
}

bool MlxBuffer::ToHostBuffer(void* dst) {
    if (deleted_) {
        MPS_LOG_ERROR("Attempting to read deleted buffer\n");
        return false;
    }

    // Zero-sized tensors have nothing to copy
    if (byte_size_ == 0) {
        return true;
    }

    if (!dst) {
        MPS_LOG_ERROR("Null destination buffer\n");
        return false;
    }

    // Validate dtype consistency - the array's dtype should match what we stored
    int current_dtype = MlxDtypeToPjrt(array_.dtype());
    if (current_dtype != dtype_) {
        MPS_LOG_ERROR("Dtype mismatch: buffer has %d but array has %d\n", dtype_, current_dtype);
        return false;
    }

    // Make the array contiguous (broadcasts create views that share memory)
    // and evaluate to ensure computation is complete
    try {
        array_ = mlx::core::contiguous(array_);
        mlx::core::eval(array_);
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("MLX eval failed in ToHostBuffer: %s\n", e.what());
        return false;
    }

    std::memcpy(dst, array_.data<void>(), byte_size_);

    MPS_LOG_DEBUG("Copied %zu bytes to host\n", byte_size_);
    return true;
}

}  // namespace jax_mps
