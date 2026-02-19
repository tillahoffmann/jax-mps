// MLX executable implementation with op dispatch

#include "pjrt_plugin/mlx_executable.h"

#include <mlx/mlx.h>

#include <functional>
#include <stdexcept>
#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/type_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Convert MLIR type to MLX dtype
mlx::core::Dtype MlirTypeToMlxDtype(mlir::Type type) {
    // Use the same logic as MlirTypeToPjrtDtype but return MLX types
    int pjrt_dtype = MlirTypeToPjrtDtype(type);

    // Check for unknown type
    if (pjrt_dtype == -1) {
        MPS_LOG_ERROR("Unknown MLIR type, defaulting to float32\n");
        return mlx::core::float32;
    }

    // Handle float64 downcast (MLX doesn't support float64)
    if (type.isF64()) {
        MPS_LOG_WARN("MLX doesn't support float64, downcasting to float32\n");
        return mlx::core::float32;
    }

    return PjrtDtypeToMlx(pjrt_dtype);
}

// Extract shape from RankedTensorType
mlx::core::Shape GetShape(mlir::RankedTensorType type) {
    mlx::core::Shape shape;
    for (int64_t dim : type.getShape()) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}

// Alias for the shared GetMlxDtypeSize function
inline size_t GetDtypeSize(mlx::core::Dtype dtype) {
    return GetMlxDtypeSize(dtype);
}

// Create MLX array from DenseElementsAttr (for constants)
// Returns empty optional on error
// Note: MLX's array constructor uses std::copy with typed pointers, so we must
// cast to the correct element type before passing to the constructor.
std::optional<mlx::core::array> CreateArrayFromDenseAttr(mlir::DenseElementsAttr attr) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
    if (!tensorType) {
        MPS_LOG_ERROR("Constant attribute is not a ranked tensor type\n");
        return std::nullopt;
    }

    auto shape = GetShape(tensorType);
    auto elemType = tensorType.getElementType();
    auto mlxDtype = MlirTypeToMlxDtype(elemType);
    auto rawData = attr.getRawData();

    // Helper to create array with correct pointer type
    auto createArray = [&](auto* typed_ptr) -> mlx::core::array {
        return mlx::core::array(typed_ptr, shape, mlxDtype);
    };

    // Handle splat constants (single value broadcast to shape)
    if (attr.isSplat()) {
        mlx::core::Shape scalarShape = {};
        std::optional<mlx::core::array> scalar_opt;

        // Create scalar with correctly-typed pointer
        switch (mlxDtype) {
            case mlx::core::int32:
                scalar_opt = mlx::core::array(reinterpret_cast<const int32_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int64:
                scalar_opt = mlx::core::array(reinterpret_cast<const int64_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint32:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint32_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint64:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint64_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::float32:
                scalar_opt = mlx::core::array(reinterpret_cast<const float*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::float16:
                scalar_opt =
                    mlx::core::array(reinterpret_cast<const mlx::core::float16_t*>(rawData.data()),
                                     scalarShape, mlxDtype);
                break;
            case mlx::core::bfloat16:
                scalar_opt =
                    mlx::core::array(reinterpret_cast<const mlx::core::bfloat16_t*>(rawData.data()),
                                     scalarShape, mlxDtype);
                break;
            case mlx::core::bool_:
                scalar_opt = mlx::core::array(reinterpret_cast<const bool*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int8:
                scalar_opt = mlx::core::array(reinterpret_cast<const int8_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint8:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint8_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int16:
                scalar_opt = mlx::core::array(reinterpret_cast<const int16_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint16:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint16_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            default:
                MPS_LOG_ERROR("Unsupported dtype %d for splat constant\n",
                              static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
                return std::nullopt;
        }

        if (!scalar_opt) {
            return std::nullopt;
        }

        if (shape.empty()) {
            return scalar_opt;
        }
        return mlx::core::broadcast_to(*scalar_opt, shape);
    }

    // Validate data size matches expected size
    size_t elemSize = GetDtypeSize(mlxDtype);
    size_t numElements = 1;
    for (int dim : shape) {
        numElements *= dim;
    }
    size_t expectedSize = numElements * elemSize;

    if (rawData.size() < expectedSize) {
        MPS_LOG_ERROR("Constant data size mismatch: got %zu bytes, expected %zu\n", rawData.size(),
                      expectedSize);
        return std::nullopt;
    }

    // Create array with correctly-typed pointer for proper std::copy behavior
    switch (mlxDtype) {
        case mlx::core::int32:
            return createArray(reinterpret_cast<const int32_t*>(rawData.data()));
        case mlx::core::int64:
            return createArray(reinterpret_cast<const int64_t*>(rawData.data()));
        case mlx::core::uint32:
            return createArray(reinterpret_cast<const uint32_t*>(rawData.data()));
        case mlx::core::uint64:
            return createArray(reinterpret_cast<const uint64_t*>(rawData.data()));
        case mlx::core::float32:
            return createArray(reinterpret_cast<const float*>(rawData.data()));
        case mlx::core::float16:
            return createArray(reinterpret_cast<const mlx::core::float16_t*>(rawData.data()));
        case mlx::core::bfloat16:
            return createArray(reinterpret_cast<const mlx::core::bfloat16_t*>(rawData.data()));
        case mlx::core::bool_:
            return createArray(reinterpret_cast<const bool*>(rawData.data()));
        case mlx::core::int8:
            return createArray(reinterpret_cast<const int8_t*>(rawData.data()));
        case mlx::core::uint8:
            return createArray(reinterpret_cast<const uint8_t*>(rawData.data()));
        case mlx::core::int16:
            return createArray(reinterpret_cast<const int16_t*>(rawData.data()));
        case mlx::core::uint16:
            return createArray(reinterpret_cast<const uint16_t*>(rawData.data()));
        default:
            MPS_LOG_ERROR("Unsupported dtype %d for constant\n",
                          static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
            return std::nullopt;
    }
}

// Value map type using void* as key (from mlir::Value's opaque pointer)
using ValueMap = std::unordered_map<void*, mlx::core::array>;

void* ToKey(mlir::Value v) {
    return v.getAsOpaquePointer();
}

// Helper to safely get value from map
std::optional<std::reference_wrapper<mlx::core::array>> GetValue(ValueMap& values, mlir::Value v) {
    auto it = values.find(ToKey(v));
    if (it == values.end()) {
        return std::nullopt;
    }
    return std::ref(it->second);
}

// Execution context passed to handlers
struct ExecContext {
    mlir::ModuleOp module;
};

// Op handler function type
using OpHandler =
    std::function<bool(mlir::Operation*, ValueMap&, std::vector<mlx::core::array>&, ExecContext&)>;

// Forward declaration for recursive call handling
bool ExecuteFunction(mlir::func::FuncOp func, const std::vector<mlx::core::array>& inputs,
                     std::vector<mlx::core::array>& outputs, ExecContext& ctx);

// Handler for stablehlo.add
bool HandleAdd(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.add: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::add(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.exponential
bool HandleExp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.exponential: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::exp(input_opt->get()));
    return true;
}

// Handler for stablehlo.log
bool HandleLog(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.log: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::log(input_opt->get()));
    return true;
}

// Handler for stablehlo.rsqrt
bool HandleRsqrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.rsqrt: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::rsqrt(input_opt->get()));
    return true;
}

// Handler for stablehlo.floor
bool HandleFloor(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.floor: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::floor(input_opt->get()));
    return true;
}

// Handler for stablehlo.sine
bool HandleSine(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.sine: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::sin(input_opt->get()));
    return true;
}

// Handler for stablehlo.cosine
bool HandleCosine(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cosine: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::cos(input_opt->get()));
    return true;
}

// Handler for stablehlo.minimum
bool HandleMinimum(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.minimum: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::minimum(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.clamp
bool HandleClamp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto min_opt = GetValue(values, op->getOperand(0));
    auto operand_opt = GetValue(values, op->getOperand(1));
    auto max_opt = GetValue(values, op->getOperand(2));
    if (!min_opt || !operand_opt || !max_opt) {
        MPS_LOG_ERROR("stablehlo.clamp: operand not found in value map\n");
        return false;
    }
    // clamp(min, x, max) -> maximum(min, minimum(x, max))
    auto clamped = mlx::core::clip(operand_opt->get(), min_opt->get(), max_opt->get());
    values.emplace(ToKey(op->getResult(0)), clamped);
    return true;
}

// Handler for stablehlo.constant
bool HandleConstant(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto constOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constOp) {
        MPS_LOG_ERROR("stablehlo.constant: failed to cast to ConstantOp\n");
        return false;
    }

    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!attr) {
        MPS_LOG_ERROR("stablehlo.constant: value is not DenseElementsAttr\n");
        return false;
    }

    auto arr_opt = CreateArrayFromDenseAttr(attr);
    if (!arr_opt) {
        return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(*arr_opt));
    return true;
}

// Handler for stablehlo.reshape
bool HandleReshape(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.reshape: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.reshape: result type is not RankedTensorType\n");
        return false;
    }

    auto newShape = GetShape(resultType);
    values.emplace(ToKey(op->getResult(0)), mlx::core::reshape(input_opt->get(), newShape));
    return true;
}

// Handler for stablehlo.broadcast_in_dim
bool HandleBroadcastInDim(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
    if (!broadcastOp) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: result type is not RankedTensorType\n");
        return false;
    }

    auto& input = input_opt->get();
    auto outputShape = GetShape(resultType);
    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    // Validate broadcast dimensions are in bounds
    for (int64_t dim : broadcastDims) {
        if (dim < 0 || static_cast<size_t>(dim) >= outputShape.size()) {
            MPS_LOG_ERROR("stablehlo.broadcast_in_dim: dimension %lld out of bounds [0, %zu)\n",
                          dim, outputShape.size());
            return false;
        }
    }

    // Build input shape
    mlx::core::Shape inputShape;
    for (int i = 0; i < input.ndim(); ++i) {
        inputShape.push_back(input.shape(i));
    }

    // Build the intermediate shape with 1s for non-broadcast dims
    // For broadcast_in_dim, we reshape input to have 1s in dimensions not in broadcastDims,
    // then broadcast to the final shape
    mlx::core::Shape intermediateShape(outputShape.size(), 1);
    for (size_t i = 0; i < broadcastDims.size(); ++i) {
        int64_t dim = broadcastDims[i];
        if (i < inputShape.size()) {
            intermediateShape[dim] = inputShape[i];
        }
    }

    // Reshape input to intermediate shape
    auto reshaped = mlx::core::reshape(input, intermediateShape);

    // Broadcast to final shape
    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, outputShape));
    return true;
}

// Common helper for return-like operations (func.return, stablehlo.return)
bool CollectReturnValues(mlir::Operation* op, ValueMap& values,
                         std::vector<mlx::core::array>& outputs, const char* opName) {
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        outputs.push_back(val_opt->get());
    }
    return true;
}

// Handler for func.return
bool HandleReturn(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "func.return");
}

// Handler for stablehlo.concatenate
bool HandleConcatenate(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op);
    if (!concatOp) {
        MPS_LOG_ERROR("stablehlo.concatenate: failed to cast\n");
        return false;
    }

    std::vector<mlx::core::array> inputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.concatenate: operand not found in value map\n");
            return false;
        }
        inputs.push_back(val_opt->get());
    }

    auto axis = concatOp.getDimension();
    values.emplace(ToKey(op->getResult(0)), mlx::core::concatenate(inputs, static_cast<int>(axis)));
    return true;
}

// Handler for stablehlo.and (bitwise and)
bool HandleAnd(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.and: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_and(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.convert (type conversion)
bool HandleConvert(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.convert: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.convert: result type is not RankedTensorType\n");
        return false;
    }

    auto targetDtype = MlirTypeToMlxDtype(resultType.getElementType());
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(input_opt->get(), targetDtype));
    return true;
}

// Handler for stablehlo.shift_right_logical
bool HandleShiftRightLogical(mlir::Operation* op, ValueMap& values,
                             std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.shift_right_logical: operand not found in value map\n");
        return false;
    }
    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();

    // StableHLO spec: shift >= bit_width gives 0 for logical right shift
    // MLX may use shift % bit_width, so we need to handle this case
    int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
    auto zero = mlx::core::zeros_like(lhs);
    auto mask = mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype()));
    auto shifted = mlx::core::right_shift(lhs, rhs);
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(mask, zero, shifted));
    return true;
}

// Handler for stablehlo.multiply
bool HandleMultiply(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.multiply: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::multiply(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.xor
bool HandleXor(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.xor: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_xor(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.or
bool HandleOr(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
              ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.or: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_or(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.shift_left
bool HandleShiftLeft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.shift_left: operand not found in value map\n");
        return false;
    }
    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();

    // StableHLO spec: shift >= bit_width gives 0 for left shift
    int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
    auto zero = mlx::core::zeros_like(lhs);
    auto mask = mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype()));
    auto shifted = mlx::core::left_shift(lhs, rhs);
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(mask, zero, shifted));
    return true;
}

// Handler for stablehlo.iota
bool HandleIota(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        MPS_LOG_ERROR("stablehlo.iota: failed to cast\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.iota: result type is not RankedTensorType\n");
        return false;
    }

    auto shape = GetShape(resultType);
    auto dtype = MlirTypeToMlxDtype(resultType.getElementType());
    uint64_t iotaDim = iotaOp.getIotaDimension();

    // Create iota: values are 0, 1, 2, ... along the iota dimension
    // Start with arange for the iota dimension size
    int dimSize = shape[iotaDim];
    auto iota1d = mlx::core::arange(0, dimSize, dtype);

    // Reshape to have 1s everywhere except the iota dimension
    mlx::core::Shape reshapeShape(shape.size(), 1);
    reshapeShape[iotaDim] = dimSize;
    auto reshaped = mlx::core::reshape(iota1d, reshapeShape);

    // Broadcast to final shape
    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, shape));
    return true;
}

// Handler for stablehlo.slice
bool HandleSlice(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
    if (!sliceOp) {
        MPS_LOG_ERROR("stablehlo.slice: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.slice: operand not found in value map\n");
        return false;
    }

    auto startIndices = sliceOp.getStartIndices();
    auto limitIndices = sliceOp.getLimitIndices();
    auto strides = sliceOp.getStrides();

    mlx::core::Shape starts;
    mlx::core::Shape stops;
    mlx::core::Shape steps;
    for (size_t i = 0; i < startIndices.size(); ++i) {
        starts.push_back(static_cast<int>(startIndices[i]));
        stops.push_back(static_cast<int>(limitIndices[i]));
        steps.push_back(static_cast<int>(strides[i]));
    }

    values.emplace(ToKey(op->getResult(0)),
                   mlx::core::slice(input_opt->get(), starts, stops, steps));
    return true;
}

// Handler for stablehlo.dynamic_slice
bool HandleDynamicSlice(mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dynamicSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
    if (!dynamicSliceOp) {
        MPS_LOG_ERROR("stablehlo.dynamic_slice: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.dynamic_slice: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto sliceSizes = dynamicSliceOp.getSliceSizes();

    // Get start indices from remaining operands (they are scalar tensors)
    mlx::core::Shape starts;
    mlx::core::Shape stops;
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
        auto idx_opt = GetValue(values, op->getOperand(i));
        if (!idx_opt) {
            MPS_LOG_ERROR("stablehlo.dynamic_slice: start index operand not found\n");
            return false;
        }
        // Eval and extract scalar value - handle different integer types
        auto& idx_arr = idx_opt->get();
        mlx::core::eval(idx_arr);

        int start;
        switch (idx_arr.dtype()) {
            case mlx::core::int32:
                start = idx_arr.item<int32_t>();
                break;
            case mlx::core::int64:
                start = static_cast<int>(idx_arr.item<int64_t>());
                break;
            case mlx::core::uint32:
                start = static_cast<int>(idx_arr.item<uint32_t>());
                break;
            case mlx::core::uint64:
                start = static_cast<int>(idx_arr.item<uint64_t>());
                break;
            default:
                // Fall back to int32, may throw if incompatible
                start = idx_arr.item<int>();
                break;
        }

        int size = static_cast<int>(sliceSizes[i - 1]);
        starts.push_back(start);
        stops.push_back(start + size);
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::slice(input, starts, stops));
    return true;
}

// Handler for stablehlo.subtract
bool HandleSubtract(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.subtract: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::subtract(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.negate
bool HandleNegate(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.negate: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::negative(input_opt->get()));
    return true;
}

// Handler for stablehlo.abs
bool HandleAbs(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.abs: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::abs(input_opt->get()));
    return true;
}

// Handler for stablehlo.sqrt
bool HandleSqrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.sqrt: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::sqrt(input_opt->get()));
    return true;
}

// Handler for stablehlo.log_plus_one (log1p)
bool HandleLogPlusOne(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.log_plus_one: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::log1p(input_opt->get()));
    return true;
}

// Handler for stablehlo.maximum
bool HandleMaximum(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.maximum: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::maximum(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.divide
bool HandleDivide(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.divide: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::divide(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.select
bool HandleSelect(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto cond_opt = GetValue(values, op->getOperand(0));
    auto true_opt = GetValue(values, op->getOperand(1));
    auto false_opt = GetValue(values, op->getOperand(2));
    if (!cond_opt || !true_opt || !false_opt) {
        MPS_LOG_ERROR("stablehlo.select: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)),
                   mlx::core::where(cond_opt->get(), true_opt->get(), false_opt->get()));
    return true;
}

// Handler for stablehlo.compare
bool HandleCompare(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        MPS_LOG_ERROR("stablehlo.compare: failed to cast\n");
        return false;
    }

    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.compare: operand not found in value map\n");
        return false;
    }

    auto direction = compareOp.getComparisonDirection();
    std::optional<mlx::core::array> result;

    using Dir = mlir::stablehlo::ComparisonDirection;
    switch (direction) {
        case Dir::EQ:
            result = mlx::core::equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::NE:
            result = mlx::core::not_equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::LT:
            result = mlx::core::less(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::LE:
            result = mlx::core::less_equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::GT:
            result = mlx::core::greater(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::GE:
            result = mlx::core::greater_equal(lhs_opt->get(), rhs_opt->get());
            break;
        default:
            MPS_LOG_ERROR("stablehlo.compare: unsupported comparison direction\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(*result));
    return true;
}

// Handler for stablehlo.bitcast_convert
bool HandleBitcastConvert(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.bitcast_convert: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.bitcast_convert: result type is not RankedTensorType\n");
        return false;
    }

    auto targetDtype = MlirTypeToMlxDtype(resultType.getElementType());
    // MLX view function reinterprets the underlying data as a different type
    values.emplace(ToKey(op->getResult(0)), mlx::core::view(input_opt->get(), targetDtype));
    return true;
}

// Forward declaration for executing a region
bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx);

// Handler for stablehlo.return (region terminator)
bool HandleStablehloReturn(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "stablehlo.return");
}

// Handler for stablehlo.while
bool HandleWhile(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op);
    if (!whileOp) {
        MPS_LOG_ERROR("stablehlo.while: failed to cast\n");
        return false;
    }

    // Get initial values from operands
    std::vector<mlx::core::array> loopVars;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.while: operand not found in value map\n");
            return false;
        }
        loopVars.push_back(val_opt->get());
    }

    auto& condRegion = whileOp.getCond();
    auto& bodyRegion = whileOp.getBody();

    while (true) {
        // Execute condition region
        std::vector<mlx::core::array> condResults;
        if (!ExecuteRegion(condRegion, loopVars, condResults, ctx)) {
            MPS_LOG_ERROR("stablehlo.while: failed to execute cond region\n");
            return false;
        }

        if (condResults.size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: cond region should return 1 value, got %zu\n",
                          condResults.size());
            return false;
        }

        // Evaluate and check condition
        mlx::core::eval(condResults[0]);

        // Verify condition is a scalar boolean
        if (condResults[0].size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: condition must be a scalar, got size %zu\n",
                          condResults[0].size());
            return false;
        }

        if (!condResults[0].item<bool>()) {
            break;
        }

        // Execute body region
        std::vector<mlx::core::array> bodyResults;
        if (!ExecuteRegion(bodyRegion, loopVars, bodyResults, ctx)) {
            MPS_LOG_ERROR("stablehlo.while: failed to execute body region\n");
            return false;
        }

        if (bodyResults.size() != loopVars.size()) {
            MPS_LOG_ERROR("stablehlo.while: body returned %zu values, expected %zu\n",
                          bodyResults.size(), loopVars.size());
            return false;
        }

        // Update loop variables
        loopVars = std::move(bodyResults);
    }

    // Map final loop variables to results
    for (size_t i = 0; i < loopVars.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(loopVars[i]));
    }

    return true;
}

// Handler for stablehlo.custom_call
bool HandleCustomCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        MPS_LOG_ERROR("stablehlo.custom_call: failed to cast\n");
        return false;
    }

    auto callTargetName = customCallOp.getCallTargetName().str();

    // Handle Sharding annotation - just pass input through
    if (callTargetName == "Sharding") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: expected 1 input and 1 output\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: operand not found\n");
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    // Handle SPMDFullToShardShape and SPMDShardToFullShape - also pass-through for single device
    if (callTargetName == "SPMDFullToShardShape" || callTargetName == "SPMDShardToFullShape") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: operand not found\n", callTargetName.c_str());
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    MPS_LOG_ERROR("stablehlo.custom_call: unsupported target '%s'\n", callTargetName.c_str());
    return false;
}

// Handler for func.call
bool HandleCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!callOp) {
        MPS_LOG_ERROR("func.call: failed to cast\n");
        return false;
    }

    // Find the callee function in the module
    auto calleeName = callOp.getCallee();
    auto calleeFunc = ctx.module.lookupSymbol<mlir::func::FuncOp>(calleeName);
    if (!calleeFunc) {
        MPS_LOG_ERROR("func.call: callee function '%s' not found\n", calleeName.str().c_str());
        return false;
    }

    // Gather input arrays
    std::vector<mlx::core::array> callInputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("func.call: operand not found in value map\n");
            return false;
        }
        callInputs.push_back(val_opt->get());
    }

    // Execute the callee function
    std::vector<mlx::core::array> callOutputs;
    if (!ExecuteFunction(calleeFunc, callInputs, callOutputs, ctx)) {
        MPS_LOG_ERROR("func.call: failed to execute callee '%s'\n", calleeName.str().c_str());
        return false;
    }

    // Map results to the call operation's results
    if (callOutputs.size() != op->getNumResults()) {
        MPS_LOG_ERROR("func.call: result count mismatch\n");
        return false;
    }

    for (size_t i = 0; i < callOutputs.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(callOutputs[i]));
    }

    return true;
}

// Op dispatch table - initialized once
const std::unordered_map<std::string, OpHandler>& GetOpHandlers() {
    static const std::unordered_map<std::string, OpHandler> handlers = {
        // Arithmetic
        {"stablehlo.add", HandleAdd},
        {"stablehlo.subtract", HandleSubtract},
        {"stablehlo.multiply", HandleMultiply},
        {"stablehlo.negate", HandleNegate},
        {"stablehlo.abs", HandleAbs},
        {"stablehlo.exponential", HandleExp},
        {"stablehlo.log", HandleLog},
        {"stablehlo.sqrt", HandleSqrt},
        {"stablehlo.rsqrt", HandleRsqrt},
        {"stablehlo.log_plus_one", HandleLogPlusOne},
        {"stablehlo.maximum", HandleMaximum},
        {"stablehlo.minimum", HandleMinimum},
        {"stablehlo.divide", HandleDivide},
        {"stablehlo.floor", HandleFloor},
        {"stablehlo.sine", HandleSine},
        {"stablehlo.cosine", HandleCosine},
        {"stablehlo.clamp", HandleClamp},
        // Bitwise
        {"stablehlo.and", HandleAnd},
        {"stablehlo.or", HandleOr},
        {"stablehlo.xor", HandleXor},
        {"stablehlo.shift_left", HandleShiftLeft},
        {"stablehlo.shift_right_logical", HandleShiftRightLogical},
        // Comparison/selection
        {"stablehlo.compare", HandleCompare},
        {"stablehlo.select", HandleSelect},
        // Type/shape
        {"stablehlo.constant", HandleConstant},
        {"stablehlo.convert", HandleConvert},
        {"stablehlo.bitcast_convert", HandleBitcastConvert},
        {"stablehlo.reshape", HandleReshape},
        {"stablehlo.broadcast_in_dim", HandleBroadcastInDim},
        {"stablehlo.concatenate", HandleConcatenate},
        {"stablehlo.iota", HandleIota},
        {"stablehlo.slice", HandleSlice},
        {"stablehlo.dynamic_slice", HandleDynamicSlice},
        // Control flow
        {"func.return", HandleReturn},
        {"func.call", HandleCall},
        {"stablehlo.custom_call", HandleCustomCall},
        {"stablehlo.while", HandleWhile},
        {"stablehlo.return", HandleStablehloReturn},
    };
    return handlers;
}

// Dispatch a single operation using the handler table
bool DispatchOp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    std::string opName = op->getName().getStringRef().str();

    MPS_LOG_DEBUG("Dispatching op: %s\n", opName.c_str());

    const auto& handlers = GetOpHandlers();
    auto it = handlers.find(opName);
    if (it == handlers.end()) {
        MPS_LOG_ERROR("Unsupported op: %s\n", opName.c_str());
        return false;
    }

    try {
        return it->second(op, values, outputs, ctx);
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("Exception dispatching %s: %s\n", opName.c_str(), e.what());
        return false;
    }
}

// Execute a region (for while loops, etc.) with given inputs
bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx) {
    if (region.empty()) {
        MPS_LOG_ERROR("ExecuteRegion: empty region\n");
        return false;
    }

    ValueMap values;
    auto& block = region.front();

    // Map block arguments to input arrays
    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        if (argIdx >= args.size()) {
            MPS_LOG_ERROR("ExecuteRegion: not enough arguments\n");
            return false;
        }
        values.emplace(ToKey(arg), args[argIdx]);
        argIdx++;
    }

    // Walk operations and dispatch
    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, results, ctx)) {
            return false;
        }
    }

    return true;
}

// Execute a function with given inputs, collecting outputs
bool ExecuteFunction(mlir::func::FuncOp func, const std::vector<mlx::core::array>& inputs,
                     std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    ValueMap values;

    auto& block = func.front();
    size_t numArgs = block.getNumArguments();

    if (inputs.size() != numArgs) {
        MPS_LOG_ERROR("ExecuteFunction: input count mismatch: expected %zu, got %zu\n", numArgs,
                      inputs.size());
        return false;
    }

    // Map function arguments to input arrays
    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        values.emplace(ToKey(arg), inputs[argIdx]);
        argIdx++;
    }

    // Walk operations and dispatch
    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, outputs, ctx)) {
            return false;
        }
    }

    return true;
}

}  // namespace

std::unordered_set<std::string> GetSupportedOpNames() {
    const auto& handlers = GetOpHandlers();
    std::unordered_set<std::string> names;
    for (const auto& pair : handlers) {
        names.insert(pair.first);
    }
    return names;
}

std::unique_ptr<MlxExecutable> MlxExecutable::Create(mps::ParsedModule parsed_module) {
    auto executable = std::unique_ptr<MlxExecutable>(new MlxExecutable());

    if (!parsed_module.ok()) {
        executable->error_ = "Invalid parsed module";
        executable->valid_ = false;
        return executable;
    }

    // Check for unsupported ops
    if (!parsed_module.unsupported_ops.empty()) {
        executable->error_ = "Unsupported operations: ";
        for (size_t i = 0; i < parsed_module.unsupported_ops.size(); ++i) {
            if (i > 0)
                executable->error_ += ", ";
            executable->error_ += parsed_module.unsupported_ops[i];
        }
        executable->valid_ = false;
        return executable;
    }

    // Store the parsed module (keeps MLIR context alive)
    executable->parsed_module_ = std::move(parsed_module);

    // Extract output info from function return type
    auto funcType = executable->parsed_module_.entry_func.getFunctionType();
    executable->num_outputs_ = funcType.getNumResults();

    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        auto resultType = funcType.getResult(i);
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
            OutputInfo info;
            info.dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
            for (int64_t dim : tensorType.getShape()) {
                info.shape.push_back(dim);
            }
            executable->output_info_.push_back(info);
        } else {
            // Non-tensor output, use defaults
            OutputInfo info;
            info.dtype = PJRT_Buffer_Type_F32;
            executable->output_info_.push_back(info);
        }
    }

    executable->valid_ = true;
    MPS_LOG_DEBUG("Created MlxExecutable with %zu outputs\n", executable->num_outputs_);

    return executable;
}

bool MlxExecutable::IsValid() const {
    return valid_;
}

std::string MlxExecutable::error() const {
    return error_;
}

size_t MlxExecutable::num_outputs() const {
    return num_outputs_;
}

MlxExecuteResult MlxExecutable::Execute(const std::vector<MlxBuffer*>& inputs) {
    MlxExecuteResult result;

    if (!valid_) {
        MPS_LOG_ERROR("Cannot execute invalid executable: %s\n", error_.c_str());
        return result;
    }

    MPS_LOG_DEBUG("Executing with %zu inputs\n", inputs.size());

    // Validate inputs
    auto& block = parsed_module_.entry_func.front();
    size_t numArgs = block.getNumArguments();

    if (inputs.size() != numArgs) {
        MPS_LOG_ERROR("Input count mismatch: expected %zu, got %zu\n", numArgs, inputs.size());
        return result;
    }

    // Convert MlxBuffer inputs to mlx::core::array
    std::vector<mlx::core::array> inputArrays;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) {
            MPS_LOG_ERROR("Null input buffer at index %zu\n", i);
            return result;
        }
        inputArrays.push_back(inputs[i]->array());
    }

    // Set up execution context with module reference
    ExecContext ctx;
    ctx.module = *parsed_module_.module;

    // Execute the entry function
    std::vector<mlx::core::array> outputs;
    if (!ExecuteFunction(parsed_module_.entry_func, inputArrays, outputs, ctx)) {
        MPS_LOG_ERROR("Failed to execute entry function\n");
        return result;
    }

    // Validate output count
    if (outputs.size() != num_outputs_) {
        MPS_LOG_ERROR("Output count mismatch: expected %zu, got %zu\n", num_outputs_,
                      outputs.size());
        return result;
    }

    // Evaluate all outputs with error handling
    if (!outputs.empty()) {
        try {
            mlx::core::eval(outputs);
        } catch (const std::exception& e) {
            MPS_LOG_ERROR("MLX evaluation failed: %s\n", e.what());
            return result;
        }
    }

    // Wrap outputs in MlxBuffer
    for (auto& arr : outputs) {
        result.buffers.push_back(MlxBuffer::FromArray(std::move(arr)));
    }

    MPS_LOG_DEBUG("Execution complete with %zu outputs\n", result.buffers.size());

    return result;
}

}  // namespace jax_mps
