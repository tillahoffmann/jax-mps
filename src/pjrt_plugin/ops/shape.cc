// Shape, type conversion, and data movement op handlers.

#include <unordered_set>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

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

// Handler for stablehlo.reverse
bool HandleReverse(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(op);
    if (!reverseOp) {
        MPS_LOG_ERROR("stablehlo.reverse: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.reverse: operand not found in value map\n");
        return false;
    }

    auto dimensions = reverseOp.getDimensions();
    auto& input = input_opt->get();
    auto ndim = static_cast<int>(input.ndim());

    // Build set of dimensions to reverse
    std::unordered_set<int64_t> reverseDims(dimensions.begin(), dimensions.end());

    // Use slice with negative strides to reverse dimensions
    mlx::core::Shape starts(ndim, 0);
    mlx::core::Shape stops;
    mlx::core::Shape steps(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        int dimSize = input.shape(i);
        if (reverseDims.count(i)) {
            // Reverse: start at end, go to beginning with step -1
            starts[i] = dimSize - 1;
            stops.push_back(-dimSize - 1);  // Past the beginning
            steps[i] = -1;
        } else {
            stops.push_back(dimSize);
        }
    }

    auto result = mlx::core::slice(input, starts, stops, steps);
    // Force contiguous for complex types - MLX's non-contiguous views (from negative strides)
    // can produce incorrect results for complex arrays in subsequent operations
    if (result.dtype() == mlx::core::complex64) {
        result = mlx::core::contiguous(result);
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.transpose
bool HandleTranspose(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        MPS_LOG_ERROR("stablehlo.transpose: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.transpose: operand not found in value map\n");
        return false;
    }

    auto permAttr = transposeOp.getPermutation();
    std::vector<int> axes;
    for (int64_t dim : permAttr) {
        axes.push_back(static_cast<int>(dim));
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::transpose(input_opt->get(), axes));
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

}  // namespace

void RegisterShapeHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.constant", HandleConstant});
    handlers.insert({"stablehlo.convert", HandleConvert});
    handlers.insert({"stablehlo.bitcast_convert", HandleBitcastConvert});
    handlers.insert({"stablehlo.reshape", HandleReshape});
    handlers.insert({"stablehlo.broadcast_in_dim", HandleBroadcastInDim});
    handlers.insert({"stablehlo.concatenate", HandleConcatenate});
    handlers.insert({"stablehlo.transpose", HandleTranspose});
    handlers.insert({"stablehlo.reverse", HandleReverse});
    handlers.insert({"stablehlo.iota", HandleIota});
}

}  // namespace jax_mps
