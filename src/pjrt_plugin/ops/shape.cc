// Shape, type conversion, and data movement op handlers.

#include <unordered_set>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.constant
bool HandleConstant(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto constOp = CastOp<mlir::stablehlo::ConstantOp>(op, "stablehlo.constant");
    if (!constOp)
        return false;

    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!attr) {
        MPS_LOG_ERROR("stablehlo.constant: value is not DenseElementsAttr\n");
        return false;
    }

    auto arr_opt = CreateArrayFromDenseAttr(attr);
    if (!arr_opt)
        return false;

    values.emplace(ToKey(op->getResult(0)), std::move(*arr_opt));
    return true;
}

// Handler for stablehlo.reshape
bool HandleReshape(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.reshape");
    if (!input)
        return false;

    auto newShape = GetResultShape(op, "stablehlo.reshape");
    if (!newShape)
        return false;

    values.emplace(ToKey(op->getResult(0)), mlx::core::reshape(*input, *newShape));
    return true;
}

// Handler for stablehlo.broadcast_in_dim
bool HandleBroadcastInDim(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto broadcastOp = CastOp<mlir::stablehlo::BroadcastInDimOp>(op, "stablehlo.broadcast_in_dim");
    if (!broadcastOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.broadcast_in_dim");
    if (!input)
        return false;

    auto outputShape = GetResultShape(op, "stablehlo.broadcast_in_dim");
    if (!outputShape)
        return false;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    // Validate broadcast dimensions are in bounds
    for (int64_t dim : broadcastDims) {
        if (dim < 0 || static_cast<size_t>(dim) >= outputShape->size()) {
            MPS_LOG_ERROR("stablehlo.broadcast_in_dim: dimension %lld out of bounds [0, %zu)\n",
                          dim, outputShape->size());
            return false;
        }
    }

    // Build the intermediate shape with 1s for non-broadcast dims
    mlx::core::Shape intermediateShape(outputShape->size(), 1);
    for (size_t i = 0; i < broadcastDims.size(); ++i) {
        int64_t dim = broadcastDims[i];
        if (static_cast<int>(i) < input->ndim()) {
            intermediateShape[dim] = input->shape(static_cast<int>(i));
        }
    }

    auto reshaped = mlx::core::reshape(*input, intermediateShape);
    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, *outputShape));
    return true;
}

// Handler for stablehlo.concatenate
bool HandleConcatenate(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto concatOp = CastOp<mlir::stablehlo::ConcatenateOp>(op, "stablehlo.concatenate");
    if (!concatOp)
        return false;

    std::vector<mlx::core::array> inputs;
    for (auto operand : op->getOperands()) {
        auto* val = RequireValue(values, operand, "stablehlo.concatenate");
        if (!val)
            return false;
        inputs.push_back(*val);
    }

    auto axis = concatOp.getDimension();
    values.emplace(ToKey(op->getResult(0)), mlx::core::concatenate(inputs, static_cast<int>(axis)));
    return true;
}

// Handler for stablehlo.convert (type conversion)
bool HandleConvert(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.convert");
    if (!input)
        return false;

    auto targetDtype = GetResultDtype(op, "stablehlo.convert");
    if (!targetDtype)
        return false;

    auto result = *input;
    auto srcDtype = input->dtype();

    // Workaround for https://github.com/ml-explore/mlx/issues/3338
    // MLX compile() incorrectly fuses chained casts when converting from an unsigned
    // integer type to a signed type followed by float promotion (e.g.,
    // uint16->int32->float32 gets fused into a single kernel that misreads bytes).
    // Work around by widening to the unsigned type of the target width, then using
    // view() to reinterpret bits as signed. view() acts as a compile barrier (it is
    // not fused with subsequent astype) and has zero runtime cost.
    bool srcUnsigned = mlx::core::issubdtype(srcDtype, mlx::core::unsignedinteger);
    bool tgtSigned = mlx::core::issubdtype(*targetDtype, mlx::core::signedinteger);
    if (srcUnsigned && tgtSigned) {
        mlx::core::Dtype targetUnsigned = mlx::core::uint32;
        if (*targetDtype == mlx::core::int8)
            targetUnsigned = mlx::core::uint8;
        else if (*targetDtype == mlx::core::int16)
            targetUnsigned = mlx::core::uint16;
        else if (*targetDtype == mlx::core::int64)
            targetUnsigned = mlx::core::uint64;
        result = mlx::core::astype(result, targetUnsigned);
        values.emplace(ToKey(op->getResult(0)), mlx::core::view(result, *targetDtype));
        return true;
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(result, *targetDtype));
    return true;
}

// Handler for stablehlo.iota
bool HandleIota(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto iotaOp = CastOp<mlir::stablehlo::IotaOp>(op, "stablehlo.iota");
    if (!iotaOp)
        return false;

    auto shape = GetResultShape(op, "stablehlo.iota");
    if (!shape)
        return false;

    auto dtype = GetResultDtype(op, "stablehlo.iota");
    if (!dtype)
        return false;

    uint64_t iotaDim = iotaOp.getIotaDimension();
    int dimSize = (*shape)[iotaDim];
    auto iota1d = mlx::core::arange(0, dimSize, *dtype);

    // Reshape to have 1s everywhere except the iota dimension
    mlx::core::Shape reshapeShape(shape->size(), 1);
    reshapeShape[iotaDim] = dimSize;
    auto reshaped = mlx::core::reshape(iota1d, reshapeShape);

    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, *shape));
    return true;
}

// Handler for stablehlo.reverse
bool HandleReverse(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto reverseOp = CastOp<mlir::stablehlo::ReverseOp>(op, "stablehlo.reverse");
    if (!reverseOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.reverse");
    if (!input)
        return false;

    auto dimensions = reverseOp.getDimensions();
    auto ndim = static_cast<int>(input->ndim());

    // Build set of dimensions to reverse
    std::unordered_set<int64_t> reverseDims(dimensions.begin(), dimensions.end());

    // Use slice with negative strides to reverse dimensions
    mlx::core::Shape starts(ndim, 0);
    mlx::core::Shape stops;
    mlx::core::Shape steps(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        int dimSize = input->shape(i);
        if (reverseDims.count(i)) {
            starts[i] = dimSize - 1;
            stops.push_back(-dimSize - 1);
            steps[i] = -1;
        } else {
            stops.push_back(dimSize);
        }
    }

    auto result = mlx::core::slice(*input, starts, stops, steps);
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
    auto transposeOp = CastOp<mlir::stablehlo::TransposeOp>(op, "stablehlo.transpose");
    if (!transposeOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.transpose");
    if (!input)
        return false;

    auto axes = ToIntVec(transposeOp.getPermutation());
    values.emplace(ToKey(op->getResult(0)), mlx::core::transpose(*input, axes));
    return true;
}

// Handler for stablehlo.bitcast_convert
bool HandleBitcastConvert(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.bitcast_convert");
    if (!input)
        return false;

    auto targetDtype = GetResultDtype(op, "stablehlo.bitcast_convert");
    auto targetShape = GetResultShape(op, "stablehlo.bitcast_convert");
    if (!targetDtype || !targetShape)
        return false;

    // MLX view reinterprets data as a different type (adjusts last dim).
    // StableHLO bitcast_convert can also change rank (e.g., 2x2xi32 -> 2xi64),
    // so reshape to the expected output shape afterward.
    auto result = mlx::core::view(*input, *targetDtype);
    if (result.shape() != *targetShape) {
        result = mlx::core::reshape(result, *targetShape);
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
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
