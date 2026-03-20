// Slice, dynamic slice, pad op handlers.

#include <algorithm>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Apply edge padding that may include negative values (trimming).
// Negative low/high means slice from that side; positive means pad.
mlx::core::array ApplyEdgePadding(const mlx::core::array& input,
                                  llvm::ArrayRef<int64_t> edgePaddingLow,
                                  llvm::ArrayRef<int64_t> edgePaddingHigh,
                                  const mlx::core::array& padValue) {
    auto ndim = edgePaddingLow.size();
    auto result = input;

    // Trim (slice) for any negative padding values.
    bool hasNeg = false;
    for (size_t i = 0; i < ndim; ++i) {
        if (edgePaddingLow[i] < 0 || edgePaddingHigh[i] < 0) {
            hasNeg = true;
            break;
        }
    }
    if (hasNeg) {
        mlx::core::Shape starts;
        mlx::core::Shape stops;
        mlx::core::Shape strides;
        auto shape = result.shape();
        for (size_t i = 0; i < ndim; ++i) {
            int64_t lo = edgePaddingLow[i];
            int64_t hi = edgePaddingHigh[i];
            starts.push_back(static_cast<int>(lo < 0 ? -lo : 0));
            stops.push_back(static_cast<int>(shape[i] + (hi < 0 ? hi : 0)));
            strides.push_back(1);
        }
        result = mlx::core::slice(result, starts, stops, strides);
    }

    // Pad with clamped-to-zero values.
    std::vector<std::pair<int, int>> padWidths;
    padWidths.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        padWidths.emplace_back(static_cast<int>(std::max<int64_t>(edgePaddingLow[i], 0)),
                               static_cast<int>(std::max<int64_t>(edgePaddingHigh[i], 0)));
    }
    return mlx::core::pad(result, padWidths, padValue);
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

    auto& input = input_opt->get();
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

    values.emplace(ToKey(op->getResult(0)), mlx::core::slice(input, starts, stops, steps));
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

    // Use purely functional MLX ops (no eval) so this works inside mlx::core::compile() tracing.
    // For each dimension, compute indices as start + arange(size) and use take().
    auto result = input;
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
        auto idx_opt = GetValue(values, op->getOperand(i));
        if (!idx_opt) {
            MPS_LOG_ERROR("stablehlo.dynamic_slice: start index operand not found\n");
            return false;
        }
        auto start_idx = mlx::core::astype(idx_opt->get(), mlx::core::int32);
        int size = static_cast<int>(sliceSizes[i - 1]);
        int axis = static_cast<int>(i - 1);
        int dim_size = input.shape(axis);

        // Clamp start index per StableHLO spec: max(0, min(start, dim_size - size))
        start_idx = mlx::core::maximum(
            mlx::core::array(0), mlx::core::minimum(start_idx, mlx::core::array(dim_size - size)));

        // Create indices: clamped_start + [0, 1, 2, ..., size-1]
        auto offsets = mlx::core::arange(0, size, mlx::core::int32);
        auto indices = mlx::core::add(start_idx, offsets);
        result = mlx::core::take(result, indices, axis);
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.dynamic_update_slice
bool HandleDynamicUpdateSlice(mlir::Operation* op, ValueMap& values,
                              std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dusOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(op);
    if (!dusOp) {
        MPS_LOG_ERROR("stablehlo.dynamic_update_slice: failed to cast\n");
        return false;
    }

    auto operand_opt = GetValue(values, dusOp.getOperand());
    auto update_opt = GetValue(values, dusOp.getUpdate());
    if (!operand_opt || !update_opt) {
        MPS_LOG_ERROR("stablehlo.dynamic_update_slice: operand not found in value map\n");
        return false;
    }

    auto& operand = operand_opt->get();
    auto& update = update_opt->get();

    // Empty update is a no-op
    if (update.size() == 0) {
        values.emplace(ToKey(op->getResult(0)), operand);
        return true;
    }

    // Use purely functional MLX ops (no eval) so this works inside mlx::core::compile() tracing.
    auto gathered = update;
    auto combined_mask = mlx::core::array(true);
    for (int d = 0; d < static_cast<int>(operand.ndim()); ++d) {
        auto idx_opt = GetValue(values, dusOp.getStartIndices()[d]);
        if (!idx_opt) {
            MPS_LOG_ERROR("stablehlo.dynamic_update_slice: start index not found\n");
            return false;
        }
        auto start_idx =
            mlx::core::astype(mlx::core::reshape(idx_opt->get(), {}), mlx::core::int32);

        int op_size = operand.shape(d);
        int up_size = update.shape(d);

        // Clamp start index: max(0, min(start, op_size - up_size))
        start_idx =
            mlx::core::maximum(mlx::core::array(0),
                               mlx::core::minimum(start_idx, mlx::core::array(op_size - up_size)));

        // Relative position of each operand index w.r.t. the update region
        auto arange_d = mlx::core::arange(0, op_size, mlx::core::int32);
        auto relative = mlx::core::subtract(arange_d, start_idx);

        // Per-dimension mask: position is inside update region
        auto mask_d =
            mlx::core::logical_and(mlx::core::greater_equal(relative, mlx::core::array(0)),
                                   mlx::core::less(relative, mlx::core::array(up_size)));

        // Reshape mask for broadcasting: [1, ..., op_size, ..., 1]
        mlx::core::Shape shape(operand.ndim(), 1);
        shape[d] = op_size;
        mask_d = mlx::core::reshape(mask_d, shape);
        combined_mask = mlx::core::logical_and(combined_mask, mask_d);

        // Clamp relative indices for gathering from update
        auto clamped = mlx::core::clip(relative, mlx::core::array(0),
                                       mlx::core::array(std::max(0, up_size - 1)));
        gathered = mlx::core::take(gathered, clamped, d);
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::where(combined_mask, gathered, operand));
    return true;
}

// Handler for stablehlo.pad
bool HandlePad(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
    if (!padOp) {
        MPS_LOG_ERROR("stablehlo.pad: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, padOp.getOperand());
    auto padValue_opt = GetValue(values, padOp.getPaddingValue());
    if (!input_opt || !padValue_opt) {
        MPS_LOG_ERROR("stablehlo.pad: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto& padValue = padValue_opt->get();
    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();

    // Check for interior padding
    bool hasInterior = false;
    for (int64_t p : interiorPadding) {
        if (p != 0) {
            hasInterior = true;
            break;
        }
    }

    if (hasInterior) {
        // Interior padding: insert `p` copies of padValue between each pair of
        // existing elements along each axis, then apply edge padding.
        auto result = input;
        auto ndim = edgePaddingLow.size();

        for (size_t axis = 0; axis < ndim; ++axis) {
            auto p = interiorPadding[axis];
            if (p <= 0)
                continue;

            auto shape = result.shape();
            auto axisSize = shape[axis];
            if (axisSize <= 1)
                continue;

            auto newAxisSize = static_cast<int32_t>(axisSize + (axisSize - 1) * p);

            mlx::core::Shape newShape(shape.begin(), shape.end());
            newShape[axis] = newAxisSize;

            // Create the dilated array filled with padValue.
            auto dilated = mlx::core::full(newShape, padValue);

            // Build indices for the original elements: 0, p+1, 2*(p+1), ...
            std::vector<int32_t> idxVals(axisSize);
            for (int32_t i = 0; i < axisSize; ++i) {
                idxVals[i] = i * static_cast<int32_t>(p + 1);
            }
            auto indices = mlx::core::array(idxVals.data(), {axisSize}, mlx::core::int32);

            // Scatter original values at strided positions along this axis.
            result = mlx::core::put_along_axis(dilated,
                                               mlx::core::reshape(indices,
                                                                  [&]() {
                                                                      mlx::core::Shape s(ndim, 1);
                                                                      s[axis] = axisSize;
                                                                      return s;
                                                                  }()),
                                               result, static_cast<int>(axis));
        }

        values.emplace(ToKey(op->getResult(0)),
                       ApplyEdgePadding(result, edgePaddingLow, edgePaddingHigh, padValue));
        return true;
    }

    values.emplace(ToKey(op->getResult(0)),
                   ApplyEdgePadding(input, edgePaddingLow, edgePaddingHigh, padValue));
    return true;
}

}  // namespace

void RegisterSliceHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.slice", HandleSlice});
    handlers.insert({"stablehlo.dynamic_slice", HandleDynamicSlice});
    handlers.insert({"stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice});
    handlers.insert({"stablehlo.pad", HandlePad});
}

}  // namespace jax_mps
