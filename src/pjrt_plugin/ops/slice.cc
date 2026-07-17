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
    auto sliceOp = CastOp<mlir::stablehlo::SliceOp>(op, "stablehlo.slice");
    if (!sliceOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.slice");
    if (!input)
        return false;

    auto starts = ToShape(sliceOp.getStartIndices());
    auto stops = ToShape(sliceOp.getLimitIndices());
    auto steps = ToShape(sliceOp.getStrides());

    values.emplace(ToKey(op->getResult(0)), mlx::core::slice(*input, starts, stops, steps));
    return true;
}

// Handler for stablehlo.dynamic_slice
bool HandleDynamicSlice(mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dynamicSliceOp = CastOp<mlir::stablehlo::DynamicSliceOp>(op, "stablehlo.dynamic_slice");
    if (!dynamicSliceOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.dynamic_slice");
    if (!input)
        return false;

    auto sliceSizes = dynamicSliceOp.getSliceSizes();

    // Lower to MLX's dynamic slice(a, start, axes, slice_size): one lazy
    // primitive whose eval is a view/copy of just the slice, instead of the
    // previous per-axis take chain (ndim gathers plus index arithmetic per
    // call — a real cost inside while-loop bodies, where scan reads one
    // xs[i] every iteration).
    const int ndim = static_cast<int>(input->ndim());
    if (ndim == 0) {
        values.emplace(ToKey(op->getResult(0)), *input);
        return true;
    }

    std::vector<mlx::core::array> startParts;
    std::vector<int> axes;
    mlx::core::Shape outSize;
    startParts.reserve(static_cast<size_t>(ndim));
    axes.reserve(static_cast<size_t>(ndim));
    for (int d = 0; d < ndim; ++d) {
        auto* idx = RequireValue(values, op->getOperand(static_cast<unsigned>(d + 1)),
                                 "stablehlo.dynamic_slice");
        if (!idx)
            return false;
        const int size = static_cast<int>(sliceSizes[static_cast<unsigned>(d)]);
        const int dimSize = input->shape(d);
        // Clamp start per StableHLO spec: max(0, min(start, dim_size - size)).
        auto start =
            mlx::core::clip(mlx::core::astype(mlx::core::reshape(*idx, {1}), mlx::core::int32),
                            mlx::core::array(0), mlx::core::array(dimSize - size));
        startParts.push_back(std::move(start));
        axes.push_back(d);
        outSize.push_back(size);
    }
    auto startArr = startParts.size() == 1 ? startParts[0] : mlx::core::concatenate(startParts, 0);
    values.emplace(ToKey(op->getResult(0)), mlx::core::slice(*input, std::move(startArr),
                                                             std::move(axes), std::move(outSize)));
    return true;
}

// Handler for stablehlo.dynamic_update_slice
bool HandleDynamicUpdateSlice(mlir::Operation* op, ValueMap& values,
                              std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dusOp =
        CastOp<mlir::stablehlo::DynamicUpdateSliceOp>(op, "stablehlo.dynamic_update_slice");
    if (!dusOp)
        return false;

    auto* operand = RequireValue(values, dusOp.getOperand(), "stablehlo.dynamic_update_slice");
    auto* update = RequireValue(values, dusOp.getUpdate(), "stablehlo.dynamic_update_slice");
    if (!operand || !update)
        return false;

    // Empty update is a no-op
    if (update->size() == 0) {
        values.emplace(ToKey(op->getResult(0)), *operand);
        return true;
    }

    // A rank-0 update replaces the scalar operand.
    if (operand->ndim() == 0) {
        values.emplace(ToKey(op->getResult(0)), *update);
        return true;
    }

    // Use MLX's native dynamic slice update so only the update region is written.
    // Clamp starts explicitly to preserve StableHLO dynamic_update_slice semantics.
    std::vector<mlx::core::array> start_indices;
    std::vector<int> axes;
    start_indices.reserve(operand->ndim());
    axes.reserve(operand->ndim());
    for (int d = 0; d < static_cast<int>(operand->ndim()); ++d) {
        auto* start_val =
            RequireValue(values, dusOp.getStartIndices()[d], "stablehlo.dynamic_update_slice");
        if (!start_val)
            return false;
        auto start_idx = mlx::core::astype(mlx::core::reshape(*start_val, {}), mlx::core::int32);

        int op_size = operand->shape(d);
        int up_size = update->shape(d);

        // Clamp start index: max(0, min(start, op_size - up_size))
        start_idx =
            mlx::core::maximum(mlx::core::array(0),
                               mlx::core::minimum(start_idx, mlx::core::array(op_size - up_size)));
        start_indices.push_back(mlx::core::reshape(start_idx, {1}));
        axes.push_back(d);
    }

    auto start = mlx::core::concatenate(start_indices, 0);
    values.emplace(ToKey(op->getResult(0)),
                   mlx::core::slice_update(*operand, *update, start, axes));
    return true;
}

// Handler for stablehlo.pad
bool HandlePad(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto padOp = CastOp<mlir::stablehlo::PadOp>(op, "stablehlo.pad");
    if (!padOp)
        return false;

    auto* input = RequireValue(values, padOp.getOperand(), "stablehlo.pad");
    auto* padValue = RequireValue(values, padOp.getPaddingValue(), "stablehlo.pad");
    if (!input || !padValue)
        return false;

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
        auto result = *input;
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
            auto dilated = mlx::core::full(newShape, *padValue);

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
                       ApplyEdgePadding(result, edgePaddingLow, edgePaddingHigh, *padValue));
        return true;
    }

    values.emplace(ToKey(op->getResult(0)),
                   ApplyEdgePadding(*input, edgePaddingLow, edgePaddingHigh, *padValue));
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
