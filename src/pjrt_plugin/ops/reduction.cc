// Reduction op handlers (reduce, reduce_window, select_and_scatter).

#include <algorithm>
#include <limits>
#include <tuple>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Helper to detect reduction type by analyzing the body region
enum class ReduceType { Sum, Max, Min, Prod, And, Or, Unknown };

ReduceType DetectReduceType(mlir::Region& body) {
    if (body.empty())
        return ReduceType::Unknown;

    // Only match bodies with exactly one "real" op (plus the return).
    // Multi-op bodies (e.g., logaddexp: max + sub + exp + log1p + add + select)
    // must not be misidentified as a simple reduction.
    auto& block = body.front();
    int realOpCount = 0;
    ReduceType detected = ReduceType::Unknown;
    for (auto& op : block.getOperations()) {
        auto opName = op.getName().getStringRef();
        if (opName == "stablehlo.return")
            continue;
        ++realOpCount;
        if (realOpCount > 1)
            return ReduceType::Unknown;
        if (opName == "stablehlo.add")
            detected = ReduceType::Sum;
        else if (opName == "stablehlo.maximum")
            detected = ReduceType::Max;
        else if (opName == "stablehlo.minimum")
            detected = ReduceType::Min;
        else if (opName == "stablehlo.multiply")
            detected = ReduceType::Prod;
        else if (opName == "stablehlo.and")
            detected = ReduceType::And;
        else if (opName == "stablehlo.or")
            detected = ReduceType::Or;
    }
    return detected;
}

// Detect argmax/argmin pattern in a 2-input reduce body.
// Returns 1 for argmax, -1 for argmin, 0 for neither.
int DetectArgReducePattern(mlir::Region& body) {
    if (body.empty())
        return 0;

    auto& block = body.front();
    if (block.getNumArguments() != 4)
        return 0;

    for (auto& op : block.getOperations()) {
        if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
            auto lhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getLhs());
            auto rhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getRhs());
            if (!lhsArg || !rhsArg)
                continue;
            // Block args for 2-input reduce are interleaved: (val0, idx0, val1, idx1)
            // So comparing values means args 0 vs 2
            if (lhsArg.getArgNumber() == 0 && rhsArg.getArgNumber() == 2) {
                auto dir = cmpOp.getComparisonDirection();
                if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                    dir == mlir::stablehlo::ComparisonDirection::GE)
                    return 1;  // argmax
                if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                    dir == mlir::stablehlo::ComparisonDirection::LE)
                    return -1;  // argmin
            }
        }
    }
    return 0;
}

// Handler for stablehlo.reduce
bool HandleReduce(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto reduceOp = CastOp<mlir::stablehlo::ReduceOp>(op, "stablehlo.reduce");
    if (!reduceOp)
        return false;

    auto axes = ToIntVec(reduceOp.getDimensions());
    auto& body = reduceOp.getBody();
    size_t numInputs = reduceOp.getInputs().size();

    // Special case: argmax/argmin pattern (2 inputs: values + indices)
    if (numInputs == 2) {
        int argDir = DetectArgReducePattern(body);
        if (argDir != 0) {
            auto* input = RequireValue(values, reduceOp.getInputs()[0], "stablehlo.reduce");
            if (!input)
                return false;

            if (axes.size() == 1) {
                auto idx = (argDir > 0) ? mlx::core::argmax(*input, axes[0], /*keepdims=*/false)
                                        : mlx::core::argmin(*input, axes[0], /*keepdims=*/false);
                auto val =
                    (argDir > 0) ? mlx::core::max(*input, axes) : mlx::core::min(*input, axes);

                values.emplace(ToKey(op->getResult(0)), std::move(val));
                values.emplace(ToKey(op->getResult(1)), mlx::core::astype(idx, mlx::core::int32));
                return true;
            }
        }
    }

    ReduceType reduceType = DetectReduceType(body);

    for (size_t i = 0; i < numInputs; ++i) {
        auto* input = RequireValue(values, reduceOp.getInputs()[i], "stablehlo.reduce");
        if (!input)
            return false;

        std::optional<mlx::core::array> result;
        switch (reduceType) {
            case ReduceType::Sum:
                result = mlx::core::sum(*input, axes);
                break;
            case ReduceType::Max:
                result = mlx::core::max(*input, axes);
                break;
            case ReduceType::Min:
                result = mlx::core::min(*input, axes);
                break;
            case ReduceType::Prod:
                result = mlx::core::prod(*input, axes);
                break;
            case ReduceType::And:
                if (input->dtype() != mlx::core::bool_) {
                    MPS_LOG_ERROR(
                        "stablehlo.reduce: bitwise And reduction not supported for non-bool "
                        "types\n");
                    return false;
                }
                result = mlx::core::all(*input, axes);
                break;
            case ReduceType::Or:
                if (input->dtype() != mlx::core::bool_) {
                    MPS_LOG_ERROR(
                        "stablehlo.reduce: bitwise Or reduction not supported for non-bool "
                        "types\n");
                    return false;
                }
                result = mlx::core::any(*input, axes);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.reduce: unsupported reduction type\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(i)), std::move(*result));
    }

    return true;
}

// Handler for stablehlo.reduce_window (cumulative ops and pooling)
bool HandleReduceWindow(mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto rwOp = CastOp<mlir::stablehlo::ReduceWindowOp>(op, "stablehlo.reduce_window");
    if (!rwOp)
        return false;

    if (rwOp.getInputs().size() != 1 || rwOp.getInitValues().size() != 1 ||
        rwOp->getNumResults() != 1) {
        MPS_LOG_ERROR("stablehlo.reduce_window: only single-input reduce_window is supported\n");
        return false;
    }

    auto* input = RequireValue(values, rwOp.getInputs()[0], "stablehlo.reduce_window");
    if (!input)
        return false;

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(rwOp.getInputs()[0].getType());
    if (!inputType) {
        MPS_LOG_ERROR("stablehlo.reduce_window: unranked input\n");
        return false;
    }
    auto inputShape = inputType.getShape();
    auto rank = static_cast<int64_t>(inputShape.size());

    // Handle scalar (0-dimensional) inputs: reduce_window on a scalar is identity.
    if (rank == 0) {
        values.emplace(ToKey(op->getResult(0)), *input);
        return true;
    }

    auto windowDims = rwOp.getWindowDimensions();
    auto stridesOpt = rwOp.getWindowStrides();
    auto baseDilOpt = rwOp.getBaseDilations();
    auto winDilOpt = rwOp.getWindowDilations();
    auto paddingAttr = rwOp.getPaddingAttr();

    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> winDil(rank, 1);
    std::vector<int64_t> padLow(rank, 0);
    std::vector<int64_t> padHigh(rank, 0);

    if (stridesOpt) {
        auto s = *stridesOpt;
        for (int64_t i = 0; i < rank; i++)
            strides[i] = s[i];
    }
    if (winDilOpt) {
        auto d = *winDilOpt;
        for (int64_t i = 0; i < rank; i++)
            winDil[i] = d[i];
    }
    if (paddingAttr) {
        auto vals = paddingAttr.getValues<int64_t>();
        for (int64_t i = 0; i < rank; i++) {
            padLow[i] = vals[{static_cast<uint64_t>(i), 0}];
            padHigh[i] = vals[{static_cast<uint64_t>(i), 1}];
        }
    }

    auto allOnes = [](const auto& opt) {
        if (!opt)
            return true;
        return std::all_of(opt->begin(), opt->end(), [](auto v) { return v == 1; });
    };

    bool allStridesOne = allOnes(stridesOpt);
    bool allBaseDilOne = allOnes(baseDilOpt);
    bool allWinDilOne = allOnes(winDilOpt);

    ReduceType reduceType = DetectReduceType(rwOp.getBody());

    // ---------- Tier 1: Cumulative pattern ----------
    if (allStridesOne && allBaseDilOne && allWinDilOne) {
        int64_t cumAxis = -1;
        bool isCumulative = true;
        for (int64_t i = 0; i < rank; i++) {
            if (windowDims[i] == 1)
                continue;
            if (windowDims[i] == inputShape[i] && cumAxis == -1) {
                cumAxis = i;
            } else {
                isCumulative = false;
                break;
            }
        }

        if (isCumulative && cumAxis >= 0) {
            int64_t axisSize = inputShape[cumAxis];
            bool reverse = false;
            bool inclusive = true;

            if (padLow[cumAxis] == axisSize - 1 && padHigh[cumAxis] == 0) {
                reverse = false;
                inclusive = true;
            } else if (padLow[cumAxis] == 0 && padHigh[cumAxis] == axisSize - 1) {
                reverse = true;
                inclusive = true;
            } else if (padLow[cumAxis] == axisSize && padHigh[cumAxis] == -1) {
                reverse = false;
                inclusive = false;
            } else if (padLow[cumAxis] == -1 && padHigh[cumAxis] == axisSize) {
                reverse = true;
                inclusive = false;
            } else {
                MPS_LOG_ERROR("stablehlo.reduce_window: unsupported cumulative padding pattern\n");
                return false;
            }

            for (int64_t i = 0; i < rank; i++) {
                if (i == cumAxis)
                    continue;
                if (padLow[i] != 0 || padHigh[i] != 0) {
                    MPS_LOG_ERROR(
                        "stablehlo.reduce_window: non-zero padding on non-cumulative axis\n");
                    return false;
                }
            }

            auto axis = static_cast<int>(cumAxis);
            std::optional<mlx::core::array> result;
            switch (reduceType) {
                case ReduceType::Sum:
                    result = mlx::core::cumsum(*input, axis, reverse, inclusive);
                    break;
                case ReduceType::Prod:
                    result = mlx::core::cumprod(*input, axis, reverse, inclusive);
                    break;
                case ReduceType::Max:
                    result = mlx::core::cummax(*input, axis, reverse, inclusive);
                    break;
                case ReduceType::Min:
                    result = mlx::core::cummin(*input, axis, reverse, inclusive);
                    break;
                default:
                    MPS_LOG_ERROR("stablehlo.reduce_window: unsupported cumulative reduce type\n");
                    return false;
            }

            values.emplace(ToKey(op->getResult(0)), std::move(*result));
            return true;
        }
    }

    // ---------- Tier 2: Pooling pattern ----------
    if (!allBaseDilOne) {
        MPS_LOG_ERROR("stablehlo.reduce_window: base dilations not supported\n");
        return false;
    }

    if (reduceType != ReduceType::Max && reduceType != ReduceType::Sum &&
        reduceType != ReduceType::Min) {
        MPS_LOG_ERROR("stablehlo.reduce_window: pooling supports max/sum/min only\n");
        return false;
    }

    // Pad input if needed.
    mlx::core::array padded = *input;
    bool needsPad = false;
    for (int64_t i = 0; i < rank; i++) {
        if (padLow[i] != 0 || padHigh[i] != 0) {
            needsPad = true;
            break;
        }
    }
    if (needsPad) {
        auto* init = RequireValue(values, rwOp.getInitValues()[0], "stablehlo.reduce_window");
        if (!init)
            return false;

        std::vector<std::pair<int, int>> padWidth(rank);
        for (int64_t i = 0; i < rank; i++) {
            padWidth[i] = {static_cast<int>(padLow[i]), static_cast<int>(padHigh[i])};
        }
        padded = mlx::core::pad(*input, padWidth, *init);
    }

    auto paddedShape = padded.shape();
    std::vector<int> outShape(rank);
    for (int64_t i = 0; i < rank; i++) {
        int64_t effWin = (windowDims[i] - 1) * winDil[i] + 1;
        outShape[i] = static_cast<int>((paddedShape[i] - effWin) / strides[i] + 1);
    }

    mlx::core::Shape viewShape;
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(outShape[i]);
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(static_cast<int32_t>(windowDims[i]));

    std::vector<int64_t> elemStrides(rank);
    elemStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        elemStrides[i] = elemStrides[i + 1] * paddedShape[i + 1];
    }

    mlx::core::Strides viewStrides;
    for (int64_t i = 0; i < rank; i++) {
        viewStrides.push_back(strides[i] * elemStrides[i]);
    }
    for (int64_t i = 0; i < rank; i++) {
        viewStrides.push_back(winDil[i] * elemStrides[i]);
    }

    auto windowed = mlx::core::as_strided(padded, viewShape, viewStrides, 0);

    std::vector<int> reduceAxes;
    for (int64_t i = rank; i < 2 * rank; i++) {
        reduceAxes.push_back(static_cast<int>(i));
    }

    mlx::core::array result = [&]() {
        if (reduceType == ReduceType::Max)
            return mlx::core::max(windowed, reduceAxes);
        if (reduceType == ReduceType::Min)
            return mlx::core::min(windowed, reduceAxes);
        return mlx::core::sum(windowed, reduceAxes);
    }();

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.select_and_scatter (backward pass of max/min pooling)
bool HandleSelectAndScatter(mlir::Operation* op, ValueMap& values,
                            std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto ssOp = CastOp<mlir::stablehlo::SelectAndScatterOp>(op, "stablehlo.select_and_scatter");
    if (!ssOp)
        return false;

    auto* operand = RequireValue(values, ssOp.getOperand(), "stablehlo.select_and_scatter");
    auto* source = RequireValue(values, ssOp.getSource(), "stablehlo.select_and_scatter");
    auto* init = RequireValue(values, ssOp.getInitValue(), "stablehlo.select_and_scatter");
    if (!operand || !source || !init)
        return false;

    auto rank = static_cast<int64_t>(operand->ndim());

    auto windowDimsOpt = ssOp.getWindowDimensions();
    if (!windowDimsOpt) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: window_dimensions required\n");
        return false;
    }
    auto windowDims = *windowDimsOpt;

    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> padLow(rank, 0);
    std::vector<int64_t> padHigh(rank, 0);

    if (auto s = ssOp.getWindowStrides()) {
        for (int64_t i = 0; i < rank; i++)
            strides[i] = (*s)[i];
    }
    if (auto p = ssOp.getPaddingAttr()) {
        auto vals = p.getValues<int64_t>();
        for (int64_t i = 0; i < rank; i++) {
            padLow[i] = vals[{(static_cast<uint64_t>(i)), 0}];
            padHigh[i] = vals[{(static_cast<uint64_t>(i)), 1}];
        }
    }

    // Detect select type: GE/GT => max, LE/LT => min.
    bool selectMax = true;
    {
        auto& body = ssOp.getSelect();
        if (body.empty()) {
            MPS_LOG_ERROR("stablehlo.select_and_scatter: empty select body\n");
            return false;
        }
        bool found = false;
        for (auto& bodyOp : body.front().getOperations()) {
            if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(bodyOp)) {
                auto dir = cmpOp.getComparisonDirection();
                selectMax = (dir == mlir::stablehlo::ComparisonDirection::GE ||
                             dir == mlir::stablehlo::ComparisonDirection::GT);
                found = true;
                break;
            }
        }
        if (!found) {
            MPS_LOG_ERROR("stablehlo.select_and_scatter: no compare in select body\n");
            return false;
        }
    }

    // Detect scatter type (typically add for pool gradients).
    ScatterType scatterType = DetectScatterType(ssOp.getScatter());
    if (scatterType != ScatterType::Add) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: only add scatter supported (got %d)\n",
                      static_cast<int>(scatterType));
        return false;
    }

    // Pad operand with -inf (max) or +inf (min) so padded elements are never selected.
    mlx::core::array paddedOp = *operand;
    bool needsPad = false;
    for (int64_t i = 0; i < rank; i++) {
        if (padLow[i] != 0 || padHigh[i] != 0) {
            needsPad = true;
            break;
        }
    }
    if (needsPad) {
        float padScalar = selectMax ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::infinity();
        auto padVal = mlx::core::full({}, padScalar, operand->dtype());
        std::vector<std::pair<int, int>> padWidth(rank, {0, 0});
        for (int64_t i = 0; i < rank; i++)
            padWidth[i] = {static_cast<int>(padLow[i]), static_cast<int>(padHigh[i])};
        paddedOp = mlx::core::pad(*operand, padWidth, padVal);
    }

    int ndim = static_cast<int>(rank);

    int64_t totalWinPositions = 1;
    for (int64_t i = 0; i < rank; i++)
        totalWinPositions *= windowDims[i];

    auto makeSliceParams = [&](int64_t flatWinIdx) {
        std::vector<int> si(ndim, 0);
        std::vector<int> ei(ndim);
        std::vector<int> st(ndim, 1);
        for (int d = 0; d < ndim; ++d)
            ei[d] = static_cast<int>(paddedOp.shape(d));

        int64_t remaining = flatWinIdx;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t wd = windowDims[d];
            int64_t offset = remaining % wd;
            remaining /= wd;
            int outSize = static_cast<int>(source->shape(d));
            int str = static_cast<int>(strides[d]);
            si[d] = static_cast<int>(offset);
            ei[d] = static_cast<int>(offset) + outSize * str;
            st[d] = str;
        }
        return std::make_tuple(mlx::core::Shape(si.begin(), si.end()),
                               mlx::core::Shape(ei.begin(), ei.end()),
                               mlx::core::Shape(st.begin(), st.end()));
    };

    // Pass 1: Compute forward pool result
    auto fwdResult = selectMax
                         ? mlx::core::full(source->shape(), -std::numeric_limits<float>::infinity(),
                                           operand->dtype())
                         : mlx::core::full(source->shape(), std::numeric_limits<float>::infinity(),
                                           operand->dtype());

    for (int64_t wi = 0; wi < totalWinPositions; ++wi) {
        auto [si, ei, st] = makeSliceParams(wi);
        auto vals = mlx::core::slice(paddedOp, si, ei, st);
        fwdResult =
            selectMax ? mlx::core::maximum(fwdResult, vals) : mlx::core::minimum(fwdResult, vals);
    }

    // Pass 2: Scatter gradients using first-occurrence mask.
    auto initVal = mlx::core::astype(*init, operand->dtype());
    auto sasResult = mlx::core::broadcast_to(initVal, paddedOp.shape());
    auto won = mlx::core::zeros(source->shape(), mlx::core::bool_);

    for (int64_t wi = 0; wi < totalWinPositions; ++wi) {
        auto [slicerStart, slicerEnd, slicerStride] = makeSliceParams(wi);
        auto inputSlice = mlx::core::slice(paddedOp, slicerStart, slicerEnd, slicerStride);
        auto isMatch = mlx::core::equal(inputSlice, fwdResult);
        auto isFirst = mlx::core::logical_and(isMatch, mlx::core::logical_not(won));
        won = mlx::core::logical_or(won, isFirst);
        auto mask = mlx::core::astype(isFirst, operand->dtype());
        auto gradContrib = mlx::core::multiply(*source, mask);
        auto curSlice = mlx::core::slice(sasResult, slicerStart, slicerEnd, slicerStride);
        sasResult = mlx::core::slice_update(sasResult, mlx::core::add(curSlice, gradContrib),
                                            slicerStart, slicerEnd, slicerStride);
    }

    // If we padded, extract the unpadded region.
    if (needsPad) {
        std::vector<int> sliceStart(ndim, 0);
        std::vector<int> sliceEnd(ndim);
        for (int d = 0; d < ndim; ++d) {
            sliceStart[d] = static_cast<int>(padLow[d]);
            sliceEnd[d] = sliceStart[d] + static_cast<int>(operand->shape(d));
        }
        sasResult =
            mlx::core::slice(sasResult, mlx::core::Shape(sliceStart.begin(), sliceStart.end()),
                             mlx::core::Shape(sliceEnd.begin(), sliceEnd.end()));
    }

    values.emplace(ToKey(op->getResult(0)), std::move(sasResult));
    return true;
}

}  // namespace

void RegisterReductionHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.reduce", HandleReduce});
    handlers.insert({"stablehlo.reduce_window", HandleReduceWindow});
    handlers.insert({"stablehlo.select_and_scatter", HandleSelectAndScatter});
}

}  // namespace jax_mps
