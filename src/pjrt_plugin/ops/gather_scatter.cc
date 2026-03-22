// Gather and scatter op handlers.

#include <algorithm>
#include <set>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

// --- Shared scatter utilities (declared in handler_utils.h) ---

ScatterType DetectScatterType(mlir::Region& body) {
    if (body.empty())
        return ScatterType::Unknown;

    auto& block = body.front();
    for (auto& op : block.getOperations()) {
        auto opName = op.getName().getStringRef();
        // The body takes (current, update) and returns the result
        // For simple update: return update (the second arg)
        if (opName == "stablehlo.return") {
            // Check if it returns the second block argument directly
            if (op.getNumOperands() == 1) {
                auto returnVal = op.getOperand(0);
                if (returnVal == block.getArgument(1)) {
                    return ScatterType::Update;
                }
            }
        }
        if (opName == "stablehlo.add")
            return ScatterType::Add;
        if (opName == "stablehlo.subtract")
            return ScatterType::Sub;
        if (opName == "stablehlo.multiply")
            return ScatterType::Mul;
        if (opName == "stablehlo.minimum")
            return ScatterType::Min;
        if (opName == "stablehlo.maximum")
            return ScatterType::Max;
    }
    return ScatterType::Unknown;
}

std::optional<mlx::core::array> ApplyScatter(ScatterType scatterType,
                                             const mlx::core::array& operand,
                                             const std::vector<mlx::core::array>& idxVec,
                                             const mlx::core::array& updates,
                                             const std::vector<int>& axes) {
    switch (scatterType) {
        case ScatterType::Update:
            return mlx::core::scatter(operand, idxVec, updates, axes);
        case ScatterType::Add:
            return mlx::core::scatter_add(operand, idxVec, updates, axes);
        case ScatterType::Sub:
            return mlx::core::scatter_add(operand, idxVec, mlx::core::negative(updates), axes);
        case ScatterType::Mul:
            return mlx::core::scatter_prod(operand, idxVec, updates, axes);
        case ScatterType::Min:
            return mlx::core::scatter_min(operand, idxVec, updates, axes);
        case ScatterType::Max:
            return mlx::core::scatter_max(operand, idxVec, updates, axes);
        default:
            MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter update type\n");
            return std::nullopt;
    }
}

namespace {

// Helper: extract per-axis index arrays from start_indices and convert to int32.
// Returns the index arrays and their axes, plus the batch shape (start_indices
// shape with index_vector_dim removed).
struct IndexExtraction {
    std::vector<mlx::core::array> idxVec;
    std::vector<int> axes;
    mlx::core::Shape idxBatchShape;
    bool hasIdxVecDim;
};

IndexExtraction ExtractPerAxisIndices(const mlx::core::array& startIndices,
                                      llvm::ArrayRef<int64_t> indexMap, int indexVectorDim) {
    IndexExtraction result;
    result.hasIdxVecDim = indexVectorDim < startIndices.ndim();

    for (int d = 0; d < startIndices.ndim(); ++d) {
        if (!result.hasIdxVecDim || d != indexVectorDim) {
            result.idxBatchShape.push_back(startIndices.shape(d));
        }
    }

    for (size_t i = 0; i < indexMap.size(); ++i) {
        int axis = static_cast<int>(indexMap[i]);
        result.axes.push_back(axis);

        mlx::core::array axisIdx = startIndices;
        if (result.hasIdxVecDim) {
            mlx::core::Shape starts(startIndices.ndim(), 0);
            mlx::core::Shape stops(startIndices.shape().begin(), startIndices.shape().end());
            starts[indexVectorDim] = static_cast<int>(i);
            stops[indexVectorDim] = static_cast<int>(i) + 1;
            axisIdx =
                mlx::core::squeeze(mlx::core::slice(startIndices, starts, stops), {indexVectorDim});
        }

        if (axisIdx.dtype() != mlx::core::int32) {
            axisIdx = mlx::core::astype(axisIdx, mlx::core::int32);
        }
        result.idxVec.push_back(axisIdx);
    }

    return result;
}

// Helper: add iota indices for batching dimensions.
void AddBatchingIotas(std::vector<mlx::core::array>& idxVec, std::vector<int>& axes,
                      const mlx::core::array& operand, const mlx::core::Shape& idxBatchShape,
                      llvm::ArrayRef<int64_t> operandBatchingDims,
                      llvm::ArrayRef<int64_t> indicesBatchingDims, bool hasIdxVecDim,
                      int indexVectorDim) {
    for (size_t i = 0; i < operandBatchingDims.size(); ++i) {
        int operandAxis = static_cast<int>(operandBatchingDims[i]);
        int idxBatchDim = static_cast<int>(indicesBatchingDims[i]);
        if (hasIdxVecDim && idxBatchDim > indexVectorDim) {
            --idxBatchDim;
        }

        axes.push_back(operandAxis);

        int batchSz = operand.shape(operandAxis);
        mlx::core::Shape iotaShape(idxBatchShape.size(), 1);
        iotaShape[idxBatchDim] = batchSz;
        auto iota = mlx::core::reshape(mlx::core::arange(batchSz, mlx::core::int32), iotaShape);
        iota = mlx::core::broadcast_to(iota, idxBatchShape);
        idxVec.push_back(iota);
    }
}

// Handler for stablehlo.gather
bool HandleGather(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto gatherOp = CastOp<mlir::stablehlo::GatherOp>(op, "stablehlo.gather");
    if (!gatherOp)
        return false;

    auto* operand = RequireValue(values, gatherOp.getOperand(), "stablehlo.gather");
    auto* startIndices = RequireValue(values, gatherOp.getStartIndices(), "stablehlo.gather");
    if (!operand || !startIndices)
        return false;

    auto dn = gatherOp.getDimensionNumbers();
    auto offsetDims = dn.getOffsetDims();
    auto collapsedSliceDims = dn.getCollapsedSliceDims();
    auto startIndexMap = dn.getStartIndexMap();
    int indexVectorDim = static_cast<int>(dn.getIndexVectorDim());
    auto operandBatchingDims = dn.getOperandBatchingDims();
    auto startIndicesBatchingDims = dn.getStartIndicesBatchingDims();

    mlx::core::Shape sliceSizes;
    for (auto s : gatherOp.getSliceSizes()) {
        sliceSizes.push_back(static_cast<int>(s));
    }

    // Step 1: Extract per-axis index arrays
    auto extraction = ExtractPerAxisIndices(*startIndices, startIndexMap, indexVectorDim);
    auto& idxVec = extraction.idxVec;
    auto& axes = extraction.axes;
    auto& idxBatchShape = extraction.idxBatchShape;

    // Clamp gather indices to valid range
    for (size_t i = 0; i < idxVec.size(); ++i) {
        int axis = axes[i];
        int maxStart = operand->shape(axis) - sliceSizes[axis];
        idxVec[i] = mlx::core::clip(idxVec[i], mlx::core::array(0, mlx::core::int32),
                                    mlx::core::array(std::max(0, maxStart), mlx::core::int32));
    }

    // Step 2: Add iota indices for operand_batching_dims
    AddBatchingIotas(idxVec, axes, *operand, idxBatchShape, operandBatchingDims,
                     startIndicesBatchingDims, extraction.hasIdxVecDim, indexVectorDim);

    // Step 3: Call mlx::core::gather
    auto result = mlx::core::gather(*operand, idxVec, axes, sliceSizes);

    // Step 4: Squeeze collapsed_slice_dims and operand_batching_dims
    int idxNdim = static_cast<int>(idxBatchShape.size());
    std::vector<int> squeezeDims;
    for (auto d : collapsedSliceDims) {
        squeezeDims.push_back(idxNdim + static_cast<int>(d));
    }
    for (auto d : operandBatchingDims) {
        squeezeDims.push_back(idxNdim + static_cast<int>(d));
    }
    std::sort(squeezeDims.begin(), squeezeDims.end());
    if (!squeezeDims.empty()) {
        result = mlx::core::squeeze(result, squeezeDims);
    }

    // Step 5: Transpose to place offset_dims at their specified positions
    int resultRank = static_cast<int>(result.ndim());
    int numOffset = static_cast<int>(offsetDims.size());
    int numBatch = resultRank - numOffset;

    std::set<int> offsetDimSet(offsetDims.begin(), offsetDims.end());
    std::vector<int> perm;
    int batchIdx = 0;
    int offsetIdx = 0;
    for (int p = 0; p < resultRank; ++p) {
        if (offsetDimSet.count(p)) {
            perm.push_back(numBatch + offsetIdx++);
        } else {
            perm.push_back(batchIdx++);
        }
    }

    if (!IsIdentityPermutation(perm)) {
        result = mlx::core::transpose(result, perm);
    }

    // Step 6: Safety reshape to expected output shape
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (resultType) {
        auto expectedShape = GetShape(resultType);
        if (result.shape() != expectedShape) {
            result = mlx::core::reshape(result, expectedShape);
        }
    }

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

bool HandleScatter(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto scatterOp = CastOp<mlir::stablehlo::ScatterOp>(op, "stablehlo.scatter");
    if (!scatterOp)
        return false;

    // Only support single-input scatter for now
    if (scatterOp.getInputs().size() != 1) {
        MPS_LOG_ERROR("stablehlo.scatter: multi-input scatter not supported\n");
        return false;
    }

    auto* operand = RequireValue(values, scatterOp.getInputs()[0], "stablehlo.scatter");
    auto* scatterIndices = RequireValue(values, scatterOp.getScatterIndices(), "stablehlo.scatter");
    auto* updates = RequireValue(values, scatterOp.getUpdates()[0], "stablehlo.scatter");
    if (!operand || !scatterIndices || !updates)
        return false;

    auto dn = scatterOp.getScatterDimensionNumbers();
    auto insertedWindowDims = dn.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dn.getScatterDimsToOperandDims();
    int indexVectorDim = static_cast<int>(dn.getIndexVectorDim());
    auto updateWindowDims = dn.getUpdateWindowDims();
    auto inputBatchingDims = dn.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dn.getScatterIndicesBatchingDims();

    auto& body = scatterOp.getUpdateComputation();
    auto scatterType = DetectScatterType(body);
    if (scatterType == ScatterType::Unknown) {
        MPS_LOG_ERROR("stablehlo.scatter: unsupported update computation\n");
        return false;
    }

    // Determine which operand dims have window extent > 1
    std::set<int> insertedSet(insertedWindowDims.begin(), insertedWindowDims.end());
    std::set<int> batchingSet(inputBatchingDims.begin(), inputBatchingDims.end());

    std::vector<int> windowOperandDims;
    for (int d = 0; d < operand->ndim(); ++d) {
        if (!insertedSet.count(d) && !batchingSet.count(d)) {
            windowOperandDims.push_back(d);
        }
    }

    bool hasWindowScatter = false;
    for (auto dim : scatterDimsToOperandDims) {
        int d = static_cast<int>(dim);
        if (!insertedSet.count(d)) {
            for (int j = 0; j < static_cast<int>(windowOperandDims.size()); ++j) {
                if (windowOperandDims[j] == d) {
                    int updateDim = static_cast<int>(updateWindowDims[j]);
                    if (updates->shape(updateDim) > 1) {
                        hasWindowScatter = true;
                    }
                    break;
                }
            }
        }
        if (hasWindowScatter)
            break;
    }

    bool hasIdxVecDim = indexVectorDim < scatterIndices->ndim();

    // ===== Window scatter path: scatter axes with window extent > 1 =====
    if (hasWindowScatter) {
        // For multi-axis window scatter, use slice_update loop.
        if (scatterDimsToOperandDims.size() > 1) {
            if (!inputBatchingDims.empty()) {
                MPS_LOG_ERROR(
                    "stablehlo.scatter: multi-axis window scatter with batching dims "
                    "is not supported\n");
                return false;
            }

            std::vector<int> batchDims;
            int batchSize = 1;
            for (int d = 0; d < scatterIndices->ndim(); ++d) {
                if (!hasIdxVecDim || d != indexVectorDim) {
                    batchDims.push_back(d);
                    batchSize *= scatterIndices->shape(d);
                }
            }

            bool singleUpdate = (batchSize == 1);
            auto axes = ToIntVec(scatterDimsToOperandDims);

            int numPositions = singleUpdate ? 1 : batchSize;
            mlx::core::array result = *operand;
            for (int b = 0; b < numPositions; ++b) {
                std::vector<mlx::core::array> perAxisIdx;
                for (size_t i = 0; i < scatterDimsToOperandDims.size(); ++i) {
                    mlx::core::Shape starts(scatterIndices->ndim(), 0);
                    mlx::core::Shape stops(scatterIndices->shape().begin(),
                                           scatterIndices->shape().end());
                    if (hasIdxVecDim) {
                        starts[indexVectorDim] = static_cast<int>(i);
                        stops[indexVectorDim] = static_cast<int>(i) + 1;
                    }
                    std::vector<int> squeezeDims;
                    if (hasIdxVecDim)
                        squeezeDims.push_back(indexVectorDim);
                    if (!singleUpdate) {
                        int remaining = b;
                        for (int bd = static_cast<int>(batchDims.size()) - 1; bd >= 0; --bd) {
                            int dim = batchDims[bd];
                            starts[dim] = remaining % scatterIndices->shape(dim);
                            stops[dim] = starts[dim] + 1;
                            remaining /= scatterIndices->shape(dim);
                        }
                        for (int bd : batchDims)
                            squeezeDims.push_back(bd);
                    } else {
                        for (int bd : batchDims) {
                            if (scatterIndices->shape(bd) == 1)
                                squeezeDims.push_back(bd);
                        }
                    }
                    auto axisIdx = mlx::core::slice(*scatterIndices, starts, stops);
                    if (!squeezeDims.empty()) {
                        axisIdx = mlx::core::squeeze(axisIdx, squeezeDims);
                    }
                    if (axisIdx.dtype() != mlx::core::int32) {
                        axisIdx = mlx::core::astype(axisIdx, mlx::core::int32);
                    }
                    perAxisIdx.push_back(mlx::core::reshape(axisIdx, {1}));
                }
                auto startArr = mlx::core::concatenate(perAxisIdx, 0);

                auto updateVal = *updates;
                std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());
                if (singleUpdate) {
                    std::vector<int> squeezeDims;
                    for (int d = 0; d < updateVal.ndim(); ++d) {
                        if (!windowDimSet.count(d) && updateVal.shape(d) == 1) {
                            squeezeDims.push_back(d);
                        }
                    }
                    if (!squeezeDims.empty()) {
                        updateVal = mlx::core::squeeze(updateVal, squeezeDims);
                    }
                } else {
                    int numBatchDims = static_cast<int>(updateVal.ndim()) -
                                       static_cast<int>(updateWindowDims.size());
                    mlx::core::Shape starts(updateVal.ndim(), 0);
                    mlx::core::Shape stops(updateVal.shape().begin(), updateVal.shape().end());
                    int remaining = b;
                    for (int bd = numBatchDims - 1; bd >= 0; --bd) {
                        starts[bd] = remaining % updateVal.shape(bd);
                        stops[bd] = starts[bd] + 1;
                        remaining /= updateVal.shape(bd);
                    }
                    updateVal = mlx::core::slice(updateVal, starts, stops);
                    std::vector<int> squeezeDims;
                    squeezeDims.reserve(numBatchDims);
                    for (int d = 0; d < numBatchDims; ++d)
                        squeezeDims.push_back(d);
                    if (!squeezeDims.empty()) {
                        updateVal = mlx::core::squeeze(updateVal, squeezeDims);
                    }
                }

                switch (scatterType) {
                    case ScatterType::Update:
                        result = mlx::core::slice_update(result, updateVal, startArr, axes);
                        break;
                    case ScatterType::Add: {
                        if (!insertedWindowDims.empty() ||
                            static_cast<int>(updateVal.ndim()) != operand->ndim()) {
                            MPS_LOG_ERROR(
                                "stablehlo.scatter: multi-dim window scatter Add requires "
                                "empty insertedWindowDims and operand-rank updates\n");
                            return false;
                        }
                        mlx::core::Shape sliceSizes(operand->shape());
                        for (int axis : axes) {
                            sliceSizes[axis] = updateVal.shape(axis);
                        }
                        auto current = mlx::core::slice(result, startArr, axes, sliceSizes);
                        result = mlx::core::slice_update(result, mlx::core::add(current, updateVal),
                                                         startArr, axes);
                        break;
                    }
                    default:
                        MPS_LOG_ERROR(
                            "stablehlo.scatter: unsupported scatter update type "
                            "for multi-dim slice update\n");
                        return false;
                }
            }

            values.emplace(ToKey(op->getResult(0)), std::move(result));
            return true;
        }

        // Single-axis window scatter: expand start indices to per-element indices
        int scatterDim = static_cast<int>(scatterDimsToOperandDims[0]);

        auto indices = *scatterIndices;
        if (hasIdxVecDim && scatterIndices->shape(indexVectorDim) == 1) {
            indices = mlx::core::squeeze(*scatterIndices, {indexVectorDim});
        }
        if (indices.dtype() != mlx::core::int32) {
            indices = mlx::core::astype(indices, mlx::core::int32);
        }
        if (indices.ndim() == 0) {
            indices = mlx::core::reshape(indices, {1});
        }

        mlx::core::Shape idxBatchShape;
        for (int d = 0; d < scatterIndices->ndim(); ++d) {
            if (!hasIdxVecDim || d != indexVectorDim) {
                idxBatchShape.push_back(scatterIndices->shape(d));
            }
        }

        int winIdx = 0;
        for (int j = 0; j < static_cast<int>(windowOperandDims.size()); ++j) {
            if (windowOperandDims[j] == scatterDim) {
                winIdx = j;
                break;
            }
        }
        int windowDimForScatterAxis = static_cast<int>(updateWindowDims[winIdx]);
        int windowSize = updates->shape(windowDimForScatterAxis);

        // Expand: indices[..., None] + arange(windowSize)
        auto offsets = mlx::core::arange(windowSize, mlx::core::int32);
        auto idxShape = indices.shape();
        mlx::core::Shape broadcastIdxShape = idxShape;
        broadcastIdxShape.push_back(1);
        auto reshapedIdx = mlx::core::reshape(indices, broadcastIdxShape);
        mlx::core::Shape broadcastOffsetShape(indices.ndim(), 1);
        broadcastOffsetShape.push_back(windowSize);
        auto reshapedOffsets = mlx::core::reshape(offsets, broadcastOffsetShape);
        auto expandedIndices = mlx::core::add(reshapedIdx, reshapedOffsets);

        int totalIndices = 1;
        for (auto s : expandedIndices.shape())
            totalIndices *= static_cast<int>(s);
        expandedIndices = mlx::core::reshape(expandedIndices, {totalIndices});

        std::vector<mlx::core::array> idxVec = {expandedIndices};
        std::vector<int> axes = {scatterDim};

        for (size_t i = 0; i < inputBatchingDims.size(); ++i) {
            int operandAxis = static_cast<int>(inputBatchingDims[i]);
            int idxBatchDim = static_cast<int>(scatterIndicesBatchingDims[i]);
            if (hasIdxVecDim && idxBatchDim > indexVectorDim) {
                --idxBatchDim;
            }

            axes.push_back(operandAxis);

            int batchSz = operand->shape(operandAxis);
            mlx::core::Shape iotaShape(idxBatchShape.size(), 1);
            iotaShape[idxBatchDim] = batchSz;
            auto iota = mlx::core::reshape(mlx::core::arange(batchSz, mlx::core::int32), iotaShape);
            iota = mlx::core::broadcast_to(iota, idxBatchShape);
            mlx::core::Shape iotaExpShape = iota.shape();
            iotaExpShape.push_back(1);
            iota = mlx::core::reshape(iota, iotaExpShape);
            mlx::core::Shape bcastShape = iota.shape();
            bcastShape.back() = windowSize;
            iota = mlx::core::broadcast_to(iota, bcastShape);
            iota = mlx::core::reshape(iota, {totalIndices});
            idxVec.push_back(iota);
        }

        std::set<int> coveredDims;
        coveredDims.insert(scatterDim);
        for (auto d : insertedWindowDims)
            coveredDims.insert(static_cast<int>(d));
        for (auto d : inputBatchingDims)
            coveredDims.insert(static_cast<int>(d));

        std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());
        int updNdim = static_cast<int>(updates->ndim());

        std::vector<int> transposeOrder;
        for (int i = 0; i < updNdim; ++i) {
            if (!windowDimSet.count(i))
                transposeOrder.push_back(i);
        }
        transposeOrder.push_back(windowDimForScatterAxis);
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) && i != windowDimForScatterAxis) {
                transposeOrder.push_back(i);
            }
        }

        auto transposedUpdates = *updates;
        if (!IsIdentityPermutation(transposeOrder)) {
            transposedUpdates = mlx::core::transpose(*updates, transposeOrder);
        }
        auto tShape = transposedUpdates.shape();

        int numIdxDims = updNdim - static_cast<int>(updateWindowDims.size());
        int flatIdxSize = 1;
        for (int i = 0; i <= numIdxDims; ++i)
            flatIdxSize *= tShape[i];

        mlx::core::Shape newShape;
        newShape.push_back(flatIdxSize);
        int otherWindowIdx = numIdxDims + 1;
        for (int d = 0; d < operand->ndim(); ++d) {
            if (coveredDims.count(d)) {
                newShape.push_back(1);
            } else {
                newShape.push_back(tShape[otherWindowIdx++]);
            }
        }
        auto reshapedUpdates = mlx::core::reshape(transposedUpdates, newShape);

        auto result = ApplyScatter(scatterType, *operand, idxVec, reshapedUpdates, axes);
        if (!result)
            return false;
        values.emplace(ToKey(op->getResult(0)), std::move(*result));
        return true;
    }

    // ===== General point scatter path =====
    auto extraction =
        ExtractPerAxisIndices(*scatterIndices, scatterDimsToOperandDims, indexVectorDim);
    auto& idxVec = extraction.idxVec;
    auto& axes = extraction.axes;
    auto& idxBatchShape = extraction.idxBatchShape;

    // Add iota indices for batching dims
    AddBatchingIotas(idxVec, axes, *operand, idxBatchShape, inputBatchingDims,
                     scatterIndicesBatchingDims, hasIdxVecDim, indexVectorDim);

    // Reshape updates from StableHLO format to MLX format
    std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());
    int updNdim = static_cast<int>(updates->ndim());

    std::vector<int> transposeOrder;
    for (int i = 0; i < updNdim; ++i) {
        if (!windowDimSet.count(i))
            transposeOrder.push_back(i);
    }
    int numIdxDims = static_cast<int>(transposeOrder.size());
    for (int i = 0; i < updNdim; ++i) {
        if (windowDimSet.count(i))
            transposeOrder.push_back(i);
    }

    auto transposedUpdates = *updates;
    if (!IsIdentityPermutation(transposeOrder)) {
        transposedUpdates = mlx::core::transpose(*updates, transposeOrder);
    }
    auto tShape = transposedUpdates.shape();

    std::set<int> coveredDims;
    for (auto d : insertedWindowDims)
        coveredDims.insert(static_cast<int>(d));
    for (auto d : inputBatchingDims)
        coveredDims.insert(static_cast<int>(d));
    for (auto d : scatterDimsToOperandDims)
        coveredDims.insert(static_cast<int>(d));

    mlx::core::Shape newShape;
    for (int i = 0; i < numIdxDims; ++i) {
        newShape.push_back(tShape[i]);
    }
    int windowIdx = 0;
    for (int d = 0; d < operand->ndim(); ++d) {
        if (coveredDims.count(d)) {
            newShape.push_back(1);
        } else {
            newShape.push_back(tShape[numIdxDims + windowIdx++]);
        }
    }
    auto reshapedUpdates = mlx::core::reshape(transposedUpdates, newShape);

    auto result = ApplyScatter(scatterType, *operand, idxVec, reshapedUpdates, axes);
    if (!result)
        return false;

    values.emplace(ToKey(op->getResult(0)), std::move(*result));
    return true;
}

}  // namespace

void RegisterGatherScatterHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.gather", HandleGather});
    handlers.insert({"stablehlo.scatter", HandleScatter});
}

}  // namespace jax_mps
