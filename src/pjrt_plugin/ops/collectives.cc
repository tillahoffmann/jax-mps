// Collective op handlers.
//
// jax-mps does not support multi-device collectives. The handlers in this file
// are single-device fallbacks for programs that still lower through collective
// StableHLO ops with one local device/partition.

#include <cstdint>

#include "pjrt_plugin/ops/handler_utils.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.all_reduce.
//
// This is a single-device/debug fallback: bypass the collective and forward each
// operand as the corresponding result. Multi-device collectives must fail
// loudly; treating them as identity would silently produce wrong values.
bool HandleAllReduce(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto replicaGroups = op->getAttrOfType<mlir::DenseIntElementsAttr>("replica_groups");
    if (replicaGroups && !replicaGroups.empty()) {
        auto groupType = mlir::dyn_cast<mlir::RankedTensorType>(replicaGroups.getType());
        bool isMultiDevice = (!groupType || groupType.getRank() < 2)
                                 ? replicaGroups.getNumElements() > 1
                                 : groupType.getShape().back() > 1;
        if (isMultiDevice) {
            ctx.error_message = "stablehlo.all_reduce: multi-device all_reduce is not supported";
            MPS_LOG_ERROR("%s\n", ctx.error_message.c_str());
            return false;
        }
    }

    if (op->getNumOperands() != op->getNumResults()) {
        ctx.error_message = "stablehlo.all_reduce: operand/result count mismatch";
        MPS_LOG_ERROR("%s\n", ctx.error_message.c_str());
        return false;
    }

    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        auto* input = RequireValue(values, op->getOperand(i), "stablehlo.all_reduce");
        if (!input)
            return false;
        values.emplace(ToKey(op->getResult(i)), *input);
    }
    return true;
}

// Handler for stablehlo.partition_id.
//
// This is a single-process/single-partition fallback: returns partition id
// zero so pmap axis_index lowerings work on one MPS device. Multi-partition
// execution must fail loudly; returning zero there would silently produce wrong
// values.
bool HandlePartitionId(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    if (auto numPartitions = ctx.module->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions")) {
        if (numPartitions.getInt() != 1) {
            ctx.error_message =
                "stablehlo.partition_id: multi-partition execution is not supported";
            MPS_LOG_ERROR("%s\n", ctx.error_message.c_str());
            return false;
        }
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 0) {
        ctx.error_message = "stablehlo.partition_id: result must be a scalar tensor";
        MPS_LOG_ERROR("%s\n", ctx.error_message.c_str());
        return false;
    }

    auto dtype = MlirTypeToMlxDtype(resultType.getElementType());
    values.emplace(ToKey(op->getResult(0)),
                   mlx::core::astype(mlx::core::array(static_cast<uint32_t>(0)), dtype));
    return true;
}

}  // namespace

void RegisterCollectivesHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.all_reduce", HandleAllReduce});
    handlers.insert({"stablehlo.partition_id", HandlePartitionId});
}

}  // namespace jax_mps
