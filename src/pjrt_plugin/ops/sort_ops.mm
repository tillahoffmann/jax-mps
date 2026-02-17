#import "pjrt_plugin/ops/gather_scatter_utils.h"
#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

int64_t inferTopKFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count ||
        outputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return outDim;
        }
    }
    return [outputShape.lastObject longLongValue];
}

NSInteger inferTopKAxisFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count ||
        inputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return (NSInteger)i;
        }
    }
    return (NSInteger)inputShape.count - 1;
}

}  // namespace

static ProcessResult HandleSort(HandlerContext& ctx) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(ctx.op);
    if (!sortOp) {
        return ProcessResult::Error("Expected stablehlo.sort");
    }

    auto dimAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("dimension");
    if (!dimAttr) {
        return ProcessResult::Error("stablehlo.sort missing dimension attribute");
    }
    NSInteger axis = (NSInteger)dimAttr.getInt();

    bool descending = false;
    if (!sortOp.getComparator().empty()) {
        for (mlir::Operation& nestedOp : sortOp.getComparator().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            auto direction = compareOp.getComparisonDirection();
            descending = direction == mlir::stablehlo::ComparisonDirection::GT ||
                         direction == mlir::stablehlo::ComparisonDirection::GE;
            break;
        }
    }

    if (ctx.op->getNumOperands() == 1 && ctx.op->getNumResults() == 1) {
        MPSGraphTensor* input = GetInputTensor(ctx, 0);
        if (!input) {
            return ProcessResult::Error("stablehlo.sort input tensor not found");
        }
        MPSGraphTensor* sorted = [ctx.graph sortWithTensor:input
                                                      axis:axis
                                                descending:descending
                                                      name:nil];
        if (!sorted) {
            return ProcessResult::Error("stablehlo.sort lowering failed");
        }
        ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = sorted;
        return ProcessResult{};
    }

    if (ctx.op->getNumOperands() >= 2 && ctx.op->getNumOperands() == ctx.op->getNumResults()) {
        // Tuple sort lowering used by lexsort-like patterns:
        // sort first N-1 tensors as lexicographic keys, apply permutation to all tensors.
        // We build permutation by stable-sorting from least-significant key to most-significant.
        MPSGraphTensor* base = GetInputTensor(ctx, 0);
        if (!base) {
            return ProcessResult::Error("stablehlo.sort key tensor not found");
        }

        MPSGraphTensor* perm = [ctx.graph coordinateAlongAxis:axis withShape:base.shape name:nil];
        perm = EnsureInt32(ctx.graph, perm);

        for (NSInteger keyIdx = (NSInteger)ctx.op->getNumOperands() - 2; keyIdx >= 0; --keyIdx) {
            MPSGraphTensor* keyTensor = GetInputTensor(ctx, (unsigned)keyIdx);
            if (!keyTensor) {
                return ProcessResult::Error("stablehlo.sort key tensor missing");
            }
            MPSGraphTensor* keyAtPerm = SafeGatherAlongAxis(ctx.graph, axis, keyTensor, perm);
            MPSGraphTensor* localOrder = [ctx.graph argSortWithTensor:keyAtPerm
                                                                 axis:axis
                                                           descending:descending
                                                                 name:nil];
            if (!localOrder) {
                return ProcessResult::Error("stablehlo.sort argSort lowering failed");
            }
            localOrder = EnsureInt32(ctx.graph, localOrder);
            perm = SafeGatherAlongAxis(ctx.graph, axis, perm, localOrder);
        }

        for (unsigned i = 0; i < ctx.op->getNumResults(); ++i) {
            MPSGraphTensor* operandTensor = GetInputTensor(ctx, i);
            if (!operandTensor) {
                return ProcessResult::Error("stablehlo.sort operand tensor missing");
            }
            MPSGraphTensor* sorted = SafeGatherAlongAxis(ctx.graph, axis, operandTensor, perm);
            if (!sorted) {
                return ProcessResult::Error("stablehlo.sort gather lowering failed");
            }
            ctx.values[ctx.op->getResult(i).getAsOpaquePointer()] = sorted;
        }
        return ProcessResult{};
    }

    return ProcessResult::Error("Unsupported stablehlo.sort operand/result shape");
}

static ProcessResult HandleTopK(HandlerContext& ctx) {
    ProcessResult result;
    MPS_LOG_DEBUG("HandleTopK: entering with %u operands, %u results\n", ctx.op->getNumOperands(),
                  ctx.op->getNumResults());
    if (ctx.op->getNumOperands() < 1 || ctx.op->getNumResults() != 2) {
        return ProcessResult::Error("Unsupported top_k operand/result shape");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPS_LOG_DEBUG("HandleTopK: got input tensor %p\n", (void*)input);
    if (!input) {
        return ProcessResult::Error("top_k input tensor not found");
    }

    int64_t k = -1;
    if (auto kAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("k")) {
        k = kAttr.getInt();
    }

    NSInteger axis = -1;
    if (auto axisAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("axis")) {
        axis = (NSInteger)axisAttr.getInt();
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(ctx.op, 0);
    if (k < 0) {
        k = inferTopKFromShapes(input.shape, valueShape);
    }
    if (axis < 0) {
        axis = inferTopKAxisFromShapes(input.shape, valueShape);
    }

    if (k <= 0) {
        return ProcessResult::Error("Failed to infer top_k parameter k");
    }
    if (axis < 0 || !input.shape || axis >= (NSInteger)input.shape.count) {
        return ProcessResult::Error("Failed to infer top_k axis");
    }

    MPS_LOG_DEBUG("HandleTopK: k=%lld, axis=%ld\n", k, (long)axis);
    NSArray<MPSGraphTensor*>* topk = nil;
    if (axis == (NSInteger)input.shape.count - 1) {
        MPS_LOG_DEBUG("HandleTopK: calling topKWithSourceTensor (last axis)\n");
        topk = [ctx.graph topKWithSourceTensor:input k:(NSUInteger)k name:nil];
    } else {
        MPS_LOG_DEBUG("HandleTopK: calling topKWithSourceTensor axis=%ld\n", (long)axis);
        topk = [ctx.graph topKWithSourceTensor:input axis:axis k:(NSUInteger)k name:nil];
    }
    MPS_LOG_DEBUG("HandleTopK: topk result count=%lu\n", topk ? (unsigned long)topk.count : 0UL);
    if (!topk || topk.count != 2) {
        return ProcessResult::Error("top_k lowering failed");
    }

    MPSGraphTensor* valueOut = topk[0];
    MPSGraphTensor* indexOut = topk[1];
    MPS_LOG_DEBUG("HandleTopK: valueOut=%p, indexOut=%p\n", (void*)valueOut, (void*)indexOut);

    MPS_LOG_DEBUG("HandleTopK: checking for casts/reshapes\n");
    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        MPS_LOG_DEBUG("HandleTopK: casting valueOut from %d to %d\n", (int)valueOut.dataType,
                      (int)valueType);
        valueOut = [ctx.graph castTensor:valueOut toType:valueType name:nil];
        MPS_LOG_DEBUG("HandleTopK: valueOut after cast=%p\n", (void*)valueOut);
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        MPS_LOG_DEBUG("HandleTopK: casting indexOut from %d to %d\n", (int)indexOut.dataType,
                      (int)indexType);
        indexOut = [ctx.graph castTensor:indexOut toType:indexType name:nil];
        MPS_LOG_DEBUG("HandleTopK: indexOut after cast=%p\n", (void*)indexOut);
    }

    // Only reshape if the shapes actually differ
    if (valueShape && ![valueShape isEqualToArray:valueOut.shape]) {
        MPS_LOG_DEBUG("HandleTopK: reshaping valueOut from %s to %s\n",
                      [[valueOut.shape description] UTF8String],
                      [[valueShape description] UTF8String]);
        valueOut = [ctx.graph reshapeTensor:valueOut withShape:valueShape name:nil];
        MPS_LOG_DEBUG("HandleTopK: valueOut after reshape=%p\n", (void*)valueOut);
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(ctx.op, 1);
    if (indexShape && ![indexShape isEqualToArray:indexOut.shape]) {
        MPS_LOG_DEBUG("HandleTopK: reshaping indexOut from %s to %s\n",
                      [[indexOut.shape description] UTF8String],
                      [[indexShape description] UTF8String]);
        indexOut = [ctx.graph reshapeTensor:indexOut withShape:indexShape name:nil];
        MPS_LOG_DEBUG("HandleTopK: indexOut after reshape=%p\n", (void*)indexOut);
    }

    MPS_LOG_DEBUG("HandleTopK: final valueOut=%p, indexOut=%p\n", (void*)valueOut, (void*)indexOut);
    MPS_LOG_DEBUG("HandleTopK: storing results at %p and %p\n",
                  ctx.op->getResult(0).getAsOpaquePointer(),
                  ctx.op->getResult(1).getAsOpaquePointer());
    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = valueOut;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indexOut;

    MPS_LOG_DEBUG("HandleTopK: done\n");
    return result;
}

// Register sort-related ops
REGISTER_MPS_OP("stablehlo.sort", HandleSort);
REGISTER_MPS_OP("chlo.top_k", HandleTopK);

// Register top_k custom call targets
REGISTER_CUSTOM_CALL("stablehlo.dynamic_top_k", HandleTopK, stablehlo_dynamic_top_k);
REGISTER_CUSTOM_CALL("mhlo.topk", HandleTopK, mhlo_topk);
REGISTER_CUSTOM_CALL("mhlo.top_k", HandleTopK, mhlo_top_k);

}  // namespace jax_mps
