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

static ProcessResult HandleSort(MPSGraph* graph, mlir::Operation* op, ValueMap& values) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        return ProcessResult::Error("Expected stablehlo.sort");
    }

    auto dimAttr = op->getAttrOfType<mlir::IntegerAttr>("dimension");
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

    if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input) {
            return ProcessResult::Error("stablehlo.sort input tensor not found");
        }
        MPSGraphTensor* sorted = [graph sortWithTensor:input
                                                  axis:axis
                                            descending:descending
                                                  name:nil];
        if (!sorted) {
            return ProcessResult::Error("stablehlo.sort lowering failed");
        }
        values[op->getResult(0).getAsOpaquePointer()] = sorted;
        return ProcessResult{};
    }

    if (op->getNumOperands() >= 2 && op->getNumOperands() == op->getNumResults()) {
        // Tuple sort lowering used by lexsort-like patterns:
        // sort first N-1 tensors as lexicographic keys, apply permutation to all tensors.
        // We build permutation by stable-sorting from least-significant key to most-significant.
        MPSGraphTensor* base = GetInputTensor(values, op, 0);
        if (!base) {
            return ProcessResult::Error("stablehlo.sort key tensor not found");
        }

        MPSGraphTensor* perm = [graph coordinateAlongAxis:axis withShape:base.shape name:nil];
        perm = EnsureInt32(graph, perm);

        for (NSInteger keyIdx = (NSInteger)op->getNumOperands() - 2; keyIdx >= 0; --keyIdx) {
            MPSGraphTensor* keyTensor = GetInputTensor(values, op, (unsigned)keyIdx);
            if (!keyTensor) {
                return ProcessResult::Error("stablehlo.sort key tensor missing");
            }
            MPSGraphTensor* keyAtPerm = [graph gatherAlongAxis:axis
                                             withUpdatesTensor:keyTensor
                                                 indicesTensor:perm
                                                          name:nil];
            MPSGraphTensor* localOrder = [graph argSortWithTensor:keyAtPerm
                                                             axis:axis
                                                       descending:descending
                                                             name:nil];
            if (!localOrder) {
                return ProcessResult::Error("stablehlo.sort argSort lowering failed");
            }
            localOrder = EnsureInt32(graph, localOrder);
            perm = [graph gatherAlongAxis:axis
                        withUpdatesTensor:perm
                            indicesTensor:localOrder
                                     name:nil];
        }

        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            MPSGraphTensor* operandTensor = GetInputTensor(values, op, i);
            if (!operandTensor) {
                return ProcessResult::Error("stablehlo.sort operand tensor missing");
            }
            MPSGraphTensor* sorted = [graph gatherAlongAxis:axis
                                          withUpdatesTensor:operandTensor
                                              indicesTensor:perm
                                                       name:nil];
            if (!sorted) {
                return ProcessResult::Error("stablehlo.sort gather lowering failed");
            }
            values[op->getResult(i).getAsOpaquePointer()] = sorted;
        }
        return ProcessResult{};
    }

    return ProcessResult::Error("Unsupported stablehlo.sort operand/result shape");
}

static ProcessResult HandleTopK(MPSGraph* graph, mlir::Operation* op, ValueMap& values) {
    ProcessResult result;
    MPS_LOG_DEBUG("HandleTopK: entering with %u operands, %u results\n", op->getNumOperands(),
                  op->getNumResults());
    if (op->getNumOperands() < 1 || op->getNumResults() != 2) {
        return ProcessResult::Error("Unsupported top_k operand/result shape");
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPS_LOG_DEBUG("HandleTopK: got input tensor %p\n", (void*)input);
    if (!input) {
        return ProcessResult::Error("top_k input tensor not found");
    }

    int64_t k = -1;
    if (auto kAttr = op->getAttrOfType<mlir::IntegerAttr>("k")) {
        k = kAttr.getInt();
    }

    NSInteger axis = -1;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
        axis = (NSInteger)axisAttr.getInt();
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(op, 0);
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
        topk = [graph topKWithSourceTensor:input k:(NSUInteger)k name:nil];
    } else {
        MPS_LOG_DEBUG("HandleTopK: calling topKWithSourceTensor axis=%ld\n", (long)axis);
        topk = [graph topKWithSourceTensor:input axis:axis k:(NSUInteger)k name:nil];
    }
    MPS_LOG_DEBUG("HandleTopK: topk result count=%lu\n", topk ? (unsigned long)topk.count : 0UL);
    if (!topk || topk.count != 2) {
        return ProcessResult::Error("top_k lowering failed");
    }

    MPSGraphTensor* valueOut = topk[0];
    MPSGraphTensor* indexOut = topk[1];
    MPS_LOG_DEBUG("HandleTopK: valueOut=%p, indexOut=%p\n", (void*)valueOut, (void*)indexOut);

    MPSDataType valueType = GetResultMpsType(op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [graph castTensor:indexOut toType:indexType name:nil];
    }

    if (valueShape) {
        valueOut = [graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(op, 1);
    if (indexShape) {
        indexOut = [graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    MPS_LOG_DEBUG("HandleTopK: storing results at %p and %p\n",
                  op->getResult(0).getAsOpaquePointer(), op->getResult(1).getAsOpaquePointer());
    values[op->getResult(0).getAsOpaquePointer()] = valueOut;
    values[op->getResult(1).getAsOpaquePointer()] = indexOut;

    // Add both outputs as auxiliary tensors - MPS requires all outputs from
    // multi-output ops to be computed even if only some are used
    result.auxiliary_tensors.push_back((__bridge void*)valueOut);
    result.auxiliary_tensors.push_back((__bridge void*)indexOut);

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
