// Reduction operations: reduce (sum, product, max, min, and, or, argmax, argmin)

#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Helper to identify the reduction operation type from the region body
// Returns the operation name if it's a simple binary reduction, empty string otherwise
std::string GetReductionOpType(mlir::Region& body) {
    if (body.empty())
        return "";

    mlir::Block& block = body.front();

    // The reduction body should have exactly one operation (plus terminator)
    // that is the reduction function
    for (mlir::Operation& op : block) {
        std::string opName = op.getName().getStringRef().str();

        // Skip the terminator (stablehlo.return)
        if (opName == "stablehlo.return")
            continue;

        // Return the first binary operation we find
        return opName;
    }

    return "";
}

// For detecting argmax/argmin patterns in multi-result reduce
enum class ArgReduceKind { kUnknown, kMax, kMin };

bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        // If the compare operands are swapped, the max/min interpretation flips.
        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

}  // namespace

// Single-result reduce: sum, product, max, min, and, or
static ProcessResult HandleSingleResultReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }

    // Get the input tensor (first operand)
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        return ProcessResult::Error("reduce: input tensor not found");
    }

    // Get reduction dimensions
    auto dimensions = reduceOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    // Identify the reduction operation from the body
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [g reductionSumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [g reductionProductWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [g reductionMaximumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [g reductionMinimumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.and") {
        result = [g reductionAndWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.or") {
        result = [g reductionOrWithTensor:input axes:axes name:nil];
    } else {
        return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
    }

    // MPS Graph reduction keeps dimensions (with size 1), but StableHLO reduce removes them
    // Reshape to the expected output shape from the MLIR operation
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape && result) {
        result = [g reshapeTensor:result withShape:outputShape name:nil];
    }

    return Result(values, op, result, "reduce");
}

// Multi-result reduce: argmax/argmin patterns
static ProcessResult HandleMultiResultReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }
    if (op->getNumResults() != 2 || op->getNumOperands() < 2) {
        return ProcessResult::Error("reduce: unsupported multi-result shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(values, op, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce: value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("reduce: only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("reduce: unsupported multi-result reduce body");
    }

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (kind == ArgReduceKind::kMax) {
        valueOut = [g reductionMaximumWithTensor:valueInput axis:axis name:nil];
        indexOut = [g reductionArgMaximumWithTensor:valueInput axis:axis name:nil];
    } else {
        valueOut = [g reductionMinimumWithTensor:valueInput axis:axis name:nil];
        indexOut = [g reductionArgMinimumWithTensor:valueInput axis:axis name:nil];
    }
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("reduce: failed to lower multi-result reduce");
    }

    MPSDataType valueType = GetResultMpsType(op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [g castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [g castTensor:indexOut toType:indexType name:nil];
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(op, 0);
    if (valueShape && valueOut) {
        valueOut = [g reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(op, 1);
    if (indexShape && indexOut) {
        indexOut = [g reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    values[op->getResult(0).getAsOpaquePointer()] = valueOut;
    values[op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

// Unified reduce handler - dispatches based on result count
static ProcessResult HandleReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    if (op->getNumResults() > 1) {
        return HandleMultiResultReduce(g, op, values);
    }
    return HandleSingleResultReduce(g, op, values);
}
REGISTER_MPS_OP("stablehlo.reduce", HandleReduce);

// stablehlo.return is a terminator used inside regions (e.g., reduce body)
// It's handled implicitly by parent operations, not executed directly
static ProcessResult HandleReturn(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    // This should never be called directly - it's handled by the parent operation
    // But we register it so it's not flagged as unsupported during module verification
    MPS_LOG_WARN("stablehlo.return should not be called directly\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", HandleReturn);

}  // namespace jax_mps
