// Reduction operations: reduce (sum, product, max, min, and, or)

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Helper to identify the reduction operation type from the region body
// Returns the operation name if it's a simple binary reduction, empty string otherwise
static std::string GetReductionOpType(mlir::Region& body) {
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

// Reduce operation - identifies reduction type and maps to MPS reduction
static ProcessResult Handle_reduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
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

    if (!result)
        return ProcessResult::Error("reduce: handler returned null");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.reduce", Handle_reduce);

// stablehlo.return is a terminator used inside regions (e.g., reduce body)
// It's handled implicitly by parent operations, not executed directly
static ProcessResult Handle_return(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    // This should never be called directly - it's handled by the parent operation
    // But we register it so it's not flagged as unsupported during module verification
    MPS_LOG_WARN("stablehlo.return should not be called directly\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", Handle_return);

}  // namespace jax_mps
