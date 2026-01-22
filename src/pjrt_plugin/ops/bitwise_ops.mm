// Bitwise operations: and, or, xor, shift_left, shift_right_logical
// Also includes concatenate which is needed for RNG

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Helper to check if operation result is boolean type
static bool isBooleanResult(mlir::Operation* op) {
    if (op->getNumResults() == 0)
        return false;
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return false;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return intType.getWidth() == 1;
    }
    return false;
}

// AND - use logical for booleans, bitwise for integers
static MPSGraphTensor* Handle_and(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                  NSArray<NSNumber*>*) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    if (isBooleanResult(op)) {
        return [g logicalANDWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    }
    return [g bitwiseANDWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
REGISTER_MPS_OP("stablehlo.and", Handle_and);

// OR - use logical for booleans, bitwise for integers
static MPSGraphTensor* Handle_or(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                 NSArray<NSNumber*>*) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    if (isBooleanResult(op)) {
        return [g logicalORWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    }
    return [g bitwiseORWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
REGISTER_MPS_OP("stablehlo.or", Handle_or);

// XOR - use logical for booleans, bitwise for integers
static MPSGraphTensor* Handle_xor(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                  NSArray<NSNumber*>*) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    if (isBooleanResult(op)) {
        return [g logicalXORWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    }
    return [g bitwiseXORWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
REGISTER_MPS_OP("stablehlo.xor", Handle_xor);

REGISTER_MLIR_BINARY_OP("stablehlo.shift_left", bitwiseLeftShift, shift_left);
REGISTER_MLIR_BINARY_OP("stablehlo.shift_right_logical", bitwiseRightShift, shift_right_logical);

// Concatenate - joins tensors along a dimension
static MPSGraphTensor* Handle_concatenate(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                          NSArray<NSNumber*>*) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op);
    if (!concatOp) {
        NSLog(@"ERROR: Expected ConcatenateOp");
        return nullptr;
    }

    // Gather all input tensors
    NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray array];
    for (mlir::Value operand : op->getOperands()) {
        MPSGraphTensor* tensor = GetTensor(values, operand);
        if (tensor) {
            [input_tensors addObject:tensor];
        }
    }

    if (input_tensors.count == 0) {
        NSLog(@"ERROR: Concatenate operation has no valid inputs");
        return nullptr;
    }

    // Get the concatenate dimension from the op
    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    return [g concatTensors:input_tensors dimension:dimension name:nil];
}
REGISTER_MPS_OP("stablehlo.concatenate", Handle_concatenate);

}  // namespace jax_mps
