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

// Helper to get bit width from tensor's element type
static int getBitWidth(mlir::Operation* op) {
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return 0;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return intType.getWidth();
    }
    return 0;
}

// Shift left - when shift >= bit_width, result should be 0
static MPSGraphTensor* Handle_shift_left(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                         NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(values, op, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(op);
    if (bitWidth == 0)
        return nullptr;

    // Perform the shift
    MPSGraphTensor* shiftedResult = [g bitwiseLeftShiftWithPrimaryTensor:input
                                                         secondaryTensor:shiftAmount
                                                                    name:nil];

    // Create constant for bit width comparison
    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftAmount.dataType];

    // Check if shift >= bit_width
    MPSGraphTensor* overflowMask = [g greaterThanOrEqualToWithPrimaryTensor:shiftAmount
                                                            secondaryTensor:bitWidthTensor
                                                                       name:nil];

    // Create zero tensor
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];

    // Select: if overflow, return 0; else return shifted result
    return [g selectWithPredicateTensor:overflowMask
                    truePredicateTensor:zeroTensor
                   falsePredicateTensor:shiftedResult
                                   name:nil];
}
REGISTER_MPS_OP("stablehlo.shift_left", Handle_shift_left);

// Shift right logical - when shift >= bit_width, result should be 0
static MPSGraphTensor* Handle_shift_right_logical(MPSGraph* g, mlir::Operation* op,
                                                  ValueMap& values, NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(values, op, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(op);
    if (bitWidth == 0)
        return nullptr;

    // Perform the shift
    MPSGraphTensor* shiftedResult = [g bitwiseRightShiftWithPrimaryTensor:input
                                                          secondaryTensor:shiftAmount
                                                                     name:nil];

    // Create constant for bit width comparison
    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftAmount.dataType];

    // Check if shift >= bit_width
    MPSGraphTensor* overflowMask = [g greaterThanOrEqualToWithPrimaryTensor:shiftAmount
                                                            secondaryTensor:bitWidthTensor
                                                                       name:nil];

    // Create zero tensor
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];

    // Select: if overflow, return 0; else return shifted result
    return [g selectWithPredicateTensor:overflowMask
                    truePredicateTensor:zeroTensor
                   falsePredicateTensor:shiftedResult
                                   name:nil];
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", Handle_shift_right_logical);

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
