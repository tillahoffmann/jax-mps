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

// Macro for logical/bitwise operations (AND, OR, XOR) that dispatch based on boolean type
#define REGISTER_LOGICAL_BITWISE_OP(mlir_op_name, logical_method, bitwise_method, reg_suffix) \
    static MPSGraphTensor* Handle_##reg_suffix(MPSGraph* g, mlir::Operation* op,              \
                                               ValueMap& values) {                            \
        MPSGraphTensor* lhs = GetInputTensor(values, op, 0);                                  \
        MPSGraphTensor* rhs = GetInputTensor(values, op, 1);                                  \
        if (!lhs || !rhs)                                                                     \
            return nullptr;                                                                   \
        if (isBooleanResult(op)) {                                                            \
            return [g logical_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];    \
        }                                                                                     \
        return [g bitwise_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];        \
    }                                                                                         \
    REGISTER_MPS_OP(mlir_op_name, Handle_##reg_suffix)

REGISTER_LOGICAL_BITWISE_OP("stablehlo.and", logicalAND, bitwiseAND, and);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.or", logicalOR, bitwiseOR, or);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.xor", logicalXOR, bitwiseXOR, xor);

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

// Shared helper for shift operations with overflow handling
static MPSGraphTensor* HandleShiftOp(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                     bool isLeftShift) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(values, op, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(op);
    if (bitWidth == 0)
        return nullptr;

    // Perform the shift
    MPSGraphTensor* shiftedResult = isLeftShift ? [g bitwiseLeftShiftWithPrimaryTensor:input
                                                                       secondaryTensor:shiftAmount
                                                                                  name:nil]
                                                : [g bitwiseRightShiftWithPrimaryTensor:input
                                                                        secondaryTensor:shiftAmount
                                                                                   name:nil];

    // Handle overflow: when shift >= bit_width, result should be 0
    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftAmount.dataType];
    MPSGraphTensor* overflowMask = [g greaterThanOrEqualToWithPrimaryTensor:shiftAmount
                                                            secondaryTensor:bitWidthTensor
                                                                       name:nil];
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];

    return [g selectWithPredicateTensor:overflowMask
                    truePredicateTensor:zeroTensor
                   falsePredicateTensor:shiftedResult
                                   name:nil];
}

static MPSGraphTensor* Handle_shift_left(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    return HandleShiftOp(g, op, values, /*isLeftShift=*/true);
}
REGISTER_MPS_OP("stablehlo.shift_left", Handle_shift_left);

static MPSGraphTensor* Handle_shift_right_logical(MPSGraph* g, mlir::Operation* op,
                                                  ValueMap& values) {
    return HandleShiftOp(g, op, values, /*isLeftShift=*/false);
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", Handle_shift_right_logical);

// Concatenate - joins tensors along a dimension
static MPSGraphTensor* Handle_concatenate(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
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
