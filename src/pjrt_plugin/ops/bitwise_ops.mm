// Bitwise operations: and, or, xor, shift_left, shift_right_logical

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

static MPSGraphTensor* Handle_not(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    if (isBooleanResult(op)) {
        MPSGraphTensor* falseTensor = [g constantWithScalar:0 dataType:input.dataType];
        return [g equalWithPrimaryTensor:input secondaryTensor:falseTensor name:nil];
    }
    return [g bitwiseNOTWithTensor:input name:nil];
}
REGISTER_MPS_OP("stablehlo.not", Handle_not);

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

static int getOperandBitWidth(mlir::Operation* op, unsigned operandIdx) {
    if (op->getNumOperands() <= operandIdx)
        return 0;
    auto operandType = op->getOperand(operandIdx).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
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

static MPSGraphTensor* Handle_shift_right_arithmetic(MPSGraph* g, mlir::Operation* op,
                                                     ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(values, op, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraphTensor* shifted = [g bitwiseRightShiftWithPrimaryTensor:input
                                                    secondaryTensor:shiftAmount
                                                               name:nil];

    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftAmount.dataType];
    MPSGraphTensor* overflowMask = [g greaterThanOrEqualToWithPrimaryTensor:shiftAmount
                                                            secondaryTensor:bitWidthTensor
                                                                       name:nil];

    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* minusOneTensor =
        [g constantWithScalar:-1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* isNegative =
        [g lessThanWithPrimaryTensor:input secondaryTensor:zeroTensor name:nil];
    MPSGraphTensor* overflowValue = [g selectWithPredicateTensor:isNegative
                                              truePredicateTensor:minusOneTensor
                                             falsePredicateTensor:zeroTensor
                                                             name:nil];

    return [g selectWithPredicateTensor:overflowMask
                    truePredicateTensor:overflowValue
                   falsePredicateTensor:shifted
                                   name:nil];
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", Handle_shift_right_arithmetic);

static MPSGraphTensor* Handle_popcnt(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    int bitWidth = getOperandBitWidth(op, 0);
    if (bitWidth <= 0) {
        MPS_LOG_ERROR("popcnt requires an integer operand\n");
        return nullptr;
    }

    MPSGraphTensor* one = [g constantWithScalar:1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* count = [g constantWithScalar:0 shape:@[@1] dataType:MPSDataTypeInt32];

    for (int i = 0; i < bitWidth; ++i) {
        MPSGraphTensor* shift = [g constantWithScalar:i shape:@[@1] dataType:input.dataType];
        MPSGraphTensor* shifted =
            [g bitwiseRightShiftWithPrimaryTensor:input secondaryTensor:shift name:nil];
        MPSGraphTensor* bit = [g bitwiseANDWithPrimaryTensor:shifted secondaryTensor:one name:nil];
        MPSGraphTensor* bit32 = [g castTensor:bit toType:MPSDataTypeInt32 name:nil];
        count = [g additionWithPrimaryTensor:count secondaryTensor:bit32 name:nil];
    }

    MPSDataType outType = GetResultMpsType(op);
    if (outType == MPSDataTypeInvalid)
        return count;
    return [g castTensor:count toType:outType name:nil];
}
REGISTER_MPS_OP("stablehlo.popcnt", Handle_popcnt);

}  // namespace jax_mps
