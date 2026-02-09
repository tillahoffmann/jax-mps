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

static MPSDataType toUnsignedIntegerDataType(MPSDataType dataType) {
    switch (dataType) {
        case MPSDataTypeInt8:
            return MPSDataTypeUInt8;
        case MPSDataTypeInt16:
            return MPSDataTypeUInt16;
        case MPSDataTypeInt32:
            return MPSDataTypeUInt32;
        case MPSDataTypeInt64:
            return MPSDataTypeUInt64;
        case MPSDataTypeUInt8:
        case MPSDataTypeUInt16:
        case MPSDataTypeUInt32:
        case MPSDataTypeUInt64:
            return dataType;
        default:
            return MPSDataTypeInvalid;
    }
}

// StableHLO defines shift overflow in terms of bit-pattern shift counts.
// MPSGraph masks shift counts modulo bit-width, so we materialize overflow masks
// explicitly and use unsigned comparisons for shift amounts.
static MPSGraphTensor* BuildShiftOverflowMask(MPSGraph* g, MPSGraphTensor* shiftAmount,
                                              int bitWidth) {
    MPSDataType unsignedShiftType = toUnsignedIntegerDataType(shiftAmount.dataType);
    MPSGraphTensor* shiftForCompare = shiftAmount;
    if (unsignedShiftType != MPSDataTypeInvalid && unsignedShiftType != shiftAmount.dataType) {
        shiftForCompare = [g castTensor:shiftAmount toType:unsignedShiftType name:nil];
    }

    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftForCompare.dataType];
    return [g greaterThanOrEqualToWithPrimaryTensor:shiftForCompare
                                    secondaryTensor:bitWidthTensor
                                               name:nil];
}

enum class ShiftMode {
    kLeft,
    kRightLogical,
    kRightArithmetic,
};

static MPSGraphTensor* BuildShiftOverflowValue(MPSGraph* g, MPSGraphTensor* input, ShiftMode mode) {
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];
    if (mode != ShiftMode::kRightArithmetic) {
        return zeroTensor;
    }

    MPSGraphTensor* minusOneTensor = [g constantWithScalar:-1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* isNegative = [g lessThanWithPrimaryTensor:input
                                              secondaryTensor:zeroTensor
                                                         name:nil];
    return [g selectWithPredicateTensor:isNegative
                    truePredicateTensor:minusOneTensor
                   falsePredicateTensor:zeroTensor
                                   name:nil];
}

// Shared helper for shift operations with StableHLO overflow handling.
static MPSGraphTensor* HandleShiftOp(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                     ShiftMode mode) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(values, op, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraphTensor* shiftedResult =
        mode == ShiftMode::kLeft
            ? [g bitwiseLeftShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil]
            : [g bitwiseRightShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil];

    MPSGraphTensor* overflowMask = BuildShiftOverflowMask(g, shiftAmount, bitWidth);
    MPSGraphTensor* overflowValue = BuildShiftOverflowValue(g, input, mode);

    return [g selectWithPredicateTensor:overflowMask
                    truePredicateTensor:overflowValue
                   falsePredicateTensor:shiftedResult
                                   name:nil];
}

static MPSGraphTensor* Handle_shift_left(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    return HandleShiftOp(g, op, values, ShiftMode::kLeft);
}
REGISTER_MPS_OP("stablehlo.shift_left", Handle_shift_left);

static MPSGraphTensor* Handle_shift_right_logical(MPSGraph* g, mlir::Operation* op,
                                                  ValueMap& values) {
    return HandleShiftOp(g, op, values, ShiftMode::kRightLogical);
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", Handle_shift_right_logical);

static MPSGraphTensor* Handle_shift_right_arithmetic(MPSGraph* g, mlir::Operation* op,
                                                     ValueMap& values) {
    return HandleShiftOp(g, op, values, ShiftMode::kRightArithmetic);
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", Handle_shift_right_arithmetic);

static MPSGraphTensor* Handle_popcnt(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    if (op->getNumOperands() == 0) {
        MPS_LOG_ERROR("popcnt requires one integer operand\n");
        return nullptr;
    }
    auto operandType = op->getOperand(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
    if (!tensorType || !mlir::isa<mlir::IntegerType>(tensorType.getElementType())) {
        MPS_LOG_ERROR("popcnt requires an integer operand\n");
        return nullptr;
    }

    MPSGraphTensor* count = [g bitwisePopulationCountWithTensor:input name:nil];

    MPSDataType outType = GetResultMpsType(op);
    if (outType == MPSDataTypeInvalid)
        return count;
    return [g castTensor:count toType:outType name:nil];
}
REGISTER_MPS_OP("stablehlo.popcnt", Handle_popcnt);

}  // namespace jax_mps
