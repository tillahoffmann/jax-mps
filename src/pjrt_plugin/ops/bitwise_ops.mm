// Bitwise operations: and, or, xor, not, shift_left, shift_right_logical,
// shift_right_arithmetic, popcnt

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
#define REGISTER_LOGICAL_BITWISE_OP(mlir_op_name, logical_method, bitwise_method, reg_suffix)      \
    static ProcessResult Handle_##reg_suffix(MPSGraph* g, mlir::Operation* op, ValueMap& values) { \
        MPSGraphTensor* lhs = GetInputTensor(values, op, 0);                                       \
        MPSGraphTensor* rhs = GetInputTensor(values, op, 1);                                       \
        if (!lhs || !rhs)                                                                          \
            return ProcessResult::Error(#reg_suffix ": missing input tensor");                     \
        MPSGraphTensor* result = nil;                                                              \
        if (isBooleanResult(op)) {                                                                 \
            result = [g logical_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];       \
        } else {                                                                                   \
            result = [g bitwise_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];       \
        }                                                                                          \
        return Result(values, op, result, #reg_suffix);                                            \
    }                                                                                              \
    REGISTER_MPS_OP(mlir_op_name, Handle_##reg_suffix)

REGISTER_LOGICAL_BITWISE_OP("stablehlo.and", logicalAND, bitwiseAND, and);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.or", logicalOR, bitwiseOR, or);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.xor", logicalXOR, bitwiseXOR, xor);

static ProcessResult HandleNot(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("not: missing input tensor");

    MPSGraphTensor* result = nil;
    if (isBooleanResult(op)) {
        MPSGraphTensor* falseTensor = [g constantWithScalar:0 dataType:input.dataType];
        result = [g equalWithPrimaryTensor:input secondaryTensor:falseTensor name:nil];
    } else {
        result = [g bitwiseNOTWithTensor:input name:nil];
    }

    return Result(values, op, result, "not");
}
REGISTER_MPS_OP("stablehlo.not", HandleNot);

// Helper to get bit width from tensor's element type
static int getBitWidth(mlir::Operation* op) {
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return 0;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return static_cast<int>(intType.getWidth());
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

static ProcessResult HandleShiftLeft(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* result = HandleShiftOp(g, op, values, ShiftMode::kLeft);
    return Result(values, op, result, "shift_left");
}
REGISTER_MPS_OP("stablehlo.shift_left", HandleShiftLeft);

static ProcessResult HandleShiftRightLogical(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* result = HandleShiftOp(g, op, values, ShiftMode::kRightLogical);
    return Result(values, op, result, "shift_right_logical");
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", HandleShiftRightLogical);

static ProcessResult HandleShiftRightArithmetic(MPSGraph* g, mlir::Operation* op,
                                                ValueMap& values) {
    MPSGraphTensor* result = HandleShiftOp(g, op, values, ShiftMode::kRightArithmetic);
    return Result(values, op, result, "shift_right_arithmetic");
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic);

static ProcessResult HandlePopcnt(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("popcnt: missing input tensor");

    if (op->getNumOperands() == 0) {
        return ProcessResult::Error("popcnt: requires one integer operand");
    }
    auto operandType = op->getOperand(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
    if (!tensorType || !mlir::isa<mlir::IntegerType>(tensorType.getElementType())) {
        return ProcessResult::Error("popcnt: requires an integer operand");
    }

    MPSGraphTensor* count = [g bitwisePopulationCountWithTensor:input name:nil];
    if (!count)
        return ProcessResult::Error("popcnt: bitwisePopulationCount returned null");

    MPSDataType outType = GetResultMpsType(op);
    MPSGraphTensor* result = count;
    if (outType != MPSDataTypeInvalid && count.dataType != outType) {
        result = [g castTensor:count toType:outType name:nil];
    }

    return Result(values, op, result, "popcnt");
}
REGISTER_MPS_OP("stablehlo.popcnt", HandlePopcnt);

}  // namespace jax_mps
