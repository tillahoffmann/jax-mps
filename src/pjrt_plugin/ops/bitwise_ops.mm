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
#define REGISTER_LOGICAL_BITWISE_OP(mlir_op_name, logical_method, bitwise_method, reg_suffix) \
    static ProcessResult Handle##reg_suffix(HandlerContext& ctx) {                            \
        MPSGraphTensor* lhs = GetInputTensor(ctx, 0);                                         \
        MPSGraphTensor* rhs = GetInputTensor(ctx, 1);                                         \
        if (!lhs || !rhs)                                                                     \
            return ProcessResult::Error(#reg_suffix ": missing input tensor");                \
        MPSGraphTensor* result = nil;                                                         \
        if (isBooleanResult(ctx.op)) {                                                        \
            result = [ctx.graph logical_method##WithPrimaryTensor:lhs                         \
                                                  secondaryTensor:rhs                         \
                                                             name:nil];                       \
        } else {                                                                              \
            result = [ctx.graph bitwise_method##WithPrimaryTensor:lhs                         \
                                                  secondaryTensor:rhs                         \
                                                             name:nil];                       \
        }                                                                                     \
        return Result(ctx, result, #reg_suffix);                                              \
    }                                                                                         \
    REGISTER_MPS_OP(mlir_op_name, Handle##reg_suffix)

REGISTER_LOGICAL_BITWISE_OP("stablehlo.and", logicalAND, bitwiseAND, And);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.or", logicalOR, bitwiseOR, Or);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.xor", logicalXOR, bitwiseXOR, Xor);

static ProcessResult HandleNot(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("not: missing input tensor");

    MPSGraphTensor* result = nil;
    if (isBooleanResult(ctx.op)) {
        MPSGraphTensor* falseTensor = [ctx.graph constantWithScalar:0 dataType:input.dataType];
        result = [ctx.graph equalWithPrimaryTensor:input secondaryTensor:falseTensor name:nil];
    } else {
        result = [ctx.graph bitwiseNOTWithTensor:input name:nil];
    }

    return Result(ctx, result, "not");
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
static MPSGraphTensor* HandleShiftOp(HandlerContext& ctx, ShiftMode mode) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(ctx, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(ctx.op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraphTensor* shiftedResult = mode == ShiftMode::kLeft
                                        ? [ctx.graph bitwiseLeftShiftWithPrimaryTensor:input
                                                                       secondaryTensor:shiftAmount
                                                                                  name:nil]
                                        : [ctx.graph bitwiseRightShiftWithPrimaryTensor:input
                                                                        secondaryTensor:shiftAmount
                                                                                   name:nil];

    MPSGraphTensor* overflowMask = BuildShiftOverflowMask(ctx.graph, shiftAmount, bitWidth);
    MPSGraphTensor* overflowValue = BuildShiftOverflowValue(ctx.graph, input, mode);

    return [ctx.graph selectWithPredicateTensor:overflowMask
                            truePredicateTensor:overflowValue
                           falsePredicateTensor:shiftedResult
                                           name:nil];
}

static ProcessResult HandleShiftLeft(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kLeft);
    return Result(ctx, result, "shift_left");
}
REGISTER_MPS_OP("stablehlo.shift_left", HandleShiftLeft);

static ProcessResult HandleShiftRightLogical(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightLogical);
    return Result(ctx, result, "shift_right_logical");
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", HandleShiftRightLogical);

static ProcessResult HandleShiftRightArithmetic(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightArithmetic);
    return Result(ctx, result, "shift_right_arithmetic");
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic);

static ProcessResult HandlePopcnt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("popcnt: missing input tensor");

    if (ctx.op->getNumOperands() == 0) {
        return ProcessResult::Error("popcnt: requires one integer operand");
    }
    auto operandType = ctx.op->getOperand(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
    if (!tensorType || !mlir::isa<mlir::IntegerType>(tensorType.getElementType())) {
        return ProcessResult::Error("popcnt: requires an integer operand");
    }

    MPSGraphTensor* count = [ctx.graph bitwisePopulationCountWithTensor:input name:nil];
    if (!count)
        return ProcessResult::Error("popcnt: bitwisePopulationCount returned null");

    MPSDataType outType = GetResultMpsType(ctx.op);
    MPSGraphTensor* result = count;
    if (outType != MPSDataTypeInvalid && count.dataType != outType) {
        result = [ctx.graph castTensor:count toType:outType name:nil];
    }

    return Result(ctx, result, "popcnt");
}
REGISTER_MPS_OP("stablehlo.popcnt", HandlePopcnt);

}  // namespace jax_mps
