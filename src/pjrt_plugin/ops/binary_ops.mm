// Binary operations: add, subtract, multiply, divide, maximum, minimum,
// compare, select, clamp, next_after, dot, dot_general

#import <set>

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_MLIR_BINARY_OP("stablehlo.add", addition, add);
REGISTER_MLIR_BINARY_OP("stablehlo.subtract", subtraction, subtract);
REGISTER_MLIR_BINARY_OP("stablehlo.multiply", multiplication, multiply);
REGISTER_MLIR_BINARY_OP("stablehlo.divide", division, divide);
REGISTER_MLIR_BINARY_OP("stablehlo.maximum", maximum, maximum);
REGISTER_MLIR_BINARY_OP("stablehlo.minimum", minimum, minimum);
REGISTER_MLIR_BINARY_OP("stablehlo.remainder", modulo, remainder);
REGISTER_MLIR_BINARY_OP("stablehlo.power", power, power);
REGISTER_MLIR_BINARY_OP("stablehlo.atan2", atan2, atan2);

// Matrix multiplication (dot)
static ProcessResult HandleDot(HandlerContext& ctx) {
    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot: missing input tensor");
    MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                              secondaryTensor:rhs
                                                                         name:nil];
    return Result(ctx, result, "dot");
}
REGISTER_MPS_OP("stablehlo.dot", HandleDot);

// Generalized matrix multiplication (dot_general)
// Handles contracting dimensions and batch dimensions.
//
// For MPSGraph's matrixMultiplication:
//   - LHS shape (..., M, K), RHS shape (..., K, N) -> Result (..., M, N)
//   - Contracts LHS's last dim with RHS's second-to-last dim
//   - Broadcasts over leading batch dimensions
//
// We transpose inputs so contracting dims are in the right positions.
static ProcessResult HandleDotGeneral(HandlerContext& ctx) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(ctx.op);
    if (!dotOp) {
        return ProcessResult::Error("dot_general: expected DotGeneralOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot_general: missing input tensor");

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    NSArray<NSNumber*>* lhsShape = lhs.shape;
    NSArray<NSNumber*>* rhsShape = rhs.shape;
    NSUInteger lhsRank = lhsShape.count;
    NSUInteger rhsRank = rhsShape.count;

    MPSGraphTensor* result = nil;

    // Simple case: standard 2D matmul with no batch dims and single contraction
    if (lhsBatchDims.empty() && rhsBatchDims.empty() && lhsRank == 2 && rhsRank == 2 &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
        int64_t lhsContractDim = lhsContractingDims[0];
        int64_t rhsContractDim = rhsContractingDims[0];

        // Standard matmul: LHS contracts on dim 1, RHS contracts on dim 0
        if (lhsContractDim == 1 && rhsContractDim == 0) {
            result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                      secondaryTensor:rhs
                                                                 name:nil];
        }
        // LHS contracts on dim 0: transpose LHS
        else if (lhsContractDim == 0 && rhsContractDim == 0) {
            MPSGraphTensor* lhsT = [ctx.graph transposeTensor:lhs permutation:@[@1, @0] name:nil];
            result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsT
                                                      secondaryTensor:rhs
                                                                 name:nil];
        }
        // RHS contracts on dim 1: transpose RHS
        else if (lhsContractDim == 1 && rhsContractDim == 1) {
            MPSGraphTensor* rhsT = [ctx.graph transposeTensor:rhs permutation:@[@1, @0] name:nil];
            result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                      secondaryTensor:rhsT
                                                                 name:nil];
        }
        // Both transpose
        else if (lhsContractDim == 0 && rhsContractDim == 1) {
            MPSGraphTensor* lhsT = [ctx.graph transposeTensor:lhs permutation:@[@1, @0] name:nil];
            MPSGraphTensor* rhsT = [ctx.graph transposeTensor:rhs permutation:@[@1, @0] name:nil];
            result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsT
                                                      secondaryTensor:rhsT
                                                                 name:nil];
        }
    }

    // Batched case: both have same number of batch dims (>0) and single contraction
    if (!result && !lhsBatchDims.empty() && lhsBatchDims.size() == rhsBatchDims.size() &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
        // Verify contracting dimensions have the same size
        int64_t lhsContractDim = lhsContractingDims[0];
        int64_t rhsContractDim = rhsContractingDims[0];
        NSNumber* lhsContractSize = lhsShape[(NSUInteger)lhsContractDim];
        NSNumber* rhsContractSize = rhsShape[(NSUInteger)rhsContractDim];

        if ([lhsContractSize isEqualToNumber:rhsContractSize]) {
            // Convert to sets for fast lookup
            std::set<int64_t> lhsBatchSet(lhsBatchDims.begin(), lhsBatchDims.end());
            std::set<int64_t> rhsBatchSet(rhsBatchDims.begin(), rhsBatchDims.end());
            std::set<int64_t> lhsContractSet(lhsContractingDims.begin(), lhsContractingDims.end());
            std::set<int64_t> rhsContractSet(rhsContractingDims.begin(), rhsContractingDims.end());

            // Build permutation for LHS: batch dims, free dims, contracting dims
            NSMutableArray<NSNumber*>* lhsPerm = [NSMutableArray array];
            for (int64_t d : lhsBatchDims)
                [lhsPerm addObject:@(d)];
            for (int64_t d = 0; d < static_cast<int64_t>(lhsRank); d++) {
                if (lhsBatchSet.count(d) == 0 && lhsContractSet.count(d) == 0)
                    [lhsPerm addObject:@(d)];
            }
            for (int64_t d : lhsContractingDims)
                [lhsPerm addObject:@(d)];

            // Build permutation for RHS: batch dims, contracting dims, free dims
            NSMutableArray<NSNumber*>* rhsPerm = [NSMutableArray array];
            for (int64_t d : rhsBatchDims)
                [rhsPerm addObject:@(d)];
            for (int64_t d : rhsContractingDims)
                [rhsPerm addObject:@(d)];
            for (int64_t d = 0; d < static_cast<int64_t>(rhsRank); d++) {
                if (rhsBatchSet.count(d) == 0 && rhsContractSet.count(d) == 0)
                    [rhsPerm addObject:@(d)];
            }

            // Check if permutation is identity
            auto isIdentityPerm = [](NSArray<NSNumber*>* perm, NSUInteger rank) {
                if (perm.count != rank)
                    return false;
                for (NSUInteger i = 0; i < rank; i++) {
                    if (perm[i].unsignedIntegerValue != i)
                        return false;
                }
                return true;
            };

            MPSGraphTensor* lhsT = lhs;
            if (!isIdentityPerm(lhsPerm, lhsRank)) {
                lhsT = [ctx.graph transposeTensor:lhs permutation:lhsPerm name:nil];
            }

            MPSGraphTensor* rhsT = rhs;
            if (!isIdentityPerm(rhsPerm, rhsRank)) {
                rhsT = [ctx.graph transposeTensor:rhs permutation:rhsPerm name:nil];
            }

            result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsT
                                                      secondaryTensor:rhsT
                                                                 name:nil];
        }
    }

    // Fallback for unhandled cases - try simple matmul (may fail for incompatible shapes)
    if (!result) {
        MPS_LOG_WARN(
            "dot_general with complex contracting/batch dims, falling back to simple matmul. "
            "LHS contracting: %lld, RHS contracting: %lld\n",
            lhsContractingDims.empty() ? -1 : lhsContractingDims[0],
            rhsContractingDims.empty() ? -1 : rhsContractingDims[0]);
        result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    }

    return Result(ctx, result, "dot_general");
}
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// Compare operation
static ProcessResult HandleCompare(HandlerContext& ctx) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(ctx.op);
    if (!compareOp) {
        return ProcessResult::Error("compare: expected CompareOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("compare: missing input tensor");

    auto direction = compareOp.getComparisonDirection();
    using Dir = mlir::stablehlo::ComparisonDirection;

    MPSGraphTensor* result = nil;
    switch (direction) {
        case Dir::LT:
            result = [ctx.graph lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::LE:
            result = [ctx.graph lessThanOrEqualToWithPrimaryTensor:lhs
                                                   secondaryTensor:rhs
                                                              name:nil];
            break;
        case Dir::GT:
            result = [ctx.graph greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::GE:
            result = [ctx.graph greaterThanOrEqualToWithPrimaryTensor:lhs
                                                      secondaryTensor:rhs
                                                                 name:nil];
            break;
        case Dir::EQ:
            result = [ctx.graph equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::NE:
            result = [ctx.graph notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        default:
            return ProcessResult::Error("compare: unknown compare direction");
    }

    return Result(ctx, result, "compare");
}
REGISTER_MPS_OP("stablehlo.compare", HandleCompare);

// Select operation (conditional selection: pred ? true_val : false_val)
static ProcessResult HandleSelect(HandlerContext& ctx) {
    MPSGraphTensor* pred = GetInputTensor(ctx, 0);
    MPSGraphTensor* onTrue = GetInputTensor(ctx, 1);
    MPSGraphTensor* onFalse = GetInputTensor(ctx, 2);
    if (!pred || !onTrue || !onFalse)
        return ProcessResult::Error("select: missing input tensor");

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:pred
                                              truePredicateTensor:onTrue
                                             falsePredicateTensor:onFalse
                                                             name:nil];
    return Result(ctx, result, "select");
}
REGISTER_MPS_OP("stablehlo.select", HandleSelect);

// Clamp operation: clamp(min, x, max)
static ProcessResult HandleClamp(HandlerContext& ctx) {
    MPSGraphTensor* minVal = GetInputTensor(ctx, 0);
    MPSGraphTensor* operand = GetInputTensor(ctx, 1);
    MPSGraphTensor* maxVal = GetInputTensor(ctx, 2);
    if (!minVal || !operand || !maxVal)
        return ProcessResult::Error("clamp: missing input tensor");

    MPSGraphTensor* result = [ctx.graph clampWithTensor:operand
                                         minValueTensor:minVal
                                         maxValueTensor:maxVal
                                                   name:nil];
    return Result(ctx, result, "clamp");
}
REGISTER_MPS_OP("stablehlo.clamp", HandleClamp);

// next_after(x, y) - returns the next representable floating point value from x towards y
// Implementation follows IEEE 754 nextafter semantics:
// 1. If x == y, return y
// 2. If x or y is NaN, return NaN
// 3. If x == 0, return smallest subnormal with sign of y
// 4. Otherwise, treat x as integer bits and increment/decrement based on direction
static ProcessResult HandleNextAfter(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    MPSGraphTensor* y = GetInputTensor(ctx, 1);
    if (!x || !y)
        return ProcessResult::Error("next_after: missing input tensor");

    MPSDataType dtype = x.dataType;

    // Handle scalar tensors - MPS reinterpretCast doesn't support rank-0
    NSArray<NSNumber*>* xShape = x.shape;
    bool isScalar = (xShape.count == 0);
    if (isScalar) {
        x = [ctx.graph reshapeTensor:x withShape:@[@1] name:nil];
        y = [ctx.graph reshapeTensor:y withShape:@[@1] name:nil];
    }

    // Constants
    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:dtype];
    MPSGraphTensor* one_int = [ctx.graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg_one_int = [ctx.graph constantWithScalar:-1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_positive_int = [ctx.graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_negative_int = [ctx.graph constantWithScalar:0x80000001
                                                            dataType:MPSDataTypeInt32];

    // Bitcast x to int32 (reinterpret bits)
    MPSGraphTensor* x_as_int = [ctx.graph reinterpretCastTensor:x toType:MPSDataTypeInt32 name:nil];

    // Check if x == y
    MPSGraphTensor* x_eq_y = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:y name:nil];

    // Check if x is zero
    MPSGraphTensor* x_is_zero = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Check if y > 0 (to determine direction when x == 0)
    MPSGraphTensor* y_gt_zero = [ctx.graph greaterThanWithPrimaryTensor:y
                                                        secondaryTensor:zero
                                                                   name:nil];

    // When x == 0, return smallest positive or negative subnormal
    MPSGraphTensor* zero_result_int = [ctx.graph selectWithPredicateTensor:y_gt_zero
                                                       truePredicateTensor:min_positive_int
                                                      falsePredicateTensor:min_negative_int
                                                                      name:nil];
    MPSGraphTensor* zero_result = [ctx.graph reinterpretCastTensor:zero_result_int
                                                            toType:dtype
                                                              name:nil];

    // For non-zero x, determine direction and increment/decrement
    // If x > 0 and y > x, or x < 0 and y > x: increment (add 1 to int representation)
    // If x > 0 and y < x, or x < 0 and y < x: decrement (subtract 1 from int representation)
    MPSGraphTensor* y_gt_x = [ctx.graph greaterThanWithPrimaryTensor:y secondaryTensor:x name:nil];

    // x > 0
    MPSGraphTensor* x_gt_zero = [ctx.graph greaterThanWithPrimaryTensor:x
                                                        secondaryTensor:zero
                                                                   name:nil];

    // Determine if we should increment the int representation
    // Increment when: (x > 0 && y > x) || (x < 0 && y < x)
    // Which simplifies to: (x > 0) == (y > x)
    MPSGraphTensor* should_increment = [ctx.graph equalWithPrimaryTensor:x_gt_zero
                                                         secondaryTensor:y_gt_x
                                                                    name:nil];

    // Compute the delta (+1 or -1)
    MPSGraphTensor* delta = [ctx.graph selectWithPredicateTensor:should_increment
                                             truePredicateTensor:one_int
                                            falsePredicateTensor:neg_one_int
                                                            name:nil];

    // Add delta to x_as_int
    MPSGraphTensor* result_int = [ctx.graph additionWithPrimaryTensor:x_as_int
                                                      secondaryTensor:delta
                                                                 name:nil];

    // Bitcast back to float
    MPSGraphTensor* non_zero_result = [ctx.graph reinterpretCastTensor:result_int
                                                                toType:dtype
                                                                  name:nil];

    // Select between zero and non-zero cases
    MPSGraphTensor* non_equal_result = [ctx.graph selectWithPredicateTensor:x_is_zero
                                                        truePredicateTensor:zero_result
                                                       falsePredicateTensor:non_zero_result
                                                                       name:nil];

    // If x == y, return y; otherwise return the computed result
    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:x_eq_y
                                              truePredicateTensor:y
                                             falsePredicateTensor:non_equal_result
                                                             name:nil];

    // Reshape back to scalar if needed
    if (isScalar) {
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(ctx, result, "next_after");
}
REGISTER_MPS_OP("chlo.next_after", HandleNextAfter);

}  // namespace jax_mps
