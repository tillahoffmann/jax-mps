// Binary operations: add, subtract, multiply, divide, maximum, minimum,
// compare, select, clamp, next_after, dot, dot_general

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
static MPSGraphTensor* Handle_dot(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
static bool _reg_dot = ::jax_mps::OpRegistry::Register("stablehlo.dot", Handle_dot);

// Generalized matrix multiplication (dot_general)
// Handles contracting dimensions and batch dimensions
static MPSGraphTensor* Handle_dot_general(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        MPS_LOG_ERROR(" Expected DotGeneralOp\n");
        return nullptr;
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    NSArray<NSNumber*>* lhsShape = lhs.shape;
    NSArray<NSNumber*>* rhsShape = rhs.shape;
    NSUInteger lhsRank = lhsShape.count;
    NSUInteger rhsRank = rhsShape.count;

    // Simple case: standard 2D matmul with contraction on last/first dims
    // LHS: (M, K), RHS: (K, N) -> (M, N)
    if (lhsBatchDims.empty() && rhsBatchDims.empty() && lhsRank == 2 && rhsRank == 2 &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
        int64_t lhsContractDim = lhsContractingDims[0];
        int64_t rhsContractDim = rhsContractingDims[0];

        // Standard matmul: LHS contracts on dim 1, RHS contracts on dim 0
        if (lhsContractDim == 1 && rhsContractDim == 0) {
            return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        }

        // LHS contracts on dim 0: need to transpose LHS
        // (K, M) @ (K, N) -> transpose LHS to (M, K), then (M, K) @ ...
        if (lhsContractDim == 0 && rhsContractDim == 0) {
            // Need: LHS^T @ RHS where inner dims match
            // LHS is (K, M), LHS^T is (M, K)
            // RHS is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* lhsT = [g transposeTensor:lhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhsT secondaryTensor:rhs name:nil];
        }

        // LHS contracts on dim 1, RHS contracts on dim 1: need to transpose RHS
        if (lhsContractDim == 1 && rhsContractDim == 1) {
            // LHS is (M, K), RHS is (N, K), RHS^T is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* rhsT = [g transposeTensor:rhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhsT name:nil];
        }

        // LHS contracts on dim 0, RHS contracts on dim 1: transpose both
        if (lhsContractDim == 0 && rhsContractDim == 1) {
            // LHS is (K, M), RHS is (N, K)
            // LHS^T is (M, K), RHS^T is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* lhsT = [g transposeTensor:lhs permutation:@[@1, @0] name:nil];
            MPSGraphTensor* rhsT = [g transposeTensor:rhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhsT secondaryTensor:rhsT name:nil];
        }
    }

    // Fall back to simple matmul for unhandled cases
    MPS_LOG_WARN("dot_general with complex contracting/batch dims, falling back to simple matmul. "
                 "LHS contracting: %lld, RHS contracting: %lld\n",
                 lhsContractingDims.empty() ? -1 : lhsContractingDims[0],
                 rhsContractingDims.empty() ? -1 : rhsContractingDims[0]);
    return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
static bool _reg_dot_general =
    ::jax_mps::OpRegistry::Register("stablehlo.dot_general", Handle_dot_general);

// Compare operation
static MPSGraphTensor* Handle_compare(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        MPS_LOG_ERROR(" Expected CompareOp\n");
        return nullptr;
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;

    auto direction = compareOp.getComparisonDirection();
    using Dir = mlir::stablehlo::ComparisonDirection;

    switch (direction) {
        case Dir::LT:
            return [g lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::LE:
            return [g lessThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::GT:
            return [g greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::GE:
            return [g greaterThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::EQ:
            return [g equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::NE:
            return [g notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        default:
            MPS_LOG_ERROR(" Unknown compare direction\n");
            return nullptr;
    }
}
REGISTER_MPS_OP("stablehlo.compare", Handle_compare);

// Select operation (conditional selection: pred ? true_val : false_val)
static MPSGraphTensor* Handle_select(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* pred = GetInputTensor(values, op, 0);
    MPSGraphTensor* onTrue = GetInputTensor(values, op, 1);
    MPSGraphTensor* onFalse = GetInputTensor(values, op, 2);
    if (!pred || !onTrue || !onFalse)
        return nullptr;

    return [g selectWithPredicateTensor:pred
                    truePredicateTensor:onTrue
                   falsePredicateTensor:onFalse
                                   name:nil];
}
REGISTER_MPS_OP("stablehlo.select", Handle_select);

// Clamp operation: clamp(min, x, max)
static MPSGraphTensor* Handle_clamp(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* minVal = GetInputTensor(values, op, 0);
    MPSGraphTensor* operand = GetInputTensor(values, op, 1);
    MPSGraphTensor* maxVal = GetInputTensor(values, op, 2);
    if (!minVal || !operand || !maxVal)
        return nullptr;

    return [g clampWithTensor:operand minValueTensor:minVal maxValueTensor:maxVal name:nil];
}
REGISTER_MPS_OP("stablehlo.clamp", Handle_clamp);

// next_after(x, y) - returns the next representable floating point value from x towards y
// Implementation follows IEEE 754 nextafter semantics:
// 1. If x == y, return y
// 2. If x or y is NaN, return NaN
// 3. If x == 0, return smallest subnormal with sign of y
// 4. Otherwise, treat x as integer bits and increment/decrement based on direction
static MPSGraphTensor* Handle_next_after(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* x = GetInputTensor(values, op, 0);
    MPSGraphTensor* y = GetInputTensor(values, op, 1);
    if (!x || !y)
        return nullptr;

    MPSDataType dtype = x.dataType;

    // Handle scalar tensors - MPS reinterpretCast doesn't support rank-0
    NSArray<NSNumber*>* xShape = x.shape;
    bool isScalar = (xShape.count == 0);
    if (isScalar) {
        x = [g reshapeTensor:x withShape:@[@1] name:nil];
        y = [g reshapeTensor:y withShape:@[@1] name:nil];
    }

    // Constants
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dtype];
    MPSGraphTensor* one_int = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg_one_int = [g constantWithScalar:-1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_positive_int = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_negative_int = [g constantWithScalar:0x80000001 dataType:MPSDataTypeInt32];

    // Bitcast x to int32 (reinterpret bits)
    MPSGraphTensor* x_as_int = [g reinterpretCastTensor:x toType:MPSDataTypeInt32 name:nil];

    // Check if x == y
    MPSGraphTensor* x_eq_y = [g equalWithPrimaryTensor:x secondaryTensor:y name:nil];

    // Check if x is zero
    MPSGraphTensor* x_is_zero = [g equalWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Check if y > 0 (to determine direction when x == 0)
    MPSGraphTensor* y_gt_zero = [g greaterThanWithPrimaryTensor:y secondaryTensor:zero name:nil];

    // When x == 0, return smallest positive or negative subnormal
    MPSGraphTensor* zero_result_int = [g selectWithPredicateTensor:y_gt_zero
                                               truePredicateTensor:min_positive_int
                                              falsePredicateTensor:min_negative_int
                                                              name:nil];
    MPSGraphTensor* zero_result = [g reinterpretCastTensor:zero_result_int toType:dtype name:nil];

    // For non-zero x, determine direction and increment/decrement
    // If x > 0 and y > x, or x < 0 and y > x: increment (add 1 to int representation)
    // If x > 0 and y < x, or x < 0 and y < x: decrement (subtract 1 from int representation)
    MPSGraphTensor* y_gt_x = [g greaterThanWithPrimaryTensor:y secondaryTensor:x name:nil];

    // x > 0
    MPSGraphTensor* x_gt_zero = [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Determine if we should increment the int representation
    // Increment when: (x > 0 && y > x) || (x < 0 && y < x)
    // Which simplifies to: (x > 0) == (y > x)
    MPSGraphTensor* should_increment = [g equalWithPrimaryTensor:x_gt_zero
                                                 secondaryTensor:y_gt_x
                                                            name:nil];

    // Compute the delta (+1 or -1)
    MPSGraphTensor* delta = [g selectWithPredicateTensor:should_increment
                                     truePredicateTensor:one_int
                                    falsePredicateTensor:neg_one_int
                                                    name:nil];

    // Add delta to x_as_int
    MPSGraphTensor* result_int = [g additionWithPrimaryTensor:x_as_int
                                              secondaryTensor:delta
                                                         name:nil];

    // Bitcast back to float
    MPSGraphTensor* non_zero_result = [g reinterpretCastTensor:result_int toType:dtype name:nil];

    // Select between zero and non-zero cases
    MPSGraphTensor* non_equal_result = [g selectWithPredicateTensor:x_is_zero
                                                truePredicateTensor:zero_result
                                               falsePredicateTensor:non_zero_result
                                                               name:nil];

    // If x == y, return y; otherwise return the computed result
    MPSGraphTensor* result = [g selectWithPredicateTensor:x_eq_y
                                      truePredicateTensor:y
                                     falsePredicateTensor:non_equal_result
                                                     name:nil];

    // Reshape back to scalar if needed
    if (isScalar) {
        result = [g reshapeTensor:result withShape:@[] name:nil];
    }

    return result;
}
REGISTER_MPS_OP("chlo.next_after", Handle_next_after);

}  // namespace jax_mps
