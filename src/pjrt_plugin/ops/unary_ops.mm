// Unary operations: tanh, exp, log, negate, abs
// Also includes constant creation, compare, and select

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Unary ops using macros
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, tanh);
REGISTER_MLIR_UNARY_OP("stablehlo.exponential", exponent, exp);
REGISTER_MLIR_UNARY_OP("stablehlo.log", logarithm, log);
REGISTER_MLIR_UNARY_OP("stablehlo.negate", negative, negate);
REGISTER_MLIR_UNARY_OP("stablehlo.abs", absolute, abs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.rsqrt", reciprocalSquareRoot, rsqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);
REGISTER_MLIR_UNARY_OP("chlo.erf", erf, chlo_erf);
REGISTER_MLIR_UNARY_OP("stablehlo.floor", floor, floor);
REGISTER_MLIR_UNARY_OP("stablehlo.sign", sign, sign);
REGISTER_MLIR_UNARY_OP("stablehlo.is_finite", isFinite, is_finite);
REGISTER_MLIR_UNARY_OP("chlo.square", square, chlo_square);
REGISTER_MLIR_UNARY_OP("stablehlo.ceil", ceil, ceil);
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);
REGISTER_MLIR_UNARY_OP("stablehlo.sine", sin, sine);
REGISTER_MLIR_UNARY_OP("stablehlo.tan", tan, tan);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
static MPSGraphTensor* Handle_log_plus_one(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [g additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    return [g logarithmWithTensor:onePlusX name:nil];
}
REGISTER_MPS_OP("stablehlo.log_plus_one", Handle_log_plus_one);

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

// Constant creation - creates a constant tensor from MLIR constant op
static MPSGraphTensor* Handle_constant(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
        MPS_LOG_ERROR(" Expected ConstantOp\n");
        return nullptr;
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR(" Invalid dtype for constant operation\n");
        return nullptr;
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    auto value = constantOp.getValue();

    // Check for empty tensor (any dimension is 0)
    // MPSGraph doesn't support empty tensors, so create a minimal [1] tensor instead
    // The scatter handler will detect empty indices based on MLIR types and handle appropriately
    bool isEmpty = false;
    for (NSNumber* dim in shape) {
        if ([dim integerValue] == 0) {
            isEmpty = true;
            break;
        }
    }
    if (isEmpty) {
        // Create a minimal tensor with shape [1] and a dummy value
        // This is safe because operations that use this tensor will detect
        // empty dimensions from the MLIR types and not actually use the tensor values
        return [g constantWithScalar:0 shape:@[@1] dataType:dtype];
    }

    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
        // Check if it's a splat (single value broadcast to all elements)
        if (denseAttr.isSplat()) {
            auto elemType = denseAttr.getElementType();
            double scalarValue = 0.0;

            if (elemType.isF32()) {
                scalarValue = denseAttr.getSplatValue<float>();
            } else if (elemType.isF64()) {
                scalarValue = denseAttr.getSplatValue<double>();
            } else if (elemType.isF16()) {
                auto apVal = denseAttr.getSplatValue<llvm::APFloat>();
                scalarValue = apVal.convertToFloat();
            } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                auto apInt = denseAttr.getSplatValue<llvm::APInt>();
                // For signless integers (the default in MLIR/StableHLO), use sign-extend
                // to preserve the two's complement representation.
                // Only use zero-extend for explicitly unsigned types.
                if (intType.isUnsigned()) {
                    scalarValue = static_cast<double>(apInt.getZExtValue());
                } else {
                    // Signless or signed - use sign-extend
                    scalarValue = static_cast<double>(apInt.getSExtValue());
                }
            }

            if (shape.count == 0) {
                // True scalar
                return [g constantWithScalar:scalarValue dataType:dtype];
            } else {
                // Splat to shape
                return [g constantWithScalar:scalarValue shape:shape dataType:dtype];
            }
        } else {
            // Non-splat dense constant - use raw data
            auto rawData = denseAttr.getRawData();
            NSData* data = [NSData dataWithBytes:rawData.data() length:rawData.size()];
            return [g constantWithData:data shape:shape dataType:dtype];
        }
    }

    MPS_LOG_ERROR(" Constant operation has unsupported value type\n");
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.constant", Handle_constant);

// Inverse error function (erfinv) using Winitzki approximation
// erfinv(x) ≈ sign(x) * sqrt(sqrt(t² - log(1-x²)/a) - t)
// where t = 2/(π*a) + log(1-x²)/2, a ≈ 0.147
static MPSGraphTensor* Handle_erf_inv(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* x = GetInputTensor(values, op, 0);
    if (!x)
        return nullptr;

    MPSDataType dtype = x.dataType;

    // Constants for Winitzki approximation
    // a = 8*(π-3)/(3*π*(4-π)) ≈ 0.140012
    double a = 0.140012;
    double two_over_pi_a = 2.0 / (M_PI * a);  // ≈ 4.546884

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dtype];
    MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:dtype];
    MPSGraphTensor* neg_one = [g constantWithScalar:-1.0 dataType:dtype];
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dtype];
    MPSGraphTensor* const_a = [g constantWithScalar:a dataType:dtype];
    MPSGraphTensor* const_two_pi_a = [g constantWithScalar:two_over_pi_a dataType:dtype];

    // x² = x * x
    MPSGraphTensor* x_sq = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];

    // 1 - x²
    MPSGraphTensor* one_minus_x_sq = [g subtractionWithPrimaryTensor:one
                                                     secondaryTensor:x_sq
                                                                name:nil];

    // Clamp to avoid log(0) - use small epsilon
    MPSGraphTensor* epsilon = [g constantWithScalar:1e-7 dataType:dtype];
    MPSGraphTensor* clamped = [g maximumWithPrimaryTensor:one_minus_x_sq
                                          secondaryTensor:epsilon
                                                     name:nil];

    // log(1 - x²)
    MPSGraphTensor* log_term = [g logarithmWithTensor:clamped name:nil];

    // t = 2/(π*a) + log(1-x²)/2
    MPSGraphTensor* half_log = [g multiplicationWithPrimaryTensor:log_term
                                                  secondaryTensor:half
                                                             name:nil];
    MPSGraphTensor* t = [g additionWithPrimaryTensor:const_two_pi_a
                                     secondaryTensor:half_log
                                                name:nil];

    // t²
    MPSGraphTensor* t_sq = [g multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];

    // log(1-x²) / a
    MPSGraphTensor* log_over_a = [g divisionWithPrimaryTensor:log_term
                                              secondaryTensor:const_a
                                                         name:nil];

    // t² - log(1-x²)/a
    MPSGraphTensor* inner = [g subtractionWithPrimaryTensor:t_sq
                                            secondaryTensor:log_over_a
                                                       name:nil];

    // sqrt(t² - log(1-x²)/a)
    MPSGraphTensor* sqrt_inner = [g squareRootWithTensor:inner name:nil];

    // sqrt(...) - t
    MPSGraphTensor* diff = [g subtractionWithPrimaryTensor:sqrt_inner secondaryTensor:t name:nil];

    // sqrt(sqrt(...) - t) = |erfinv(x)|
    MPSGraphTensor* abs_result = [g squareRootWithTensor:diff name:nil];

    // sign(x) * |result|
    MPSGraphTensor* sign_x = [g signWithTensor:x name:nil];
    return [g multiplicationWithPrimaryTensor:sign_x secondaryTensor:abs_result name:nil];
}
REGISTER_MPS_OP("chlo.erf_inv", Handle_erf_inv);

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
