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
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
static MPSGraphTensor* Handle_log_plus_one(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                           NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [g additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    return [g logarithmWithTensor:onePlusX name:nil];
}
REGISTER_MPS_OP("stablehlo.log_plus_one", Handle_log_plus_one);

// Compare operation
static MPSGraphTensor* Handle_compare(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                      NSArray<NSNumber*>*) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        NSLog(@"ERROR: Expected CompareOp");
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
            NSLog(@"ERROR: Unknown compare direction");
            return nullptr;
    }
}
REGISTER_MPS_OP("stablehlo.compare", Handle_compare);

// Select operation (conditional selection: pred ? true_val : false_val)
static MPSGraphTensor* Handle_select(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                     NSArray<NSNumber*>*) {
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

// Clamp operation: clamp(min, x, max) = min(max(min, x), max)
static MPSGraphTensor* Handle_clamp(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                    NSArray<NSNumber*>*) {
    MPSGraphTensor* minVal = GetInputTensor(values, op, 0);
    MPSGraphTensor* operand = GetInputTensor(values, op, 1);
    MPSGraphTensor* maxVal = GetInputTensor(values, op, 2);
    if (!minVal || !operand || !maxVal)
        return nullptr;

    // clamp = min(max(minVal, operand), maxVal)
    MPSGraphTensor* clamped = [g maximumWithPrimaryTensor:minVal secondaryTensor:operand name:nil];
    return [g minimumWithPrimaryTensor:clamped secondaryTensor:maxVal name:nil];
}
REGISTER_MPS_OP("stablehlo.clamp", Handle_clamp);

// Constant creation - creates a constant tensor from MLIR constant op
static MPSGraphTensor* Handle_constant(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                       NSArray<NSNumber*>*) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
        NSLog(@"ERROR: Expected ConstantOp");
        return nullptr;
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype for constant operation");
        return nullptr;
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    auto value = constantOp.getValue();

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
                scalarValue = static_cast<double>(apInt.getZExtValue());
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

    NSLog(@"ERROR: Constant operation has unsupported value type");
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.constant", Handle_constant);

// Inverse error function (erfinv) using Winitzki approximation
// erfinv(x) ≈ sign(x) * sqrt(sqrt(t² - log(1-x²)/a) - t)
// where t = 2/(π*a) + log(1-x²)/2, a ≈ 0.147
static MPSGraphTensor* Handle_erf_inv(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                      NSArray<NSNumber*>*) {
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

}  // namespace jax_mps
