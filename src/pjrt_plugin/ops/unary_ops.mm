// Unary operations: math functions, complex part extraction/construction

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Unary ops using macros
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, tanh);
REGISTER_MLIR_UNARY_OP("stablehlo.exponential", exponent, exp);
REGISTER_MLIR_UNARY_OP("stablehlo.log", logarithm, log);
REGISTER_MLIR_UNARY_OP("stablehlo.negate", negative, negate);
// abs: MPS absoluteWithTensor: on complex input returns complex (magnitude in
// real part, zero imaginary). StableHLO expects a real-valued result, so we
// extract the real part for complex inputs.
static ProcessResult Handle_abs(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("abs: missing input tensor");
    MPSGraphTensor* result = [g absoluteWithTensor:input name:nil];
    if (input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16)
        result = [g realPartOfTensor:result name:nil];
    return Result(values, op, result, "abs");
}
REGISTER_MPS_OP("stablehlo.abs", Handle_abs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.rsqrt", reciprocalSquareRoot, rsqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);
REGISTER_MLIR_UNARY_OP("chlo.erf", erf, chlo_erf);
REGISTER_MLIR_UNARY_OP("stablehlo.floor", floor, floor);
// sign: for complex inputs, stablehlo.sign returns x / |x| (or 0 for x == 0).
// MPS signWithTensor: applies component-wise sign which is wrong for complex.
static ProcessResult Handle_sign(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("sign: missing input tensor");

    MPSGraphTensor* result = nil;
    if (input.dataType != MPSDataTypeComplexFloat32 &&
        input.dataType != MPSDataTypeComplexFloat16) {
        result = [g signWithTensor:input name:nil];
    } else {
        MPSGraphTensor* re = [g realPartOfTensor:input name:nil];
        MPSGraphTensor* im = [g imaginaryPartOfTensor:input name:nil];
        MPSDataType floatType = re.dataType;

        // magnitude = |x| (as real)
        MPSGraphTensor* magnitude = [g realPartOfTensor:[g absoluteWithTensor:input name:nil]
                                                   name:nil];

        // Avoid division by zero: use 1 where magnitude is 0, then mask result to 0.
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:floatType];
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:floatType];
        MPSGraphTensor* is_zero = [g equalWithPrimaryTensor:magnitude
                                            secondaryTensor:zero
                                                       name:nil];
        MPSGraphTensor* safe_mag = [g selectWithPredicateTensor:is_zero
                                            truePredicateTensor:one
                                           falsePredicateTensor:magnitude
                                                           name:nil];

        // x / |x|, zeroed where |x| == 0
        MPSGraphTensor* norm_re = [g divisionWithPrimaryTensor:re
                                               secondaryTensor:safe_mag
                                                          name:nil];
        MPSGraphTensor* norm_im = [g divisionWithPrimaryTensor:im
                                               secondaryTensor:safe_mag
                                                          name:nil];
        norm_re = [g selectWithPredicateTensor:is_zero
                           truePredicateTensor:zero
                          falsePredicateTensor:norm_re
                                          name:nil];
        norm_im = [g selectWithPredicateTensor:is_zero
                           truePredicateTensor:zero
                          falsePredicateTensor:norm_im
                                          name:nil];

        result = [g complexTensorWithRealTensor:norm_re imaginaryTensor:norm_im name:nil];
    }

    return Result(values, op, result, "sign");
}
REGISTER_MPS_OP("stablehlo.sign", Handle_sign);
REGISTER_MLIR_UNARY_OP("stablehlo.is_finite", isFinite, is_finite);
REGISTER_MLIR_UNARY_OP("chlo.square", square, chlo_square);
REGISTER_MLIR_UNARY_OP("stablehlo.ceil", ceil, ceil);
REGISTER_MLIR_UNARY_OP("stablehlo.round_nearest_even", rint, round_nearest_even);
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);
REGISTER_MLIR_UNARY_OP("stablehlo.sine", sin, sine);
REGISTER_MLIR_UNARY_OP("stablehlo.tan", tan, tan);
REGISTER_MLIR_UNARY_OP("chlo.asin", asin, asin);
REGISTER_MLIR_UNARY_OP("chlo.acos", acos, acos);
REGISTER_MLIR_UNARY_OP("chlo.sinh", sinh, sinh);
REGISTER_MLIR_UNARY_OP("chlo.cosh", cosh, cosh);
REGISTER_MLIR_UNARY_OP("chlo.asinh", asinh, asinh);
REGISTER_MLIR_UNARY_OP("chlo.acosh", acosh, acosh);
REGISTER_MLIR_UNARY_OP("chlo.atanh", atanh, atanh);

// Custom call targets for mhlo.* variants of unary ops
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.erf", erf, mhlo_erf);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asin", asin, mhlo_asin);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acos", acos, mhlo_acos);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.sinh", sinh, mhlo_sinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.cosh", cosh, mhlo_cosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asinh", asinh, mhlo_asinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acosh", acosh, mhlo_acosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.atanh", atanh, mhlo_atanh);

// Complex part extraction (methods use OfTensor, not WithTensor, so can't use the macro)
static ProcessResult Handle_real(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("real: missing input tensor");
    MPSGraphTensor* result = [g realPartOfTensor:input name:nil];
    return Result(values, op, result, "real");
}
REGISTER_MPS_OP("stablehlo.real", Handle_real);

static ProcessResult Handle_imag(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("imag: missing input tensor");
    MPSGraphTensor* result = [g imaginaryPartOfTensor:input name:nil];
    return Result(values, op, result, "imag");
}
REGISTER_MPS_OP("stablehlo.imag", Handle_imag);

// Complex construction from real and imaginary parts
static ProcessResult Handle_complex(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* real = GetInputTensor(values, op, 0);
    MPSGraphTensor* imag = GetInputTensor(values, op, 1);
    if (!real || !imag)
        return ProcessResult::Error("complex: missing input tensor");
    MPSGraphTensor* result = [g complexTensorWithRealTensor:real imaginaryTensor:imag name:nil];
    return Result(values, op, result, "complex");
}
REGISTER_MPS_OP("stablehlo.complex", Handle_complex);

// exponential_minus_one: exp(x) - 1
static ProcessResult Handle_expm1(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("exponential_minus_one: missing input tensor");
    MPSGraphTensor* exp_x = [g exponentWithTensor:input name:nil];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* result = [g subtractionWithPrimaryTensor:exp_x secondaryTensor:one name:nil];
    return Result(values, op, result, "exponential_minus_one");
}
REGISTER_MPS_OP("stablehlo.exponential_minus_one", Handle_expm1);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
static ProcessResult Handle_log_plus_one(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("log_plus_one: missing input tensor");

    // FIXME: This naive implementation is numerically unstable for small inputs.
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [g additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    MPSGraphTensor* result = [g logarithmWithTensor:onePlusX name:nil];
    return Result(values, op, result, "log_plus_one");
}
REGISTER_MPS_OP("stablehlo.log_plus_one", Handle_log_plus_one);

// Inverse error function (erfinv) using Winitzki approximation
// erfinv(x) ≈ sign(x) * sqrt(sqrt(t² - log(1-x²)/a) - t)
// where t = 2/(π*a) + log(1-x²)/2, a ≈ 0.147
static ProcessResult Handle_erf_inv(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* x = GetInputTensor(values, op, 0);
    if (!x)
        return ProcessResult::Error("erf_inv: missing input tensor");

    MPSDataType dtype = x.dataType;

    // Constants for Winitzki approximation
    // a = 8*(π-3)/(3*π*(4-π)) ≈ 0.140012
    double a = 0.140012;
    double two_over_pi_a = 2.0 / (M_PI * a);  // ≈ 4.546884

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dtype];
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
    MPSGraphTensor* result = [g multiplicationWithPrimaryTensor:sign_x
                                                secondaryTensor:abs_result
                                                           name:nil];
    return Result(values, op, result, "erf_inv");
}
REGISTER_MPS_OP("chlo.erf_inv", Handle_erf_inv);

// cbrt: cube root via sign(x) * pow(|x|, 1/3)
static ProcessResult Handle_cbrt(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("cbrt: missing input tensor");
    MPSGraphTensor* abs_input = [g absoluteWithTensor:input name:nil];
    MPSGraphTensor* third = [g constantWithScalar:(1.0 / 3.0) dataType:input.dataType];
    MPSGraphTensor* pow_result = [g powerWithPrimaryTensor:abs_input
                                           secondaryTensor:third
                                                      name:nil];
    MPSGraphTensor* sign = [g signWithTensor:input name:nil];
    MPSGraphTensor* result = [g multiplicationWithPrimaryTensor:sign
                                                secondaryTensor:pow_result
                                                           name:nil];
    return Result(values, op, result, "cbrt");
}
REGISTER_MPS_OP("stablehlo.cbrt", Handle_cbrt);

}  // namespace jax_mps
