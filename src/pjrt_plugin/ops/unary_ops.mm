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
static ProcessResult HandleAbs(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("abs: missing input tensor");
    MPSGraphTensor* result = [ctx.graph absoluteWithTensor:input name:nil];
    if (input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16)
        result = [ctx.graph realPartOfTensor:result name:nil];
    return Result(ctx, result, "abs");
}
REGISTER_MPS_OP("stablehlo.abs", HandleAbs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.rsqrt", reciprocalSquareRoot, rsqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);
REGISTER_MLIR_UNARY_OP("chlo.erf", erf, chlo_erf);
REGISTER_MLIR_UNARY_OP("stablehlo.floor", floor, floor);
// sign: for complex inputs, stablehlo.sign returns x / |x| (or 0 for x == 0).
// MPS signWithTensor: applies component-wise sign which is wrong for complex.
static ProcessResult HandleSign(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("sign: missing input tensor");

    MPSGraphTensor* result = nil;
    if (input.dataType != MPSDataTypeComplexFloat32 &&
        input.dataType != MPSDataTypeComplexFloat16) {
        result = [ctx.graph signWithTensor:input name:nil];
    } else {
        MPSGraphTensor* re = [ctx.graph realPartOfTensor:input name:nil];
        MPSGraphTensor* im = [ctx.graph imaginaryPartOfTensor:input name:nil];
        MPSDataType floatType = re.dataType;

        // magnitude = |x| (as real)
        MPSGraphTensor* magnitude =
            [ctx.graph realPartOfTensor:[ctx.graph absoluteWithTensor:input name:nil] name:nil];

        // Avoid division by zero: use 1 where magnitude is 0, then mask result to 0.
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:floatType];
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:floatType];
        MPSGraphTensor* is_zero = [ctx.graph equalWithPrimaryTensor:magnitude
                                                    secondaryTensor:zero
                                                               name:nil];
        MPSGraphTensor* safe_mag = [ctx.graph selectWithPredicateTensor:is_zero
                                                    truePredicateTensor:one
                                                   falsePredicateTensor:magnitude
                                                                   name:nil];

        // x / |x|, zeroed where |x| == 0
        MPSGraphTensor* norm_re = [ctx.graph divisionWithPrimaryTensor:re
                                                       secondaryTensor:safe_mag
                                                                  name:nil];
        MPSGraphTensor* norm_im = [ctx.graph divisionWithPrimaryTensor:im
                                                       secondaryTensor:safe_mag
                                                                  name:nil];
        norm_re = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_re
                                                  name:nil];
        norm_im = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_im
                                                  name:nil];

        result = [ctx.graph complexTensorWithRealTensor:norm_re imaginaryTensor:norm_im name:nil];
    }

    return Result(ctx, result, "sign");
}
REGISTER_MPS_OP("stablehlo.sign", HandleSign);
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
static ProcessResult HandleReal(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("real: missing input tensor");
    MPSGraphTensor* result = [ctx.graph realPartOfTensor:input name:nil];
    return Result(ctx, result, "real");
}
REGISTER_MPS_OP("stablehlo.real", HandleReal);

static ProcessResult HandleImag(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("imag: missing input tensor");
    MPSGraphTensor* result = [ctx.graph imaginaryPartOfTensor:input name:nil];
    return Result(ctx, result, "imag");
}
REGISTER_MPS_OP("stablehlo.imag", HandleImag);

// Complex construction from real and imaginary parts
static ProcessResult HandleComplex(HandlerContext& ctx) {
    MPSGraphTensor* real = GetInputTensor(ctx, 0);
    MPSGraphTensor* imag = GetInputTensor(ctx, 1);
    if (!real || !imag)
        return ProcessResult::Error("complex: missing input tensor");
    MPSGraphTensor* result = [ctx.graph complexTensorWithRealTensor:real
                                                    imaginaryTensor:imag
                                                               name:nil];
    return Result(ctx, result, "complex");
}
REGISTER_MPS_OP("stablehlo.complex", HandleComplex);

// exponential_minus_one: exp(x) - 1
static ProcessResult HandleExpm1(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("exponential_minus_one: missing input tensor");
    MPSGraphTensor* exp_x = [ctx.graph exponentWithTensor:input name:nil];
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* result = [ctx.graph subtractionWithPrimaryTensor:exp_x
                                                     secondaryTensor:one
                                                                name:nil];
    return Result(ctx, result, "exponential_minus_one");
}
REGISTER_MPS_OP("stablehlo.exponential_minus_one", HandleExpm1);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
static ProcessResult HandleLogPlusOne(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("log_plus_one: missing input tensor");

    // FIXME: This naive implementation is numerically unstable for small inputs.
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [ctx.graph additionWithPrimaryTensor:input
                                                    secondaryTensor:one
                                                               name:nil];
    MPSGraphTensor* result = [ctx.graph logarithmWithTensor:onePlusX name:nil];
    return Result(ctx, result, "log_plus_one");
}
REGISTER_MPS_OP("stablehlo.log_plus_one", HandleLogPlusOne);

// Inverse error function (erfinv) using Winitzki approximation
// erfinv(x) ≈ sign(x) * sqrt(sqrt(t² - log(1-x²)/a) - t)
// where t = 2/(π*a) + log(1-x²)/2, a ≈ 0.147
static ProcessResult HandleErfInv(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    if (!x)
        return ProcessResult::Error("erf_inv: missing input tensor");

    MPSDataType dtype = x.dataType;

    // Constants for Winitzki approximation
    // a = 8*(π-3)/(3*π*(4-π)) ≈ 0.140012
    double a = 0.140012;
    double two_over_pi_a = 2.0 / (M_PI * a);  // ≈ 4.546884

    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:dtype];
    MPSGraphTensor* half = [ctx.graph constantWithScalar:0.5 dataType:dtype];
    MPSGraphTensor* const_a = [ctx.graph constantWithScalar:a dataType:dtype];
    MPSGraphTensor* const_two_pi_a = [ctx.graph constantWithScalar:two_over_pi_a dataType:dtype];

    // x² = x * x
    MPSGraphTensor* x_sq = [ctx.graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];

    // 1 - x²
    MPSGraphTensor* one_minus_x_sq = [ctx.graph subtractionWithPrimaryTensor:one
                                                             secondaryTensor:x_sq
                                                                        name:nil];

    // Clamp to avoid log(0) - use small epsilon
    MPSGraphTensor* epsilon = [ctx.graph constantWithScalar:1e-7 dataType:dtype];
    MPSGraphTensor* clamped = [ctx.graph maximumWithPrimaryTensor:one_minus_x_sq
                                                  secondaryTensor:epsilon
                                                             name:nil];

    // log(1 - x²)
    MPSGraphTensor* log_term = [ctx.graph logarithmWithTensor:clamped name:nil];

    // t = 2/(π*a) + log(1-x²)/2
    MPSGraphTensor* half_log = [ctx.graph multiplicationWithPrimaryTensor:log_term
                                                          secondaryTensor:half
                                                                     name:nil];
    MPSGraphTensor* t = [ctx.graph additionWithPrimaryTensor:const_two_pi_a
                                             secondaryTensor:half_log
                                                        name:nil];

    // t²
    MPSGraphTensor* t_sq = [ctx.graph multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];

    // log(1-x²) / a
    MPSGraphTensor* log_over_a = [ctx.graph divisionWithPrimaryTensor:log_term
                                                      secondaryTensor:const_a
                                                                 name:nil];

    // t² - log(1-x²)/a
    MPSGraphTensor* inner = [ctx.graph subtractionWithPrimaryTensor:t_sq
                                                    secondaryTensor:log_over_a
                                                               name:nil];

    // sqrt(t² - log(1-x²)/a)
    MPSGraphTensor* sqrt_inner = [ctx.graph squareRootWithTensor:inner name:nil];

    // sqrt(...) - t
    MPSGraphTensor* diff = [ctx.graph subtractionWithPrimaryTensor:sqrt_inner
                                                   secondaryTensor:t
                                                              name:nil];

    // sqrt(sqrt(...) - t) = |erfinv(x)|
    MPSGraphTensor* abs_result = [ctx.graph squareRootWithTensor:diff name:nil];

    // sign(x) * |result|
    MPSGraphTensor* sign_x = [ctx.graph signWithTensor:x name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign_x
                                                        secondaryTensor:abs_result
                                                                   name:nil];
    return Result(ctx, result, "erf_inv");
}
REGISTER_MPS_OP("chlo.erf_inv", HandleErfInv);

// cbrt: cube root via sign(x) * pow(|x|, 1/3)
static ProcessResult HandleCbrt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("cbrt: missing input tensor");
    MPSGraphTensor* abs_input = [ctx.graph absoluteWithTensor:input name:nil];
    MPSGraphTensor* third = [ctx.graph constantWithScalar:(1.0 / 3.0) dataType:input.dataType];
    MPSGraphTensor* pow_result = [ctx.graph powerWithPrimaryTensor:abs_input
                                                   secondaryTensor:third
                                                              name:nil];
    MPSGraphTensor* sign = [ctx.graph signWithTensor:input name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign
                                                        secondaryTensor:pow_result
                                                                   name:nil];
    return Result(ctx, result, "cbrt");
}
REGISTER_MPS_OP("stablehlo.cbrt", HandleCbrt);

}  // namespace jax_mps
