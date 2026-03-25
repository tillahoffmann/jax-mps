// Arithmetic, bitwise, and comparison op handlers.

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.clamp
bool HandleClamp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto* min_val = RequireValue(values, op->getOperand(0), "stablehlo.clamp");
    auto* operand = RequireValue(values, op->getOperand(1), "stablehlo.clamp");
    auto* max_val = RequireValue(values, op->getOperand(2), "stablehlo.clamp");
    if (!min_val || !operand || !max_val)
        return false;
    values.emplace(ToKey(op->getResult(0)), mlx::core::clip(*operand, *min_val, *max_val));
    return true;
}

// Handler for stablehlo.remainder
bool HandleRemainder(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto* a = RequireValue(values, op->getOperand(0), "stablehlo.remainder");
    auto* b = RequireValue(values, op->getOperand(1), "stablehlo.remainder");
    if (!a || !b)
        return false;

    // stablehlo.remainder is C-style fmod: a - trunc(a/b) * b
    auto dtype = a->dtype();
    bool isUnsigned = (dtype == mlx::core::uint8 || dtype == mlx::core::uint16 ||
                       dtype == mlx::core::uint32 || dtype == mlx::core::uint64);
    bool isSigned = (dtype == mlx::core::int8 || dtype == mlx::core::int16 ||
                     dtype == mlx::core::int32 || dtype == mlx::core::int64);

    mlx::core::array result(0);
    if (isUnsigned) {
        // For unsigned integers, Python-style remainder == C-style remainder
        result = mlx::core::remainder(*a, *b);
    } else if (isSigned) {
        // For signed integers, Python remainder (rounds toward -inf) needs correction
        // to C-style remainder (truncates toward zero)
        auto py_rem = mlx::core::remainder(*a, *b);
        auto zero = mlx::core::zeros_like(*a);
        auto diff_sign = mlx::core::not_equal(mlx::core::less(*a, zero), mlx::core::less(*b, zero));
        auto needs_fix = mlx::core::logical_and(mlx::core::not_equal(py_rem, zero), diff_sign);
        result = mlx::core::where(needs_fix, mlx::core::subtract(py_rem, *b), py_rem);
    } else {
        // For float types, trunc(x) = sign(x) * floor(abs(x))
        auto quotient = mlx::core::divide(*a, *b);
        auto truncated = mlx::core::multiply(mlx::core::sign(quotient),
                                             mlx::core::floor(mlx::core::abs(quotient)));
        result = mlx::core::subtract(*a, mlx::core::multiply(truncated, *b));
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.cbrt (cube root = sign(x) * |x|^(1/3))
bool HandleCbrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.cbrt");
    if (!x)
        return false;
    auto third = mlx::core::array(1.0F / 3.0F, x->dtype());
    auto result =
        mlx::core::multiply(mlx::core::sign(*x), mlx::core::power(mlx::core::abs(*x), third));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.not (bitwise not)
bool HandleNot(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.not");
    if (!x)
        return false;
    if (x->dtype() == mlx::core::bool_) {
        values.emplace(ToKey(op->getResult(0)), mlx::core::logical_not(*x));
    } else {
        // Bitwise NOT for integers: ~x = x XOR all-ones
        auto all_ones = mlx::core::full(x->shape(), -1, x->dtype());
        values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_xor(*x, all_ones));
    }
    return true;
}

// Handler for stablehlo.shift_right_arithmetic
bool HandleShiftRightArithmetic(mlir::Operation* op, ValueMap& values,
                                std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto* lhs = RequireValue(values, op->getOperand(0), "stablehlo.shift_right_arithmetic");
    auto* rhs = RequireValue(values, op->getOperand(1), "stablehlo.shift_right_arithmetic");
    if (!lhs || !rhs)
        return false;
    // StableHLO spec: shift < 0 or shift >= bit_width for arithmetic right shift:
    // positive values -> 0, negative values -> -1 (sign bit propagation)
    int bit_width = static_cast<int>(GetDtypeSize(lhs->dtype()) * 8);
    auto oob = mlx::core::logical_or(
        mlx::core::less(*rhs, mlx::core::array(0, rhs->dtype())),
        mlx::core::greater_equal(*rhs, mlx::core::array(bit_width, rhs->dtype())));
    auto shifted =
        mlx::core::right_shift(*lhs, mlx::core::maximum(*rhs, mlx::core::array(0, rhs->dtype())));
    // For arithmetic shift, oob result depends on sign: 0 for positive, -1 for negative
    auto oob_val = mlx::core::where(mlx::core::less(*lhs, mlx::core::array(0, lhs->dtype())),
                                    mlx::core::full(lhs->shape(), -1, lhs->dtype()),
                                    mlx::core::zeros_like(*lhs));
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(oob, oob_val, shifted));
    return true;
}

// Handler for stablehlo.popcnt (population count)
bool HandlePopcount(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.popcnt");
    if (!x)
        return false;
    auto dtype = x->dtype();
    size_t bit_width = GetDtypeSize(dtype) * 8;

    // For signed types, first cast to unsigned of same width to avoid sign extension
    mlx::core::array val = *x;
    if (dtype == mlx::core::int8) {
        val = mlx::core::astype(val, mlx::core::uint8);
    } else if (dtype == mlx::core::int16) {
        val = mlx::core::astype(val, mlx::core::uint16);
    } else if (dtype == mlx::core::int64) {
        val = mlx::core::astype(val, mlx::core::uint64);
    }

    if (bit_width > 32) {
        // 64-bit Hamming weight algorithm
        val = mlx::core::astype(val, mlx::core::uint64);
        auto m1 = mlx::core::array(static_cast<uint64_t>(0x5555555555555555ULL), mlx::core::uint64);
        auto m2 = mlx::core::array(static_cast<uint64_t>(0x3333333333333333ULL), mlx::core::uint64);
        auto m4 = mlx::core::array(static_cast<uint64_t>(0x0F0F0F0F0F0F0F0FULL), mlx::core::uint64);
        val = mlx::core::subtract(
            val, mlx::core::bitwise_and(
                     mlx::core::right_shift(val, mlx::core::array(1, mlx::core::uint64)), m1));
        val = mlx::core::add(
            mlx::core::bitwise_and(val, m2),
            mlx::core::bitwise_and(
                mlx::core::right_shift(val, mlx::core::array(2, mlx::core::uint64)), m2));
        val = mlx::core::bitwise_and(
            mlx::core::add(val,
                           mlx::core::right_shift(val, mlx::core::array(4, mlx::core::uint64))),
            m4);
        val = mlx::core::add(val,
                             mlx::core::right_shift(val, mlx::core::array(8, mlx::core::uint64)));
        val = mlx::core::add(val,
                             mlx::core::right_shift(val, mlx::core::array(16, mlx::core::uint64)));
        val = mlx::core::add(val,
                             mlx::core::right_shift(val, mlx::core::array(32, mlx::core::uint64)));
        val = mlx::core::bitwise_and(
            val, mlx::core::array(static_cast<uint64_t>(0x7FU), mlx::core::uint64));
    } else {
        // 32-bit Hamming weight algorithm
        val = mlx::core::astype(val, mlx::core::uint32);

        // Mask to original bit width to ensure upper bits are zero
        if (bit_width < 32) {
            uint32_t mask = (1U << bit_width) - 1;
            val = mlx::core::bitwise_and(val, mlx::core::array(mask, mlx::core::uint32));
        }

        auto m1 = mlx::core::array(0x55555555U, mlx::core::uint32);
        auto m2 = mlx::core::array(0x33333333U, mlx::core::uint32);
        auto m4 = mlx::core::array(0x0F0F0F0FU, mlx::core::uint32);
        val = mlx::core::subtract(
            val, mlx::core::bitwise_and(
                     mlx::core::right_shift(val, mlx::core::array(1, mlx::core::uint32)), m1));
        val = mlx::core::add(
            mlx::core::bitwise_and(val, m2),
            mlx::core::bitwise_and(
                mlx::core::right_shift(val, mlx::core::array(2, mlx::core::uint32)), m2));
        val = mlx::core::bitwise_and(
            mlx::core::add(val,
                           mlx::core::right_shift(val, mlx::core::array(4, mlx::core::uint32))),
            m4);
        val = mlx::core::add(val,
                             mlx::core::right_shift(val, mlx::core::array(8, mlx::core::uint32)));
        val = mlx::core::add(val,
                             mlx::core::right_shift(val, mlx::core::array(16, mlx::core::uint32)));
        val = mlx::core::bitwise_and(val, mlx::core::array(0x3FU, mlx::core::uint32));
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(val, dtype));
    return true;
}

// Handler for stablehlo.select
bool HandleSelect(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto* cond = RequireValue(values, op->getOperand(0), "stablehlo.select");
    auto* true_val = RequireValue(values, op->getOperand(1), "stablehlo.select");
    auto* false_val = RequireValue(values, op->getOperand(2), "stablehlo.select");
    if (!cond || !true_val || !false_val)
        return false;
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(*cond, *true_val, *false_val));
    return true;
}

// Handler for stablehlo.compare
bool HandleCompare(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto compareOp = CastOp<mlir::stablehlo::CompareOp>(op, "stablehlo.compare");
    if (!compareOp)
        return false;

    auto* lhs = RequireValue(values, op->getOperand(0), "stablehlo.compare");
    auto* rhs = RequireValue(values, op->getOperand(1), "stablehlo.compare");
    if (!lhs || !rhs)
        return false;

    auto direction = compareOp.getComparisonDirection();
    std::optional<mlx::core::array> result;

    using Dir = mlir::stablehlo::ComparisonDirection;
    switch (direction) {
        case Dir::EQ:
            result = mlx::core::equal(*lhs, *rhs);
            break;
        case Dir::NE:
            result = mlx::core::not_equal(*lhs, *rhs);
            break;
        case Dir::LT:
            result = mlx::core::less(*lhs, *rhs);
            break;
        case Dir::LE:
            result = mlx::core::less_equal(*lhs, *rhs);
            break;
        case Dir::GT:
            result = mlx::core::greater(*lhs, *rhs);
            break;
        case Dir::GE:
            result = mlx::core::greater_equal(*lhs, *rhs);
            break;
        default:
            MPS_LOG_ERROR("stablehlo.compare: unsupported comparison direction\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(*result));
    return true;
}

// Handler for stablehlo.round_nearest_afz (round half away from zero)
bool HandleRoundNearestAfz(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.round_nearest_afz");
    if (!x)
        return false;
    // round_nearest_afz: sign(x) * floor(abs(x) + 0.5)
    auto half = mlx::core::array(0.5F, x->dtype());
    auto result = mlx::core::multiply(mlx::core::sign(*x),
                                      mlx::core::floor(mlx::core::add(mlx::core::abs(*x), half)));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.count_leading_zeros
bool HandleCountLeadingZeros(mlir::Operation* op, ValueMap& values,
                             std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.count_leading_zeros");
    if (!x)
        return false;
    auto dtype = x->dtype();
    size_t bit_width = GetDtypeSize(dtype) * 8;

    // Work in unsigned to avoid sign extension issues
    mlx::core::array val = *x;
    auto workDtype = bit_width > 32 ? mlx::core::uint64 : mlx::core::uint32;
    val = mlx::core::astype(val, workDtype);

    // Mask to original bit width for sub-32-bit types
    if (bit_width < 32) {
        uint32_t mask = (1U << bit_width) - 1;
        val = mlx::core::bitwise_and(val, mlx::core::array(mask, workDtype));
    }

    // CLZ via bit-smearing: propagate the highest set bit rightward, then CLZ = bit_width -
    // popcount Smear the highest bit to the right: val |= val >> 1; val |= val >> 2; ...
    for (size_t shift = 1; shift < bit_width; shift <<= 1) {
        val = mlx::core::bitwise_or(
            val,
            mlx::core::right_shift(val, mlx::core::array(static_cast<uint32_t>(shift), workDtype)));
    }

    // CLZ = bit_width - popcount(val) after smearing
    // Inline popcount using Hamming weight
    if (bit_width > 32) {
        auto m1 = mlx::core::array(static_cast<uint64_t>(0x5555555555555555ULL), mlx::core::uint64);
        auto m2 = mlx::core::array(static_cast<uint64_t>(0x3333333333333333ULL), mlx::core::uint64);
        auto m4 = mlx::core::array(static_cast<uint64_t>(0x0F0F0F0F0F0F0F0FULL), mlx::core::uint64);
        auto one = mlx::core::array(static_cast<uint64_t>(1), mlx::core::uint64);
        auto two = mlx::core::array(static_cast<uint64_t>(2), mlx::core::uint64);
        auto four = mlx::core::array(static_cast<uint64_t>(4), mlx::core::uint64);
        auto eight = mlx::core::array(static_cast<uint64_t>(8), mlx::core::uint64);
        auto sixteen = mlx::core::array(static_cast<uint64_t>(16), mlx::core::uint64);
        auto thirtytwo = mlx::core::array(static_cast<uint64_t>(32), mlx::core::uint64);
        val =
            mlx::core::subtract(val, mlx::core::bitwise_and(mlx::core::right_shift(val, one), m1));
        val = mlx::core::add(mlx::core::bitwise_and(val, m2),
                             mlx::core::bitwise_and(mlx::core::right_shift(val, two), m2));
        val = mlx::core::bitwise_and(mlx::core::add(val, mlx::core::right_shift(val, four)), m4);
        val = mlx::core::add(val, mlx::core::right_shift(val, eight));
        val = mlx::core::add(val, mlx::core::right_shift(val, sixteen));
        val = mlx::core::add(val, mlx::core::right_shift(val, thirtytwo));
        val = mlx::core::bitwise_and(
            val, mlx::core::array(static_cast<uint64_t>(0x7FU), mlx::core::uint64));
    } else {
        auto m1 = mlx::core::array(0x55555555U, mlx::core::uint32);
        auto m2 = mlx::core::array(0x33333333U, mlx::core::uint32);
        auto m4 = mlx::core::array(0x0F0F0F0FU, mlx::core::uint32);
        auto one = mlx::core::array(1U, mlx::core::uint32);
        auto two = mlx::core::array(2U, mlx::core::uint32);
        auto four = mlx::core::array(4U, mlx::core::uint32);
        auto eight = mlx::core::array(8U, mlx::core::uint32);
        auto sixteen = mlx::core::array(16U, mlx::core::uint32);
        val =
            mlx::core::subtract(val, mlx::core::bitwise_and(mlx::core::right_shift(val, one), m1));
        val = mlx::core::add(mlx::core::bitwise_and(val, m2),
                             mlx::core::bitwise_and(mlx::core::right_shift(val, two), m2));
        val = mlx::core::bitwise_and(mlx::core::add(val, mlx::core::right_shift(val, four)), m4);
        val = mlx::core::add(val, mlx::core::right_shift(val, eight));
        val = mlx::core::add(val, mlx::core::right_shift(val, sixteen));
        val = mlx::core::bitwise_and(val, mlx::core::array(0x3FU, mlx::core::uint32));
    }

    // CLZ = bit_width - popcount
    auto bw = mlx::core::array(static_cast<uint32_t>(bit_width), workDtype);
    val = mlx::core::subtract(bw, val);
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(val, dtype));
    return true;
}

// Handler for stablehlo.reduce_precision
bool HandleReducePrecision(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto rpOp = CastOp<mlir::stablehlo::ReducePrecisionOp>(op, "stablehlo.reduce_precision");
    if (!rpOp)
        return false;
    auto* x = RequireValue(values, op->getOperand(0), "stablehlo.reduce_precision");
    if (!x)
        return false;

    uint32_t exponent_bits = rpOp.getExponentBits();
    uint32_t mantissa_bits = rpOp.getMantissaBits();

    // Work in float32 (IEEE 754: 8 exponent bits, 23 mantissa bits)
    auto val = mlx::core::astype(*x, mlx::core::float32);

    // Reinterpret as uint32 for bit manipulation
    auto bits = mlx::core::view(val, mlx::core::uint32);

    // Truncate mantissa: zero out the low (23 - mantissa_bits) bits
    if (mantissa_bits < 23) {
        uint32_t mantissa_mask = ~((1U << (23 - mantissa_bits)) - 1);
        // Round to nearest even: add rounding bias
        uint32_t round_bit = 1U << (23 - mantissa_bits - 1);
        auto round_val = mlx::core::array(round_bit, mlx::core::uint32);
        // Check if we're exactly at the midpoint (trailing bits are exactly the round bit)
        auto trailing_mask_val =
            mlx::core::array((1U << (23 - mantissa_bits)) - 1, mlx::core::uint32);
        auto trailing = mlx::core::bitwise_and(bits, trailing_mask_val);
        auto is_midpoint = mlx::core::equal(trailing, round_val);
        // For midpoint, round to even (clear LSB of kept mantissa)
        auto lsb_bit = mlx::core::array(1U << (23 - mantissa_bits), mlx::core::uint32);
        auto lsb_set = mlx::core::not_equal(mlx::core::bitwise_and(bits, lsb_bit),
                                            mlx::core::array(0U, mlx::core::uint32));
        // Add round_bit, but for exact midpoint only round if LSB is set (round to even)
        auto should_round = mlx::core::logical_or(mlx::core::greater(trailing, round_val),
                                                  mlx::core::logical_and(is_midpoint, lsb_set));
        bits = mlx::core::where(should_round, mlx::core::add(bits, round_val), bits);
        bits = mlx::core::bitwise_and(bits, mlx::core::array(mantissa_mask, mlx::core::uint32));
    }

    // Clamp exponent: if exponent_bits < 8, clamp to the representable range
    if (exponent_bits < 8) {
        // Max biased exponent in reduced format (exclude all-ones which is reserved for inf/NaN)
        uint32_t max_biased = (1U << exponent_bits) - 2;
        // Bias difference: float32 bias is 127, reduced bias is (1 << (exponent_bits-1)) - 1
        uint32_t reduced_bias = (1U << (exponent_bits - 1)) - 1;
        // Max/min unbiased exponent in reduced format
        int max_exp = static_cast<int>(max_biased - reduced_bias);
        int min_exp = static_cast<int>(1 - reduced_bias);
        // Corresponding float32 biased exponents
        uint32_t max_f32_biased = static_cast<uint32_t>(max_exp + 127);
        uint32_t min_f32_biased = static_cast<uint32_t>(min_exp + 127);

        // Extract sign and exponent
        auto sign_bit =
            mlx::core::bitwise_and(bits, mlx::core::array(0x80000000U, mlx::core::uint32));
        auto exp_field = mlx::core::bitwise_and(
            mlx::core::right_shift(bits, mlx::core::array(23U, mlx::core::uint32)),
            mlx::core::array(0xFFU, mlx::core::uint32));

        // Preserve NaN/inf (float32 exponent == 255): skip clamping for these values
        auto is_nan_or_inf =
            mlx::core::equal(exp_field, mlx::core::array(0xFFU, mlx::core::uint32));

        // If exponent > max (and not NaN/inf), set to infinity (preserve sign)
        auto max_exp_arr = mlx::core::array(max_f32_biased, mlx::core::uint32);
        auto overflow = mlx::core::logical_and(mlx::core::greater(exp_field, max_exp_arr),
                                               mlx::core::logical_not(is_nan_or_inf));
        auto inf_bits =
            mlx::core::bitwise_or(sign_bit, mlx::core::array(0x7F800000U, mlx::core::uint32));
        bits = mlx::core::where(overflow, inf_bits, bits);

        // If exponent < min (and not zero/subnormal), set to zero (preserve sign)
        auto min_exp_arr = mlx::core::array(min_f32_biased, mlx::core::uint32);
        auto is_zero = mlx::core::equal(exp_field, mlx::core::array(0U, mlx::core::uint32));
        auto underflow = mlx::core::logical_and(mlx::core::less(exp_field, min_exp_arr),
                                                mlx::core::logical_not(is_zero));
        bits = mlx::core::where(underflow, sign_bit, bits);
    }

    // Reinterpret back to float32
    val = mlx::core::view(bits, mlx::core::float32);
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(val, x->dtype()));
    return true;
}

}  // namespace

void RegisterArithmeticHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    // Trivial unary ops (one input -> one output, direct MLX call)
    handlers.insert(
        {"stablehlo.exponential", MakeUnaryHandler("stablehlo.exponential", mlx::core::exp)});
    handlers.insert({"stablehlo.log", MakeUnaryHandler("stablehlo.log", mlx::core::log)});
    handlers.insert({"stablehlo.sqrt", MakeUnaryHandler("stablehlo.sqrt", mlx::core::sqrt)});
    handlers.insert({"stablehlo.rsqrt", MakeUnaryHandler("stablehlo.rsqrt", mlx::core::rsqrt)});
    handlers.insert(
        {"stablehlo.negate", MakeUnaryHandler("stablehlo.negate", mlx::core::negative)});
    handlers.insert({"stablehlo.abs", MakeUnaryHandler("stablehlo.abs", mlx::core::abs)});
    handlers.insert({"stablehlo.floor", MakeUnaryHandler("stablehlo.floor", mlx::core::floor)});
    handlers.insert({"stablehlo.ceil", MakeUnaryHandler("stablehlo.ceil", mlx::core::ceil)});
    handlers.insert({"stablehlo.sine", MakeUnaryHandler("stablehlo.sine", mlx::core::sin)});
    handlers.insert({"stablehlo.cosine", MakeUnaryHandler("stablehlo.cosine", mlx::core::cos)});
    handlers.insert({"stablehlo.tanh", MakeUnaryHandler("stablehlo.tanh", mlx::core::tanh)});
    handlers.insert({"stablehlo.tan", MakeUnaryHandler("stablehlo.tan", mlx::core::tan)});
    handlers.insert({"stablehlo.sign", MakeUnaryHandler("stablehlo.sign", mlx::core::sign)});
    handlers.insert(
        {"stablehlo.log_plus_one", MakeUnaryHandler("stablehlo.log_plus_one", mlx::core::log1p)});
    handlers.insert({"stablehlo.round_nearest_even",
                     MakeUnaryHandler("stablehlo.round_nearest_even", mlx::core::round)});
    handlers.insert(
        {"stablehlo.is_finite", MakeUnaryHandler("stablehlo.is_finite", mlx::core::isfinite)});
    handlers.insert({"stablehlo.exponential_minus_one",
                     MakeUnaryHandler("stablehlo.exponential_minus_one", mlx::core::expm1)});
    handlers.insert({"stablehlo.real", MakeUnaryHandler("stablehlo.real", mlx::core::real)});
    handlers.insert({"stablehlo.imag", MakeUnaryHandler("stablehlo.imag", mlx::core::imag)});

    // Trivial binary ops (two inputs -> one output, direct MLX call)
    handlers.insert({"stablehlo.add", MakeBinaryHandler("stablehlo.add", mlx::core::add)});
    handlers.insert(
        {"stablehlo.subtract", MakeBinaryHandler("stablehlo.subtract", mlx::core::subtract)});
    handlers.insert(
        {"stablehlo.multiply", MakeBinaryHandler("stablehlo.multiply", mlx::core::multiply)});
    handlers.insert({"stablehlo.divide", MakeBinaryHandler("stablehlo.divide", mlx::core::divide)});
    handlers.insert(
        {"stablehlo.maximum", MakeBinaryHandler("stablehlo.maximum", mlx::core::maximum)});
    handlers.insert(
        {"stablehlo.minimum", MakeBinaryHandler("stablehlo.minimum", mlx::core::minimum)});
    handlers.insert({"stablehlo.power", MakeBinaryHandler("stablehlo.power", mlx::core::power)});
    handlers.insert({"stablehlo.atan2", MakeBinaryHandler("stablehlo.atan2", mlx::core::arctan2)});
    handlers.insert({"stablehlo.and", MakeBinaryHandler("stablehlo.and", mlx::core::bitwise_and)});
    handlers.insert({"stablehlo.or", MakeBinaryHandler("stablehlo.or", mlx::core::bitwise_or)});
    handlers.insert({"stablehlo.xor", MakeBinaryHandler("stablehlo.xor", mlx::core::bitwise_xor)});

    // Logical shift ops (OOB shift -> 0)
    handlers.insert({"stablehlo.shift_left",
                     MakeLogicalShiftHandler("stablehlo.shift_left", mlx::core::left_shift)});
    handlers.insert(
        {"stablehlo.shift_right_logical",
         MakeLogicalShiftHandler("stablehlo.shift_right_logical", mlx::core::right_shift)});

    // Non-trivial arithmetic
    handlers.insert({"stablehlo.clamp", HandleClamp});
    handlers.insert({"stablehlo.remainder", HandleRemainder});
    handlers.insert({"stablehlo.cbrt", HandleCbrt});

    // Bitwise (non-trivial)
    handlers.insert({"stablehlo.not", HandleNot});
    handlers.insert({"stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic});
    handlers.insert({"stablehlo.popcnt", HandlePopcount});

    // Comparison/selection
    handlers.insert({"stablehlo.compare", HandleCompare});
    handlers.insert({"stablehlo.select", HandleSelect});

    // Round, CLZ, reduce_precision
    handlers.insert({"stablehlo.round_nearest_afz", HandleRoundNearestAfz});
    handlers.insert({"stablehlo.count_leading_zeros", HandleCountLeadingZeros});
    handlers.insert({"stablehlo.reduce_precision", HandleReducePrecision});
}

}  // namespace jax_mps
