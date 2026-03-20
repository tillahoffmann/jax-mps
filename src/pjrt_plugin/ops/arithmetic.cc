// Arithmetic, bitwise, and comparison op handlers.

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.clamp
bool HandleClamp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto min_opt = GetValue(values, op->getOperand(0));
    auto operand_opt = GetValue(values, op->getOperand(1));
    auto max_opt = GetValue(values, op->getOperand(2));
    if (!min_opt || !operand_opt || !max_opt) {
        MPS_LOG_ERROR("stablehlo.clamp: operand not found in value map\n");
        return false;
    }
    // clamp(min, x, max) -> maximum(min, minimum(x, max))
    auto clamped = mlx::core::clip(operand_opt->get(), min_opt->get(), max_opt->get());
    values.emplace(ToKey(op->getResult(0)), clamped);
    return true;
}

// Handler for stablehlo.remainder
bool HandleRemainder(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.remainder: operand not found in value map\n");
        return false;
    }
    // stablehlo.remainder is C-style fmod: a - trunc(a/b) * b
    auto& a = lhs_opt->get();
    auto& b = rhs_opt->get();

    auto dtype = a.dtype();
    bool isUnsigned = (dtype == mlx::core::uint8 || dtype == mlx::core::uint16 ||
                       dtype == mlx::core::uint32 || dtype == mlx::core::uint64);
    bool isSigned = (dtype == mlx::core::int8 || dtype == mlx::core::int16 ||
                     dtype == mlx::core::int32 || dtype == mlx::core::int64);

    mlx::core::array result(0);
    if (isUnsigned) {
        // For unsigned integers, Python-style remainder == C-style remainder
        result = mlx::core::remainder(a, b);
    } else if (isSigned) {
        // For signed integers, Python remainder (rounds toward -inf) needs correction
        // to C-style remainder (truncates toward zero)
        auto py_rem = mlx::core::remainder(a, b);
        // Correction needed when a and b have different signs and py_rem != 0
        auto zero = mlx::core::zeros_like(a);
        auto diff_sign = mlx::core::not_equal(mlx::core::less(a, zero), mlx::core::less(b, zero));
        auto needs_fix = mlx::core::logical_and(mlx::core::not_equal(py_rem, zero), diff_sign);
        result = mlx::core::where(needs_fix, mlx::core::subtract(py_rem, b), py_rem);
    } else {
        // For float types, trunc(x) = sign(x) * floor(abs(x))
        auto quotient = mlx::core::divide(a, b);
        auto truncated = mlx::core::multiply(mlx::core::sign(quotient),
                                             mlx::core::floor(mlx::core::abs(quotient)));
        result = mlx::core::subtract(a, mlx::core::multiply(truncated, b));
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.cbrt (cube root = sign(x) * |x|^(1/3))
bool HandleCbrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cbrt: operand not found in value map\n");
        return false;
    }
    auto& x = input_opt->get();
    // cbrt(x) = sign(x) * |x|^(1/3)
    auto third = mlx::core::array(1.0F / 3.0F, x.dtype());
    auto result =
        mlx::core::multiply(mlx::core::sign(x), mlx::core::power(mlx::core::abs(x), third));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.not (bitwise not)
bool HandleNot(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.not: operand not found in value map\n");
        return false;
    }
    auto& x = input_opt->get();
    if (x.dtype() == mlx::core::bool_) {
        values.emplace(ToKey(op->getResult(0)), mlx::core::logical_not(x));
    } else {
        // Bitwise NOT for integers: ~x = x XOR all-ones
        auto all_ones = mlx::core::full(x.shape(), -1, x.dtype());
        values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_xor(x, all_ones));
    }
    return true;
}

// Handler for stablehlo.shift_right_arithmetic
bool HandleShiftRightArithmetic(mlir::Operation* op, ValueMap& values,
                                std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.shift_right_arithmetic: operand not found in value map\n");
        return false;
    }
    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();
    // StableHLO spec: shift < 0 or shift >= bit_width for arithmetic right shift:
    // positive values → 0, negative values → -1 (sign bit propagation)
    int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
    auto oob = mlx::core::logical_or(
        mlx::core::less(rhs, mlx::core::array(0, rhs.dtype())),
        mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype())));
    auto shifted =
        mlx::core::right_shift(lhs, mlx::core::maximum(rhs, mlx::core::array(0, rhs.dtype())));
    // For arithmetic shift, oob result depends on sign: 0 for positive, -1 for negative
    auto oob_val =
        mlx::core::where(mlx::core::less(lhs, mlx::core::array(0, lhs.dtype())),
                         mlx::core::full(lhs.shape(), -1, lhs.dtype()), mlx::core::zeros_like(lhs));
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(oob, oob_val, shifted));
    return true;
}

// Handler for stablehlo.popcnt (population count)
bool HandlePopcount(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.popcnt: operand not found in value map\n");
        return false;
    }
    auto& x = input_opt->get();
    // Implement popcount using bit manipulation for integer types
    auto dtype = x.dtype();

    // Determine bit width for masking after uint32 cast
    size_t bit_width = GetDtypeSize(dtype) * 8;

    // For signed types, first cast to unsigned of same width to avoid sign extension
    mlx::core::array val = x;
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
    auto cond_opt = GetValue(values, op->getOperand(0));
    auto true_opt = GetValue(values, op->getOperand(1));
    auto false_opt = GetValue(values, op->getOperand(2));
    if (!cond_opt || !true_opt || !false_opt) {
        MPS_LOG_ERROR("stablehlo.select: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)),
                   mlx::core::where(cond_opt->get(), true_opt->get(), false_opt->get()));
    return true;
}

// Handler for stablehlo.compare
bool HandleCompare(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        MPS_LOG_ERROR("stablehlo.compare: failed to cast\n");
        return false;
    }

    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.compare: operand not found in value map\n");
        return false;
    }

    auto direction = compareOp.getComparisonDirection();
    std::optional<mlx::core::array> result;

    using Dir = mlir::stablehlo::ComparisonDirection;
    switch (direction) {
        case Dir::EQ:
            result = mlx::core::equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::NE:
            result = mlx::core::not_equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::LT:
            result = mlx::core::less(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::LE:
            result = mlx::core::less_equal(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::GT:
            result = mlx::core::greater(lhs_opt->get(), rhs_opt->get());
            break;
        case Dir::GE:
            result = mlx::core::greater_equal(lhs_opt->get(), rhs_opt->get());
            break;
        default:
            MPS_LOG_ERROR("stablehlo.compare: unsupported comparison direction\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(*result));
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
}

}  // namespace jax_mps
