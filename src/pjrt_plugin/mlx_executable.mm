// MLX executable implementation with op dispatch

#include "pjrt_plugin/mlx_executable.h"

#include <mlx/compile.h>
#include <mlx/einsum.h>
#include <mlx/fft.h>
#include <mlx/memory.h>
#include <mlx/mlx.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/type_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Profiling infrastructure
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// Check if profiling is enabled (cached on first call)
bool IsProfilingEnabled() {
    static bool enabled = std::getenv("MPS_PROFILE") != nullptr;
    return enabled;
}

// Per-op timing accumulator
struct OpTimingStats {
    double total_ms = 0.0;
    size_t count = 0;
};

// Global profiling state (only used when MPS_PROFILE=1)
struct ProfilingState {
    std::unordered_map<std::string, OpTimingStats> op_times;
    double dispatch_overhead_ms = 0.0;
    double eval_time_ms = 0.0;
    double total_execution_ms = 0.0;
    size_t execution_count = 0;

    // Cumulative stats across all executions
    double cumulative_dispatch_ms = 0.0;
    double cumulative_eval_ms = 0.0;
    double cumulative_total_ms = 0.0;

    // Track time between Execute() calls
    Clock::time_point last_execute_end;
    double cumulative_between_calls_ms = 0.0;
    bool has_last_time = false;

    void Reset() {
        op_times.clear();
        dispatch_overhead_ms = 0.0;
        eval_time_ms = 0.0;
        total_execution_ms = 0.0;
    }

    void RecordOp(const std::string& name, double ms) {
        auto& stats = op_times[name];
        stats.total_ms += ms;
        stats.count++;
    }

    void PrintSummary() {
        execution_count++;

        // Accumulate stats
        cumulative_dispatch_ms += dispatch_overhead_ms;
        cumulative_eval_ms += eval_time_ms;
        cumulative_total_ms += total_execution_ms;

        // Only print final summary (every 1000 executions)
        if (execution_count % 1000 != 0) {
            return;
        }

        fprintf(stderr, "\n=== MPS Final Summary (%zu executions) ===\n", execution_count);
        fprintf(stderr, "Total GPU time: %.0f ms (dispatch: %.0f ms, eval: %.0f ms)\n",
                cumulative_total_ms, cumulative_dispatch_ms, cumulative_eval_ms);

        // Sort ops by total time
        std::vector<std::pair<std::string, OpTimingStats>> sorted_ops(op_times.begin(),
                                                                      op_times.end());
        std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& a, const auto& b) {
            return a.second.total_ms > b.second.total_ms;
        });

        fprintf(stderr, "\nTop ops by dispatch time:\n");
        int shown = 0;
        for (const auto& [name, stats] : sorted_ops) {
            if (shown++ >= 5)
                break;
            fprintf(stderr, "  %-25s: %.1f ms (%zu calls)\n", name.c_str(), stats.total_ms,
                    stats.count);
        }

        // Memory stats
        size_t peak_mem = mlx::core::get_peak_memory();
        fprintf(stderr, "Peak memory: %.0f MB\n", static_cast<double>(peak_mem) / 1e6);
        fprintf(stderr, "=========================================\n");

        // Reset op times for next batch
        Reset();
    }
};

ProfilingState& GetProfilingState() {
    static ProfilingState state;
    return state;
}

// Convert MLIR type to MLX dtype
mlx::core::Dtype MlirTypeToMlxDtype(mlir::Type type) {
    // Use the same logic as MlirTypeToPjrtDtype but return MLX types
    int pjrt_dtype = MlirTypeToPjrtDtype(type);

    // Check for unknown type
    if (pjrt_dtype == -1) {
        MPS_LOG_ERROR("Unknown MLIR type, defaulting to float32\n");
        return mlx::core::float32;
    }

    // Handle float64 downcast (MLX doesn't support float64)
    if (type.isF64()) {
        MPS_LOG_WARN("MLX doesn't support float64, downcasting to float32\n");
        return mlx::core::float32;
    }

    return PjrtDtypeToMlx(pjrt_dtype);
}

// Extract shape from RankedTensorType
mlx::core::Shape GetShape(mlir::RankedTensorType type) {
    mlx::core::Shape shape;
    for (int64_t dim : type.getShape()) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}

// Alias for the shared GetMlxDtypeSize function
inline size_t GetDtypeSize(mlx::core::Dtype dtype) {
    return GetMlxDtypeSize(dtype);
}

// Create MLX array from DenseElementsAttr (for constants)
// Returns empty optional on error
// Note: MLX's array constructor uses std::copy with typed pointers, so we must
// cast to the correct element type before passing to the constructor.
std::optional<mlx::core::array> CreateArrayFromDenseAttr(mlir::DenseElementsAttr attr) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
    if (!tensorType) {
        MPS_LOG_ERROR("Constant attribute is not a ranked tensor type\n");
        return std::nullopt;
    }

    auto shape = GetShape(tensorType);
    auto elemType = tensorType.getElementType();
    auto mlxDtype = MlirTypeToMlxDtype(elemType);
    auto rawData = attr.getRawData();

    // Helper to create array with correct pointer type
    auto createArray = [&](auto* typed_ptr) -> mlx::core::array {
        return mlx::core::array(typed_ptr, shape, mlxDtype);
    };

    // Handle splat constants (single value broadcast to shape)
    if (attr.isSplat()) {
        mlx::core::Shape scalarShape = {};
        std::optional<mlx::core::array> scalar_opt;

        // Create scalar with correctly-typed pointer
        switch (mlxDtype) {
            case mlx::core::int32:
                scalar_opt = mlx::core::array(reinterpret_cast<const int32_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int64:
                scalar_opt = mlx::core::array(reinterpret_cast<const int64_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint32:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint32_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint64:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint64_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::float32:
                scalar_opt = mlx::core::array(reinterpret_cast<const float*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::float16:
                scalar_opt =
                    mlx::core::array(reinterpret_cast<const mlx::core::float16_t*>(rawData.data()),
                                     scalarShape, mlxDtype);
                break;
            case mlx::core::bfloat16:
                scalar_opt =
                    mlx::core::array(reinterpret_cast<const mlx::core::bfloat16_t*>(rawData.data()),
                                     scalarShape, mlxDtype);
                break;
            case mlx::core::bool_:
                scalar_opt = mlx::core::array(reinterpret_cast<const bool*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int8:
                scalar_opt = mlx::core::array(reinterpret_cast<const int8_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint8:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint8_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::int16:
                scalar_opt = mlx::core::array(reinterpret_cast<const int16_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::uint16:
                scalar_opt = mlx::core::array(reinterpret_cast<const uint16_t*>(rawData.data()),
                                              scalarShape, mlxDtype);
                break;
            case mlx::core::complex64:
                scalar_opt = mlx::core::array(
                    reinterpret_cast<const mlx::core::complex64_t*>(rawData.data()), scalarShape,
                    mlxDtype);
                break;
            default:
                MPS_LOG_ERROR("Unsupported dtype %d for splat constant\n",
                              static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
                return std::nullopt;
        }

        if (!scalar_opt) {
            return std::nullopt;
        }

        if (shape.empty()) {
            return scalar_opt;
        }
        return mlx::core::broadcast_to(*scalar_opt, shape);
    }

    // Validate data size matches expected size
    size_t elemSize = GetDtypeSize(mlxDtype);
    size_t numElements = 1;
    for (int dim : shape) {
        numElements *= dim;
    }
    size_t expectedSize = numElements * elemSize;

    if (rawData.size() < expectedSize) {
        MPS_LOG_ERROR("Constant data size mismatch: got %zu bytes, expected %zu\n", rawData.size(),
                      expectedSize);
        return std::nullopt;
    }

    // Create array with correctly-typed pointer for proper std::copy behavior
    switch (mlxDtype) {
        case mlx::core::int32:
            return createArray(reinterpret_cast<const int32_t*>(rawData.data()));
        case mlx::core::int64:
            return createArray(reinterpret_cast<const int64_t*>(rawData.data()));
        case mlx::core::uint32:
            return createArray(reinterpret_cast<const uint32_t*>(rawData.data()));
        case mlx::core::uint64:
            return createArray(reinterpret_cast<const uint64_t*>(rawData.data()));
        case mlx::core::float32:
            return createArray(reinterpret_cast<const float*>(rawData.data()));
        case mlx::core::float16:
            return createArray(reinterpret_cast<const mlx::core::float16_t*>(rawData.data()));
        case mlx::core::bfloat16:
            return createArray(reinterpret_cast<const mlx::core::bfloat16_t*>(rawData.data()));
        case mlx::core::bool_:
            return createArray(reinterpret_cast<const bool*>(rawData.data()));
        case mlx::core::int8:
            return createArray(reinterpret_cast<const int8_t*>(rawData.data()));
        case mlx::core::uint8:
            return createArray(reinterpret_cast<const uint8_t*>(rawData.data()));
        case mlx::core::int16:
            return createArray(reinterpret_cast<const int16_t*>(rawData.data()));
        case mlx::core::uint16:
            return createArray(reinterpret_cast<const uint16_t*>(rawData.data()));
        case mlx::core::complex64:
            return createArray(reinterpret_cast<const mlx::core::complex64_t*>(rawData.data()));
        default:
            MPS_LOG_ERROR("Unsupported dtype %d for constant\n",
                          static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
            return std::nullopt;
    }
}

// Value map type using void* as key (from mlir::Value's opaque pointer)
using ValueMap = std::unordered_map<void*, mlx::core::array>;

void* ToKey(mlir::Value v) {
    return v.getAsOpaquePointer();
}

// Helper to safely get value from map
std::optional<std::reference_wrapper<mlx::core::array>> GetValue(ValueMap& values, mlir::Value v) {
    auto it = values.find(ToKey(v));
    if (it == values.end()) {
        return std::nullopt;
    }
    return std::ref(it->second);
}

// Execution context passed to handlers
struct ExecContext {
    mlir::ModuleOp module;
    bool inside_compile = false;  // true when running inside mlx::core::compile()
};

// Exception thrown when an op is incompatible with mlx::core::compile() tracing.
// Propagates through DispatchOp without logging, allowing the compile() attempt
// to fail cleanly and fall back to direct execution.
class CompileIncompatibleError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Op handler function type
using OpHandler =
    std::function<bool(mlir::Operation*, ValueMap&, std::vector<mlx::core::array>&, ExecContext&)>;

// Forward declaration for recursive call handling
bool ExecuteFunction(mlir::func::FuncOp func, const std::vector<mlx::core::array>& inputs,
                     std::vector<mlx::core::array>& outputs, ExecContext& ctx);

// Handler for stablehlo.add
bool HandleAdd(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.add: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::add(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.exponential
bool HandleExp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.exponential: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::exp(input_opt->get()));
    return true;
}

// Handler for stablehlo.log
bool HandleLog(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.log: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::log(input_opt->get()));
    return true;
}

// Handler for stablehlo.rsqrt
bool HandleRsqrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.rsqrt: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::rsqrt(input_opt->get()));
    return true;
}

// Handler for stablehlo.floor
bool HandleFloor(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.floor: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::floor(input_opt->get()));
    return true;
}

// Handler for stablehlo.sine
bool HandleSine(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.sine: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::sin(input_opt->get()));
    return true;
}

// Handler for stablehlo.cosine
bool HandleCosine(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cosine: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::cos(input_opt->get()));
    return true;
}

// Handler for stablehlo.minimum
bool HandleMinimum(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.minimum: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::minimum(lhs_opt->get(), rhs_opt->get()));
    return true;
}

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

// Handler for stablehlo.tanh
bool HandleTanh(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.tanh: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::tanh(input_opt->get()));
    return true;
}

// Handler for stablehlo.tan
bool HandleTan(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.tan: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::tan(input_opt->get()));
    return true;
}

// Handler for stablehlo.sign
bool HandleSign(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.sign: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::sign(input_opt->get()));
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

// Handler for stablehlo.ceil
bool HandleCeil(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.ceil: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::ceil(input_opt->get()));
    return true;
}

// Handler for stablehlo.round_nearest_even
bool HandleRoundNearestEven(mlir::Operation* op, ValueMap& values,
                            std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.round_nearest_even: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::round(input_opt->get()));
    return true;
}

// Handler for stablehlo.is_finite
bool HandleIsFinite(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.is_finite: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::isfinite(input_opt->get()));
    return true;
}

// Handler for stablehlo.exponential_minus_one
bool HandleExpm1(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.exponential_minus_one: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::expm1(input_opt->get()));
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

// Handler for stablehlo.atan2 (maps to atan2 in MLX via arctan2)
bool HandleAtan2(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.atan2: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::arctan2(lhs_opt->get(), rhs_opt->get()));
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
    }
    val = mlx::core::astype(val, mlx::core::uint32);

    // Mask to original bit width to ensure upper bits are zero
    if (bit_width < 32) {
        uint32_t mask = (1U << bit_width) - 1;
        val = mlx::core::bitwise_and(val, mlx::core::array(mask, mlx::core::uint32));
    }

    // Standard Hamming weight algorithm
    auto m1 = mlx::core::array(0x55555555U, mlx::core::uint32);
    auto m2 = mlx::core::array(0x33333333U, mlx::core::uint32);
    auto m4 = mlx::core::array(0x0F0F0F0FU, mlx::core::uint32);
    val = mlx::core::subtract(
        val, mlx::core::bitwise_and(
                 mlx::core::right_shift(val, mlx::core::array(1, mlx::core::uint32)), m1));
    val = mlx::core::add(
        mlx::core::bitwise_and(val, m2),
        mlx::core::bitwise_and(mlx::core::right_shift(val, mlx::core::array(2, mlx::core::uint32)),
                               m2));
    val = mlx::core::bitwise_and(
        mlx::core::add(val, mlx::core::right_shift(val, mlx::core::array(4, mlx::core::uint32))),
        m4);
    val = mlx::core::add(val, mlx::core::right_shift(val, mlx::core::array(8, mlx::core::uint32)));
    val = mlx::core::add(val, mlx::core::right_shift(val, mlx::core::array(16, mlx::core::uint32)));
    val = mlx::core::bitwise_and(val, mlx::core::array(0x3FU, mlx::core::uint32));
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(val, dtype));
    return true;
}

// Apply edge padding that may include negative values (trimming).
// Negative low/high means slice from that side; positive means pad.
mlx::core::array ApplyEdgePadding(const mlx::core::array& input,
                                  llvm::ArrayRef<int64_t> edgePaddingLow,
                                  llvm::ArrayRef<int64_t> edgePaddingHigh,
                                  const mlx::core::array& padValue) {
    auto ndim = edgePaddingLow.size();
    auto result = input;

    // Trim (slice) for any negative padding values.
    bool hasNeg = false;
    for (size_t i = 0; i < ndim; ++i) {
        if (edgePaddingLow[i] < 0 || edgePaddingHigh[i] < 0) {
            hasNeg = true;
            break;
        }
    }
    if (hasNeg) {
        mlx::core::Shape starts;
        mlx::core::Shape stops;
        mlx::core::Shape strides;
        auto shape = result.shape();
        for (size_t i = 0; i < ndim; ++i) {
            int64_t lo = edgePaddingLow[i];
            int64_t hi = edgePaddingHigh[i];
            starts.push_back(static_cast<int>(lo < 0 ? -lo : 0));
            stops.push_back(static_cast<int>(shape[i] + (hi < 0 ? hi : 0)));
            strides.push_back(1);
        }
        result = mlx::core::slice(result, starts, stops, strides);
    }

    // Pad with clamped-to-zero values.
    std::vector<std::pair<int, int>> padWidths;
    padWidths.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        padWidths.emplace_back(static_cast<int>(std::max<int64_t>(edgePaddingLow[i], 0)),
                               static_cast<int>(std::max<int64_t>(edgePaddingHigh[i], 0)));
    }
    return mlx::core::pad(result, padWidths, padValue);
}

// Handler for stablehlo.pad
bool HandlePad(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
    if (!padOp) {
        MPS_LOG_ERROR("stablehlo.pad: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, padOp.getOperand());
    auto padValue_opt = GetValue(values, padOp.getPaddingValue());
    if (!input_opt || !padValue_opt) {
        MPS_LOG_ERROR("stablehlo.pad: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto& padValue = padValue_opt->get();
    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();

    // Check for interior padding (not yet supported)
    bool hasInterior = false;
    for (int64_t p : interiorPadding) {
        if (p != 0) {
            hasInterior = true;
            break;
        }
    }

    if (hasInterior) {
        // Interior padding: insert `p` copies of padValue between each pair of
        // existing elements along each axis, then apply edge padding.
        //
        // For an axis of size N with interior padding p, the result has size
        // N + (N-1)*p (before edge padding).
        auto result = input;
        auto ndim = edgePaddingLow.size();

        for (size_t axis = 0; axis < ndim; ++axis) {
            auto p = interiorPadding[axis];
            if (p <= 0)
                continue;

            auto shape = result.shape();
            auto axisSize = shape[axis];
            if (axisSize <= 1)
                continue;

            // Build the dilated result by interleaving slices with padding.
            // New axis size = axisSize + (axisSize - 1) * p
            auto newAxisSize = static_cast<int32_t>(axisSize + (axisSize - 1) * p);

            // Create a full-sized array of padValue, then scatter original values.
            mlx::core::Shape newShape(shape.begin(), shape.end());
            newShape[axis] = newAxisSize;
            auto padScalar = mlx::core::broadcast_to(padValue, {1});
            auto dilated = mlx::core::broadcast_to(
                mlx::core::reshape(padScalar, mlx::core::Shape(ndim, 1)), newShape);

            // Build indices for the original elements: 0, p+1, 2*(p+1), ...
            std::vector<int32_t> idxVals(axisSize);
            for (int32_t i = 0; i < axisSize; ++i) {
                idxVals[i] = i * static_cast<int32_t>(p + 1);
            }
            auto indices = mlx::core::array(idxVals.data(), {axisSize}, mlx::core::int32);

            // Use take + put pattern: gather from result along axis, place into dilated.
            // Actually, the simplest approach: use slice + scatter via put_along_axis
            // or build with concatenation.
            //
            // Simplest correct approach: iterate and concatenate.
            // For each element i in [0, axisSize), take slice i, then append p pad slices.
            // But that's O(N) concatenations which is slow.
            //
            // Better: use scatter. Create zeros of the target shape, then scatter
            // original values at the strided positions.
            //
            // Even better: use as_strided on a zero-initialized array.
            // Simplest correct: create output full of pad_value, then use
            // scatter with strided indices.

            // Create the dilated array filled with padValue.
            dilated = mlx::core::full(newShape, padValue);

            // Scatter original values at strided positions along this axis.
            // We need to put result[..., i, ...] at position i*(p+1) along axis.
            result = mlx::core::put_along_axis(dilated,
                                               mlx::core::reshape(indices,
                                                                  [&]() {
                                                                      mlx::core::Shape s(ndim, 1);
                                                                      s[axis] = axisSize;
                                                                      return s;
                                                                  }()),
                                               result, static_cast<int>(axis));
        }

        values.emplace(ToKey(op->getResult(0)),
                       ApplyEdgePadding(result, edgePaddingLow, edgePaddingHigh, padValue));
        return true;
    }

    values.emplace(ToKey(op->getResult(0)),
                   ApplyEdgePadding(input, edgePaddingLow, edgePaddingHigh, padValue));
    return true;
}

// Handler for stablehlo.dynamic_update_slice
bool HandleDynamicUpdateSlice(mlir::Operation* op, ValueMap& values,
                              std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dusOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(op);
    if (!dusOp) {
        MPS_LOG_ERROR("stablehlo.dynamic_update_slice: failed to cast\n");
        return false;
    }

    auto operand_opt = GetValue(values, dusOp.getOperand());
    auto update_opt = GetValue(values, dusOp.getUpdate());
    if (!operand_opt || !update_opt) {
        MPS_LOG_ERROR("stablehlo.dynamic_update_slice: operand not found in value map\n");
        return false;
    }

    auto& operand = operand_opt->get();
    auto& update = update_opt->get();

    // Empty update is a no-op
    if (update.size() == 0) {
        values.emplace(ToKey(op->getResult(0)), operand);
        return true;
    }

    // Use purely functional MLX ops (no eval) so this works inside mlx::core::compile() tracing.
    // For each dimension, build a mask indicating which positions fall within the
    // update region, and gather from the update using clamped relative indices.
    auto gathered = update;
    auto combined_mask = mlx::core::array(true);
    for (int d = 0; d < static_cast<int>(operand.ndim()); ++d) {
        auto idx_opt = GetValue(values, dusOp.getStartIndices()[d]);
        if (!idx_opt) {
            MPS_LOG_ERROR("stablehlo.dynamic_update_slice: start index not found\n");
            return false;
        }
        auto start_idx =
            mlx::core::astype(mlx::core::reshape(idx_opt->get(), {}), mlx::core::int32);

        int op_size = operand.shape(d);
        int up_size = update.shape(d);

        // Clamp start index: max(0, min(start, op_size - up_size))
        start_idx =
            mlx::core::maximum(mlx::core::array(0),
                               mlx::core::minimum(start_idx, mlx::core::array(op_size - up_size)));

        // Relative position of each operand index w.r.t. the update region
        auto arange_d = mlx::core::arange(0, op_size, mlx::core::int32);
        auto relative = mlx::core::subtract(arange_d, start_idx);

        // Per-dimension mask: position is inside update region
        auto mask_d =
            mlx::core::logical_and(mlx::core::greater_equal(relative, mlx::core::array(0)),
                                   mlx::core::less(relative, mlx::core::array(up_size)));

        // Reshape mask for broadcasting: [1, ..., op_size, ..., 1]
        mlx::core::Shape shape(operand.ndim(), 1);
        shape[d] = op_size;
        mask_d = mlx::core::reshape(mask_d, shape);
        combined_mask = mlx::core::logical_and(combined_mask, mask_d);

        // Clamp relative indices for gathering from update
        auto clamped = mlx::core::clip(relative, mlx::core::array(0),
                                       mlx::core::array(std::max(0, up_size - 1)));
        gathered = mlx::core::take(gathered, clamped, d);
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::where(combined_mask, gathered, operand));
    return true;
}

// Handler for stablehlo.gather
bool HandleGather(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op);
    if (!gatherOp) {
        MPS_LOG_ERROR("stablehlo.gather: failed to cast\n");
        return false;
    }

    auto operand_opt = GetValue(values, gatherOp.getOperand());
    auto indices_opt = GetValue(values, gatherOp.getStartIndices());
    if (!operand_opt || !indices_opt) {
        MPS_LOG_ERROR("stablehlo.gather: operand not found in value map\n");
        return false;
    }

    auto& operand = operand_opt->get();
    auto& startIndices = indices_opt->get();

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    auto indexVectorDim = static_cast<int>(dimNumbers.getIndexVectorDim());
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();

    // Simple case: single index dimension, optionally collapsed
    // This handles patterns like: gather(data, indices) -> data[indices]
    // When collapsedSliceDims is non-empty, the gathered dim is removed from output.
    // When collapsedSliceDims is empty, the gathered dim is kept (as a slice dim).
    if (startIndexMap.size() == 1 &&
        (collapsedSliceDims.empty() ||
         (collapsedSliceDims.size() == 1 &&
          static_cast<int64_t>(startIndexMap[0]) == static_cast<int64_t>(collapsedSliceDims[0])))) {
        int gatherDim = static_cast<int>(startIndexMap[0]);
        bool collapsed = !collapsedSliceDims.empty();

        // Extract the index vector
        auto indices = startIndices;
        if (indexVectorDim < static_cast<int>(startIndices.shape().size())) {
            // Index vector dim exists - squeeze it if it's size 1 and the dim is collapsed
            // When not collapsed, keep the index vector dim so take_along_axis
            // produces the right output shape (with the slice dimension).
            if (collapsed && startIndices.shape(indexVectorDim) == 1) {
                indices = mlx::core::squeeze(startIndices, {indexVectorDim});
            }
        }

        // Ensure indices are int32
        if (indices.dtype() != mlx::core::int32) {
            indices = mlx::core::astype(indices, mlx::core::int32);
        }

        mlx::core::array result = [&]() {
            if (!operandBatchingDims.empty()) {
                // Batched gather: use take_along_axis which naturally handles
                // per-element indexing (result[b,i] = operand[b, indices[b,i]]).
                // take_along_axis requires indices to have the same ndim as operand.
                // Append trailing size-1 dims for any offset dims beyond the gather axis.
                auto batchedIndices = indices;
                if (batchedIndices.ndim() < operand.ndim()) {
                    mlx::core::Shape expandedShape = batchedIndices.shape();
                    while (static_cast<int>(expandedShape.size()) < operand.ndim()) {
                        expandedShape.push_back(1);
                    }
                    batchedIndices = mlx::core::reshape(batchedIndices, expandedShape);
                }
                return mlx::core::take_along_axis(operand, batchedIndices, gatherDim);
            }
            return mlx::core::take(operand, indices, gatherDim);
        }();

        // Check if we need to rearrange dimensions to match the expected output layout
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (resultType) {
            auto expectedShape = GetShape(resultType);
            auto actualShape = result.shape();
            if (actualShape != expectedShape) {
                result = mlx::core::reshape(result, expectedShape);
            }
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Multi-dim gather: multiple start_index_map dims, all collapsed
    // This handles patterns like operand[idx[0], idx[1], ..., idx[N-1]]
    // where indexVectorDim points to the axis containing per-dim indices.
    // When offset_dims is non-empty, some operand dims are preserved in output.
    if (startIndexMap.size() == collapsedSliceDims.size() && startIndexMap.size() > 1) {
        auto indices = startIndices;
        auto idxShape = indices.shape();

        // Build per-axis index arrays and extract from index_vector_dim
        std::vector<mlx::core::array> idxVec;
        std::vector<int> axes;
        for (size_t i = 0; i < startIndexMap.size(); ++i) {
            int axis = static_cast<int>(startIndexMap[i]);
            axes.push_back(axis);

            mlx::core::Shape starts(indices.ndim(), 0);
            mlx::core::Shape stops(idxShape.begin(), idxShape.end());
            starts[indexVectorDim] = static_cast<int>(i);
            stops[indexVectorDim] = static_cast<int>(i) + 1;
            auto axisIndices =
                mlx::core::squeeze(mlx::core::slice(indices, starts, stops), {indexVectorDim});
            if (axisIndices.dtype() != mlx::core::int32) {
                axisIndices = mlx::core::astype(axisIndices, mlx::core::int32);
            }
            // Clamp to valid range
            axisIndices =
                mlx::core::clip(axisIndices, mlx::core::array(0, mlx::core::int32),
                                mlx::core::array(operand.shape(axis) - 1, mlx::core::int32));
            idxVec.push_back(axisIndices);
        }

        // Identify which operand dims are offset dims (not collapsed)
        std::set<int> collapsedSet(collapsedSliceDims.begin(), collapsedSliceDims.end());
        std::vector<int> offsetOperandDims;
        for (int d = 0; d < operand.ndim(); ++d) {
            if (collapsedSet.count(d) == 0) {
                offsetOperandDims.push_back(d);
            }
        }

        mlx::core::array result = operand;

        if (offsetOperandDims.empty()) {
            // Full-index gather: all dims collapsed, flatten and use linear indices
            auto flatOperand = mlx::core::flatten(operand);
            auto linearIdx = mlx::core::array(0, mlx::core::int32);
            for (size_t i = 0; i < axes.size(); ++i) {
                int stride = 1;
                for (int d = axes[i] + 1; d < operand.ndim(); ++d) {
                    stride *= operand.shape(d);
                }
                linearIdx = mlx::core::add(
                    linearIdx,
                    mlx::core::multiply(idxVec[i], mlx::core::array(stride, mlx::core::int32)));
            }
            result = mlx::core::take(flatOperand, linearIdx, 0);
        } else {
            // Partial-index gather: some dims are offset (preserved), rest are collapsed.
            // Transpose operand to put offset dims first, collapsed dims last,
            // flatten the collapsed dims, compute linear indices, then use take_along_axis.
            std::vector<int> perm;
            perm.reserve(operand.ndim());
            for (int d : offsetOperandDims) {
                perm.push_back(d);
            }
            for (int d = 0; d < operand.ndim(); ++d) {
                if (collapsedSet.count(d) != 0) {
                    perm.push_back(d);
                }
            }

            bool needsPerm = false;
            for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
                if (perm[i] != i) {
                    needsPerm = true;
                    break;
                }
            }
            auto permuted = needsPerm ? mlx::core::transpose(operand, perm) : operand;

            // Flatten collapsed dims into a single dim
            int numOffset = static_cast<int>(offsetOperandDims.size());
            mlx::core::Shape flatShape;
            for (int i = 0; i < numOffset; ++i) {
                flatShape.push_back(permuted.shape(i));
            }
            int flattenedSize = 1;
            for (int i = numOffset; i < permuted.ndim(); ++i) {
                flattenedSize *= permuted.shape(i);
            }
            flatShape.push_back(flattenedSize);
            auto flatOperand = mlx::core::reshape(permuted, flatShape);

            // Compute linear indices within the collapsed dims.
            // Collapsed dims are flattened in sorted operand-dim order, so compute
            // strides based on that order rather than startIndexMap order.
            std::vector<int> collapsedDimsSorted(collapsedSet.begin(), collapsedSet.end());
            std::sort(collapsedDimsSorted.begin(), collapsedDimsSorted.end());
            std::map<int, int> strideMap;
            int s = 1;
            for (int i = static_cast<int>(collapsedDimsSorted.size()) - 1; i >= 0; --i) {
                strideMap[collapsedDimsSorted[i]] = s;
                s *= operand.shape(collapsedDimsSorted[i]);
            }
            auto linearIdx = mlx::core::array(0, mlx::core::int32);
            for (size_t i = 0; i < axes.size(); ++i) {
                linearIdx = mlx::core::add(
                    linearIdx, mlx::core::multiply(idxVec[i], mlx::core::array(strideMap[axes[i]],
                                                                               mlx::core::int32)));
            }

            // Expand linearIdx to have ndim == flatOperand.ndim() for take_along_axis.
            // Prepend numOffset singleton dims (offset axes) and append any missing
            // trailing dims so the total rank matches flatOperand exactly.
            mlx::core::Shape expandedShape(numOffset, 1);
            auto idxBatchShape = linearIdx.shape();
            for (int d : idxBatchShape) {
                expandedShape.push_back(d);
            }
            while (static_cast<int>(expandedShape.size()) < flatOperand.ndim()) {
                expandedShape.push_back(1);
            }
            linearIdx = mlx::core::reshape(linearIdx, expandedShape);

            result = mlx::core::take_along_axis(flatOperand, linearIdx, numOffset);
        }

        // Reshape to expected output shape
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (resultType) {
            auto expectedShape = GetShape(resultType);
            if (result.shape() != expectedShape) {
                result = mlx::core::reshape(result, expectedShape);
            }
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    MPS_LOG_ERROR("stablehlo.gather: unsupported gather pattern (indexVectorDim=%d, "
                  "startIndexMap.size=%zu, collapsedSliceDims.size=%zu)\n",
                  indexVectorDim, startIndexMap.size(), collapsedSliceDims.size());
    return false;
}

// Handler for stablehlo.constant
bool HandleConstant(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto constOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constOp) {
        MPS_LOG_ERROR("stablehlo.constant: failed to cast to ConstantOp\n");
        return false;
    }

    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!attr) {
        MPS_LOG_ERROR("stablehlo.constant: value is not DenseElementsAttr\n");
        return false;
    }

    auto arr_opt = CreateArrayFromDenseAttr(attr);
    if (!arr_opt) {
        return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(*arr_opt));
    return true;
}

// Handler for stablehlo.reshape
bool HandleReshape(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.reshape: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.reshape: result type is not RankedTensorType\n");
        return false;
    }

    auto newShape = GetShape(resultType);
    values.emplace(ToKey(op->getResult(0)), mlx::core::reshape(input_opt->get(), newShape));
    return true;
}

// Handler for stablehlo.broadcast_in_dim
bool HandleBroadcastInDim(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
    if (!broadcastOp) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.broadcast_in_dim: result type is not RankedTensorType\n");
        return false;
    }

    auto& input = input_opt->get();
    auto outputShape = GetShape(resultType);
    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    // Validate broadcast dimensions are in bounds
    for (int64_t dim : broadcastDims) {
        if (dim < 0 || static_cast<size_t>(dim) >= outputShape.size()) {
            MPS_LOG_ERROR("stablehlo.broadcast_in_dim: dimension %lld out of bounds [0, %zu)\n",
                          dim, outputShape.size());
            return false;
        }
    }

    // Build input shape
    mlx::core::Shape inputShape;
    for (int i = 0; i < input.ndim(); ++i) {
        inputShape.push_back(input.shape(i));
    }

    // Build the intermediate shape with 1s for non-broadcast dims
    // For broadcast_in_dim, we reshape input to have 1s in dimensions not in broadcastDims,
    // then broadcast to the final shape
    mlx::core::Shape intermediateShape(outputShape.size(), 1);
    for (size_t i = 0; i < broadcastDims.size(); ++i) {
        int64_t dim = broadcastDims[i];
        if (i < inputShape.size()) {
            intermediateShape[dim] = inputShape[i];
        }
    }

    // Reshape input to intermediate shape
    auto reshaped = mlx::core::reshape(input, intermediateShape);

    // Broadcast to final shape
    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, outputShape));
    return true;
}

// Common helper for return-like operations (func.return, stablehlo.return)
bool CollectReturnValues(mlir::Operation* op, ValueMap& values,
                         std::vector<mlx::core::array>& outputs, const char* opName) {
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        outputs.push_back(val_opt->get());
    }
    return true;
}

// Handler for func.return
bool HandleReturn(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "func.return");
}

// Handler for stablehlo.concatenate
bool HandleConcatenate(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op);
    if (!concatOp) {
        MPS_LOG_ERROR("stablehlo.concatenate: failed to cast\n");
        return false;
    }

    std::vector<mlx::core::array> inputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.concatenate: operand not found in value map\n");
            return false;
        }
        inputs.push_back(val_opt->get());
    }

    auto axis = concatOp.getDimension();
    values.emplace(ToKey(op->getResult(0)), mlx::core::concatenate(inputs, static_cast<int>(axis)));
    return true;
}

// Handler for stablehlo.and (bitwise and)
bool HandleAnd(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.and: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_and(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.convert (type conversion)
bool HandleConvert(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.convert: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.convert: result type is not RankedTensorType\n");
        return false;
    }

    auto targetDtype = MlirTypeToMlxDtype(resultType.getElementType());
    values.emplace(ToKey(op->getResult(0)), mlx::core::astype(input_opt->get(), targetDtype));
    return true;
}

// Handler for stablehlo.shift_right_logical
bool HandleShiftRightLogical(mlir::Operation* op, ValueMap& values,
                             std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.shift_right_logical: operand not found in value map\n");
        return false;
    }
    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();

    // StableHLO spec: shift < 0 or shift >= bit_width gives 0 for logical right shift
    int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
    auto zero = mlx::core::zeros_like(lhs);
    auto oob = mlx::core::logical_or(
        mlx::core::less(rhs, mlx::core::array(0, rhs.dtype())),
        mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype())));
    auto shifted =
        mlx::core::right_shift(lhs, mlx::core::maximum(rhs, mlx::core::array(0, rhs.dtype())));
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(oob, zero, shifted));
    return true;
}

// Handler for stablehlo.multiply
bool HandleMultiply(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.multiply: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::multiply(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.xor
bool HandleXor(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.xor: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_xor(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.or
bool HandleOr(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
              ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.or: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::bitwise_or(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.shift_left
bool HandleShiftLeft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.shift_left: operand not found in value map\n");
        return false;
    }
    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();

    // StableHLO spec: shift < 0 or shift >= bit_width gives 0 for left shift
    int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
    auto zero = mlx::core::zeros_like(lhs);
    auto oob = mlx::core::logical_or(
        mlx::core::less(rhs, mlx::core::array(0, rhs.dtype())),
        mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype())));
    auto shifted =
        mlx::core::left_shift(lhs, mlx::core::maximum(rhs, mlx::core::array(0, rhs.dtype())));
    values.emplace(ToKey(op->getResult(0)), mlx::core::where(oob, zero, shifted));
    return true;
}

// Handler for stablehlo.iota
bool HandleIota(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        MPS_LOG_ERROR("stablehlo.iota: failed to cast\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.iota: result type is not RankedTensorType\n");
        return false;
    }

    auto shape = GetShape(resultType);
    auto dtype = MlirTypeToMlxDtype(resultType.getElementType());
    uint64_t iotaDim = iotaOp.getIotaDimension();

    // Create iota: values are 0, 1, 2, ... along the iota dimension
    // Start with arange for the iota dimension size
    int dimSize = shape[iotaDim];
    auto iota1d = mlx::core::arange(0, dimSize, dtype);

    // Reshape to have 1s everywhere except the iota dimension
    mlx::core::Shape reshapeShape(shape.size(), 1);
    reshapeShape[iotaDim] = dimSize;
    auto reshaped = mlx::core::reshape(iota1d, reshapeShape);

    // Broadcast to final shape
    values.emplace(ToKey(op->getResult(0)), mlx::core::broadcast_to(reshaped, shape));
    return true;
}

// Handler for stablehlo.slice
bool HandleSlice(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
    if (!sliceOp) {
        MPS_LOG_ERROR("stablehlo.slice: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.slice: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto startIndices = sliceOp.getStartIndices();
    auto limitIndices = sliceOp.getLimitIndices();
    auto strides = sliceOp.getStrides();

    mlx::core::Shape starts;
    mlx::core::Shape stops;
    mlx::core::Shape steps;
    for (size_t i = 0; i < startIndices.size(); ++i) {
        starts.push_back(static_cast<int>(startIndices[i]));
        stops.push_back(static_cast<int>(limitIndices[i]));
        steps.push_back(static_cast<int>(strides[i]));
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::slice(input, starts, stops, steps));
    return true;
}

// Handler for stablehlo.dynamic_slice
bool HandleDynamicSlice(mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto dynamicSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
    if (!dynamicSliceOp) {
        MPS_LOG_ERROR("stablehlo.dynamic_slice: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.dynamic_slice: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto sliceSizes = dynamicSliceOp.getSliceSizes();

    // Use purely functional MLX ops (no eval) so this works inside mlx::core::compile() tracing.
    // For each dimension, compute indices as start + arange(size) and use take().
    auto result = input;
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
        auto idx_opt = GetValue(values, op->getOperand(i));
        if (!idx_opt) {
            MPS_LOG_ERROR("stablehlo.dynamic_slice: start index operand not found\n");
            return false;
        }
        auto start_idx = mlx::core::astype(idx_opt->get(), mlx::core::int32);
        int size = static_cast<int>(sliceSizes[i - 1]);
        int axis = static_cast<int>(i - 1);
        int dim_size = input.shape(axis);

        // Clamp start index per StableHLO spec: max(0, min(start, dim_size - size))
        start_idx = mlx::core::maximum(
            mlx::core::array(0), mlx::core::minimum(start_idx, mlx::core::array(dim_size - size)));

        // Create indices: clamped_start + [0, 1, 2, ..., size-1]
        auto offsets = mlx::core::arange(0, size, mlx::core::int32);
        auto indices = mlx::core::add(start_idx, offsets);
        result = mlx::core::take(result, indices, axis);
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.subtract
bool HandleSubtract(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.subtract: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::subtract(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.negate
bool HandleNegate(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.negate: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::negative(input_opt->get()));
    return true;
}

// Handler for stablehlo.abs
bool HandleAbs(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.abs: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::abs(input_opt->get()));
    return true;
}

// Handler for stablehlo.sqrt
bool HandleSqrt(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.sqrt: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::sqrt(input_opt->get()));
    return true;
}

// Handler for stablehlo.log_plus_one (log1p)
bool HandleLogPlusOne(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.log_plus_one: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::log1p(input_opt->get()));
    return true;
}

// Handler for stablehlo.maximum
bool HandleMaximum(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.maximum: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::maximum(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Handler for stablehlo.divide
bool HandleDivide(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.divide: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::divide(lhs_opt->get(), rhs_opt->get()));
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

// Handler for stablehlo.bitcast_convert
bool HandleBitcastConvert(mlir::Operation* op, ValueMap& values,
                          std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.bitcast_convert: operand not found in value map\n");
        return false;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("stablehlo.bitcast_convert: result type is not RankedTensorType\n");
        return false;
    }

    auto targetDtype = MlirTypeToMlxDtype(resultType.getElementType());
    // MLX view function reinterprets the underlying data as a different type
    values.emplace(ToKey(op->getResult(0)), mlx::core::view(input_opt->get(), targetDtype));
    return true;
}

// Forward declaration for executing a region
// parentValues allows inner regions to access values defined in outer scopes
// (e.g., constants hoisted out of while-loop regions by optimization passes)
bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx,
                   const ValueMap* parentValues = nullptr);

// Handler for stablehlo.return (region terminator)
bool HandleStablehloReturn(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "stablehlo.return");
}

// Handler for stablehlo.while
bool HandleWhile(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op);
    if (!whileOp) {
        MPS_LOG_ERROR("stablehlo.while: failed to cast\n");
        return false;
    }

    // While loops require eval() each iteration for the condition and to bound
    // memory, which is incompatible with mlx::core::compile() tracing.
    if (ctx.inside_compile) {
        throw CompileIncompatibleError("stablehlo.while requires eval()");
    }

    // Get initial values from operands
    std::vector<mlx::core::array> loopVars;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.while: operand not found in value map\n");
            return false;
        }
        loopVars.push_back(val_opt->get());
    }

    auto& condRegion = whileOp.getCond();
    auto& bodyRegion = whileOp.getBody();

    // Try to compile condition and body regions into fused MLX functions so each
    // iteration is a single compiled dispatch instead of N individual ops.
    // Falls back to interpreted if compile() itself throws or if the regions
    // contain ops incompatible with compile tracing (e.g. case/switch need eval).
    using CompiledFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;
    CompiledFn compiledCond;
    CompiledFn compiledBody;
    bool useCompiled = false;
    try {
        compiledCond = mlx::core::compile(
            [&condRegion, &ctx, &values](
                const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                auto args = inputs;
                std::vector<mlx::core::array> results;
                ExecContext compileCtx;
                compileCtx.module = ctx.module;
                compileCtx.inside_compile = true;
                if (!ExecuteRegion(condRegion, args, results, compileCtx, &values)) {
                    return {};
                }
                return results;
            });

        compiledBody = mlx::core::compile(
            [&bodyRegion, &ctx, &values](
                const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                auto args = inputs;
                std::vector<mlx::core::array> results;
                ExecContext compileCtx;
                compileCtx.module = ctx.module;
                compileCtx.inside_compile = true;
                if (!ExecuteRegion(bodyRegion, args, results, compileCtx, &values)) {
                    return {};
                }
                return results;
            });

        // Probe: run compiled cond+body once to verify they work
        auto testCond = compiledCond(loopVars);
        if (testCond.size() == 1) {
            auto testBody = compiledBody(loopVars);
            if (testBody.size() == loopVars.size()) {
                std::vector<mlx::core::array> toEval;
                toEval.insert(toEval.end(), testCond.begin(), testCond.end());
                toEval.insert(toEval.end(), testBody.begin(), testBody.end());
                mlx::core::eval(toEval);
                useCompiled = true;
            }
        }
    } catch (...) {
        useCompiled = false;
    }

    while (true) {
        std::vector<mlx::core::array> condResults;
        if (useCompiled) {
            condResults = compiledCond(loopVars);
        } else {
            if (!ExecuteRegion(condRegion, loopVars, condResults, ctx, &values)) {
                MPS_LOG_ERROR("stablehlo.while: failed to execute cond region\n");
                return false;
            }
        }

        if (condResults.size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: cond region should return 1 value, got %zu\n",
                          condResults.size());
            return false;
        }

        // Evaluate condition (and loop vars from previous iteration to bound memory).
        // Combining into a single eval() call minimizes GPU sync round-trips.
        loopVars.push_back(condResults[0]);
        mlx::core::eval(loopVars);
        auto condVal = std::move(loopVars.back());
        loopVars.pop_back();

        if (condVal.size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: condition must be a scalar, got size %zu\n",
                          condVal.size());
            return false;
        }

        if (!condVal.item<bool>()) {
            break;
        }

        std::vector<mlx::core::array> bodyResults;
        if (useCompiled) {
            bodyResults = compiledBody(loopVars);
        } else {
            if (!ExecuteRegion(bodyRegion, loopVars, bodyResults, ctx, &values)) {
                MPS_LOG_ERROR("stablehlo.while: failed to execute body region\n");
                return false;
            }
        }

        if (bodyResults.size() != loopVars.size()) {
            MPS_LOG_ERROR("stablehlo.while: body returned %zu values, expected %zu\n",
                          bodyResults.size(), loopVars.size());
            return false;
        }

        // Update loop variables (evaluated at start of next iteration)
        loopVars = std::move(bodyResults);
    }

    // Map final loop variables to results
    for (size_t i = 0; i < loopVars.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(loopVars[i]));
    }

    return true;
}

// Handler for stablehlo.case (implements lax.cond and lax.switch)
bool HandleCase(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto caseOp = mlir::dyn_cast<mlir::stablehlo::CaseOp>(op);
    if (!caseOp) {
        MPS_LOG_ERROR("stablehlo.case: failed to cast\n");
        return false;
    }

    // Get the branch index
    auto index_opt = GetValue(values, caseOp.getIndex());
    if (!index_opt) {
        MPS_LOG_ERROR("stablehlo.case: index operand not found\n");
        return false;
    }

    // Evaluate the index to determine which branch to take.
    // Control flow ops fundamentally require concrete values to decide which path to execute,
    // so eval() is unavoidable here (same as while loop conditions).
    if (ctx.inside_compile) {
        throw CompileIncompatibleError("stablehlo.case requires eval()");
    }
    mlx::core::eval(index_opt->get());
    int branchIdx = index_opt->get().item<int>();

    // Clamp to valid range (out-of-bounds goes to last branch per StableHLO spec)
    int numBranches = static_cast<int>(caseOp.getBranches().size());
    if (branchIdx < 0 || branchIdx >= numBranches) {
        branchIdx = numBranches - 1;
    }

    // Execute the selected branch
    auto& branch = caseOp.getBranches()[branchIdx];
    std::vector<mlx::core::array> branchArgs;  // case branches take no args
    std::vector<mlx::core::array> branchResults;
    if (!ExecuteRegion(branch, branchArgs, branchResults, ctx, &values)) {
        MPS_LOG_ERROR("stablehlo.case: failed to execute branch %d\n", branchIdx);
        return false;
    }

    // Map branch results to op results
    for (size_t i = 0; i < branchResults.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(branchResults[i]));
    }

    return true;
}

// Handler for stablehlo.reverse
bool HandleReverse(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(op);
    if (!reverseOp) {
        MPS_LOG_ERROR("stablehlo.reverse: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.reverse: operand not found in value map\n");
        return false;
    }

    auto dimensions = reverseOp.getDimensions();
    auto& input = input_opt->get();
    auto ndim = static_cast<int>(input.ndim());

    // Build set of dimensions to reverse
    std::unordered_set<int64_t> reverseDims(dimensions.begin(), dimensions.end());

    // Use slice with negative strides to reverse dimensions
    // For each dimension, if it's in reverseDims, use step=-1 and swap start/stop
    mlx::core::Shape starts(ndim, 0);
    mlx::core::Shape stops;
    mlx::core::Shape steps(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        int dimSize = input.shape(i);
        if (reverseDims.count(i)) {
            // Reverse: start at end, go to beginning with step -1
            starts[i] = dimSize - 1;
            stops.push_back(-dimSize - 1);  // Past the beginning
            steps[i] = -1;
        } else {
            stops.push_back(dimSize);
        }
    }

    auto result = mlx::core::slice(input, starts, stops, steps);
    // Force contiguous for complex types - MLX's non-contiguous views (from negative strides)
    // can produce incorrect results for complex arrays in subsequent operations
    if (result.dtype() == mlx::core::complex64) {
        result = mlx::core::contiguous(result);
    }
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.transpose
bool HandleTranspose(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        MPS_LOG_ERROR("stablehlo.transpose: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.transpose: operand not found in value map\n");
        return false;
    }

    auto permAttr = transposeOp.getPermutation();
    std::vector<int> axes;
    for (int64_t dim : permAttr) {
        axes.push_back(static_cast<int>(dim));
    }

    values.emplace(ToKey(op->getResult(0)), mlx::core::transpose(input_opt->get(), axes));
    return true;
}

// Handler for stablehlo.power
bool HandlePower(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.power: operand not found in value map\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::power(lhs_opt->get(), rhs_opt->get()));
    return true;
}

// Helper to check if a permutation is the identity (no transpose needed)
bool IsIdentityPermutation(const std::vector<int>& perm) {
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
            return false;
        }
    }
    return true;
}

// Use einsum for dot_general by default (single fused op vs transpose+reshape+matmul+reshape).
// Disable with MPS_NO_EINSUM=1 to fall back to the manual path.
bool UseEinsumForDotGeneral() {
    static bool use_einsum = std::getenv("MPS_NO_EINSUM") == nullptr;
    return use_einsum;
}

// Build einsum subscript string for dot_general
// Returns (lhs_subscript, rhs_subscript, output_subscript)
std::string BuildEinsumSubscript(int lhsRank, int rhsRank, llvm::ArrayRef<int64_t> lhsBatchDims,
                                 llvm::ArrayRef<int64_t> rhsBatchDims,
                                 llvm::ArrayRef<int64_t> lhsContractDims,
                                 llvm::ArrayRef<int64_t> rhsContractDims) {
    // Use letters a-z for dimensions
    char nextChar = 'a';

    // Maps from dimension index to einsum character
    std::vector<char> lhsChars(lhsRank, 0);
    std::vector<char> rhsChars(rhsRank, 0);

    // Assign shared characters for batch dims
    for (size_t i = 0; i < lhsBatchDims.size(); ++i) {
        char c = nextChar++;
        lhsChars[lhsBatchDims[i]] = c;
        rhsChars[rhsBatchDims[i]] = c;
    }

    // Assign shared characters for contracting dims
    for (size_t i = 0; i < lhsContractDims.size(); ++i) {
        char c = nextChar++;
        lhsChars[lhsContractDims[i]] = c;
        rhsChars[rhsContractDims[i]] = c;
    }

    // Assign unique characters for free dims
    for (int i = 0; i < lhsRank; ++i) {
        if (lhsChars[i] == 0) {
            lhsChars[i] = nextChar++;
        }
    }
    for (int i = 0; i < rhsRank; ++i) {
        if (rhsChars[i] == 0) {
            rhsChars[i] = nextChar++;
        }
    }

    // Build subscript strings
    std::string lhsSub(lhsChars.begin(), lhsChars.end());
    std::string rhsSub(rhsChars.begin(), rhsChars.end());

    // Build output: batch dims first, then lhs free dims, then rhs free dims
    std::string outSub;
    // Batch dims (from lhs, same as rhs)
    for (int64_t d : lhsBatchDims) {
        outSub += lhsChars[d];
    }
    // LHS free dims (in order of appearance in lhs)
    for (int i = 0; i < lhsRank; ++i) {
        bool isBatch = std::find(lhsBatchDims.begin(), lhsBatchDims.end(), i) != lhsBatchDims.end();
        bool isContract =
            std::find(lhsContractDims.begin(), lhsContractDims.end(), i) != lhsContractDims.end();
        if (!isBatch && !isContract) {
            outSub += lhsChars[i];
        }
    }
    // RHS free dims (in order of appearance in rhs)
    for (int i = 0; i < rhsRank; ++i) {
        bool isBatch = std::find(rhsBatchDims.begin(), rhsBatchDims.end(), i) != rhsBatchDims.end();
        bool isContract =
            std::find(rhsContractDims.begin(), rhsContractDims.end(), i) != rhsContractDims.end();
        if (!isBatch && !isContract) {
            outSub += rhsChars[i];
        }
    }

    return lhsSub + "," + rhsSub + "->" + outSub;
}

// Helper to detect reduction type by analyzing the body region
enum class ReduceType { Sum, Max, Min, Prod, And, Or, Unknown };

ReduceType DetectReduceType(mlir::Region& body) {
    if (body.empty())
        return ReduceType::Unknown;

    auto& block = body.front();
    for (auto& op : block.getOperations()) {
        auto opName = op.getName().getStringRef();
        if (opName == "stablehlo.add")
            return ReduceType::Sum;
        if (opName == "stablehlo.maximum")
            return ReduceType::Max;
        if (opName == "stablehlo.minimum")
            return ReduceType::Min;
        if (opName == "stablehlo.multiply")
            return ReduceType::Prod;
        if (opName == "stablehlo.and")
            return ReduceType::And;
        if (opName == "stablehlo.or")
            return ReduceType::Or;
    }
    return ReduceType::Unknown;
}

// Detect argmax/argmin pattern in a 2-input reduce body.
// Returns 1 for argmax, -1 for argmin, 0 for neither.
int DetectArgReducePattern(mlir::Region& body) {
    if (body.empty())
        return 0;

    // argmax/argmin reduces have 4 block args: (val_lhs, val_rhs, idx_lhs, idx_rhs)
    // and use compare + select to implement the reduction.
    // Look for the first compare on the value pair (block args 0 & 1).
    auto& block = body.front();
    if (block.getNumArguments() != 4)
        return 0;

    for (auto& op : block.getOperations()) {
        if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
            // Check if comparing the first two block args (values, not indices)
            auto lhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getLhs());
            auto rhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getRhs());
            if (!lhsArg || !rhsArg)
                continue;
            // Block args for 2-input reduce are interleaved: (val0, idx0, val1, idx1)
            // So comparing values means args 0 vs 2
            if (lhsArg.getArgNumber() == 0 && rhsArg.getArgNumber() == 2) {
                auto dir = cmpOp.getComparisonDirection();
                if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                    dir == mlir::stablehlo::ComparisonDirection::GE)
                    return 1;  // argmax
                if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                    dir == mlir::stablehlo::ComparisonDirection::LE)
                    return -1;  // argmin
            }
        }
    }
    return 0;
}

// Handler for stablehlo.reduce
bool HandleReduce(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        MPS_LOG_ERROR("stablehlo.reduce: failed to cast\n");
        return false;
    }

    // Get reduction dimensions
    auto dimensions = reduceOp.getDimensions();
    std::vector<int> axes;
    for (int64_t dim : dimensions) {
        axes.push_back(static_cast<int>(dim));
    }

    auto& body = reduceOp.getBody();
    size_t numInputs = reduceOp.getInputs().size();

    // Special case: argmax/argmin pattern (2 inputs: values + indices)
    if (numInputs == 2) {
        int argDir = DetectArgReducePattern(body);
        if (argDir != 0) {
            auto input_opt = GetValue(values, reduceOp.getInputs()[0]);
            if (!input_opt) {
                MPS_LOG_ERROR("stablehlo.reduce: argmax/argmin input not found\n");
                return false;
            }

            // argmax/argmin only supports single axis reduction
            if (axes.size() == 1) {
                auto& input = input_opt->get();
                auto idx = (argDir > 0) ? mlx::core::argmax(input, axes[0], /*keepdims=*/false)
                                        : mlx::core::argmin(input, axes[0], /*keepdims=*/false);
                auto val = (argDir > 0) ? mlx::core::max(input, axes) : mlx::core::min(input, axes);

                // Result 0 is the reduced values, result 1 is the indices
                values.emplace(ToKey(op->getResult(0)), std::move(val));
                values.emplace(ToKey(op->getResult(1)), mlx::core::astype(idx, mlx::core::int32));
                return true;
            }
        }
    }

    // Detect reduction type from body
    ReduceType reduceType = DetectReduceType(body);

    // Handle each input-output pair
    for (size_t i = 0; i < numInputs; ++i) {
        auto input_opt = GetValue(values, reduceOp.getInputs()[i]);
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.reduce: input %zu not found in value map\n", i);
            return false;
        }

        std::optional<mlx::core::array> result;
        switch (reduceType) {
            case ReduceType::Sum:
                result = mlx::core::sum(input_opt->get(), axes);
                break;
            case ReduceType::Max:
                result = mlx::core::max(input_opt->get(), axes);
                break;
            case ReduceType::Min:
                result = mlx::core::min(input_opt->get(), axes);
                break;
            case ReduceType::Prod:
                result = mlx::core::prod(input_opt->get(), axes);
                break;
            case ReduceType::And:
                result = mlx::core::all(input_opt->get(), axes);
                break;
            case ReduceType::Or:
                result = mlx::core::any(input_opt->get(), axes);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.reduce: unsupported reduction type\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(i)), std::move(*result));
    }

    return true;
}

// Handler for stablehlo.reduce_window (cumulative ops and pooling)
bool HandleReduceWindow(mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto rwOp = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(op);
    if (!rwOp) {
        MPS_LOG_ERROR("stablehlo.reduce_window: failed to cast\n");
        return false;
    }

    // Only support single-input / single-init / single-result.
    if (rwOp.getInputs().size() != 1 || rwOp.getInitValues().size() != 1 ||
        rwOp->getNumResults() != 1) {
        MPS_LOG_ERROR("stablehlo.reduce_window: only single-input reduce_window is supported\n");
        return false;
    }

    auto input_opt = GetValue(values, rwOp.getInputs()[0]);
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.reduce_window: input not found\n");
        return false;
    }
    auto& input = input_opt->get();

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(rwOp.getInputs()[0].getType());
    if (!inputType) {
        MPS_LOG_ERROR("stablehlo.reduce_window: unranked input\n");
        return false;
    }
    auto inputShape = inputType.getShape();
    auto rank = static_cast<int64_t>(inputShape.size());

    // Handle scalar (0-dimensional) inputs: reduce_window on a scalar is identity.
    if (rank == 0) {
        values.emplace(ToKey(op->getResult(0)), input);
        return true;
    }

    auto windowDims = rwOp.getWindowDimensions();
    auto stridesOpt = rwOp.getWindowStrides();
    auto baseDilOpt = rwOp.getBaseDilations();
    auto winDilOpt = rwOp.getWindowDilations();
    auto paddingAttr = rwOp.getPaddingAttr();

    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> winDil(rank, 1);
    std::vector<int64_t> padLow(rank, 0);
    std::vector<int64_t> padHigh(rank, 0);

    if (stridesOpt) {
        auto s = *stridesOpt;
        for (int64_t i = 0; i < rank; i++)
            strides[i] = s[i];
    }
    if (winDilOpt) {
        auto d = *winDilOpt;
        for (int64_t i = 0; i < rank; i++)
            winDil[i] = d[i];
    }
    if (paddingAttr) {
        auto vals = paddingAttr.getValues<int64_t>();
        for (int64_t i = 0; i < rank; i++) {
            padLow[i] = vals[{(uint64_t)i, 0}];
            padHigh[i] = vals[{(uint64_t)i, 1}];
        }
    }

    bool allStridesOne = !stridesOpt;
    if (stridesOpt) {
        allStridesOne = true;
        for (auto s : *stridesOpt) {
            if (s != 1) {
                allStridesOne = false;
                break;
            }
        }
    }
    bool allBaseDilOne = !baseDilOpt;
    if (baseDilOpt) {
        allBaseDilOne = true;
        for (auto d : *baseDilOpt) {
            if (d != 1) {
                allBaseDilOne = false;
                break;
            }
        }
    }
    bool allWinDilOne = !winDilOpt;
    if (winDilOpt) {
        allWinDilOne = true;
        for (auto d : *winDilOpt) {
            if (d != 1) {
                allWinDilOne = false;
                break;
            }
        }
    }

    ReduceType reduceType = DetectReduceType(rwOp.getBody());

    // ---------- Tier 1: Cumulative pattern ----------
    // All strides=1, all dilations=1, exactly one axis with window == input_shape
    if (allStridesOne && allBaseDilOne && allWinDilOne) {
        int64_t cumAxis = -1;
        bool isCumulative = true;
        for (int64_t i = 0; i < rank; i++) {
            if (windowDims[i] == 1)
                continue;
            if (windowDims[i] == inputShape[i] && cumAxis == -1) {
                cumAxis = i;
            } else {
                isCumulative = false;
                break;
            }
        }

        if (isCumulative && cumAxis >= 0) {
            int64_t axisSize = inputShape[cumAxis];
            bool reverse = false;
            bool inclusive = true;

            if (padLow[cumAxis] == axisSize - 1 && padHigh[cumAxis] == 0) {
                reverse = false;
                inclusive = true;
            } else if (padLow[cumAxis] == 0 && padHigh[cumAxis] == axisSize - 1) {
                reverse = true;
                inclusive = true;
            } else if (padLow[cumAxis] == axisSize && padHigh[cumAxis] == -1) {
                reverse = false;
                inclusive = false;
            } else if (padLow[cumAxis] == -1 && padHigh[cumAxis] == axisSize) {
                reverse = true;
                inclusive = false;
            } else {
                MPS_LOG_ERROR("stablehlo.reduce_window: unsupported cumulative padding pattern\n");
                return false;
            }

            // Check non-cumulative axes have zero padding.
            for (int64_t i = 0; i < rank; i++) {
                if (i == cumAxis)
                    continue;
                if (padLow[i] != 0 || padHigh[i] != 0) {
                    MPS_LOG_ERROR(
                        "stablehlo.reduce_window: non-zero padding on non-cumulative axis\n");
                    return false;
                }
            }

            auto axis = static_cast<int>(cumAxis);
            std::optional<mlx::core::array> result;
            switch (reduceType) {
                case ReduceType::Sum:
                    result = mlx::core::cumsum(input, axis, reverse, inclusive);
                    break;
                case ReduceType::Prod:
                    result = mlx::core::cumprod(input, axis, reverse, inclusive);
                    break;
                case ReduceType::Max:
                    result = mlx::core::cummax(input, axis, reverse, inclusive);
                    break;
                case ReduceType::Min:
                    result = mlx::core::cummin(input, axis, reverse, inclusive);
                    break;
                default:
                    MPS_LOG_ERROR("stablehlo.reduce_window: unsupported cumulative reduce type\n");
                    return false;
            }

            values.emplace(ToKey(op->getResult(0)), std::move(*result));
            return true;
        }
    }

    // ---------- Tier 2: Pooling pattern ----------
    // base_dilations all 1, at least one spatial axis with window > 1
    if (!allBaseDilOne) {
        MPS_LOG_ERROR("stablehlo.reduce_window: base dilations not supported\n");
        return false;
    }

    if (reduceType != ReduceType::Max && reduceType != ReduceType::Sum &&
        reduceType != ReduceType::Min) {
        MPS_LOG_ERROR("stablehlo.reduce_window: pooling supports max/sum/min only\n");
        return false;
    }

    // Pad input if needed.
    mlx::core::array padded = input;
    bool needsPad = false;
    for (int64_t i = 0; i < rank; i++) {
        if (padLow[i] != 0 || padHigh[i] != 0) {
            needsPad = true;
            break;
        }
    }
    if (needsPad) {
        // Get the init value for padding.
        auto init_opt = GetValue(values, rwOp.getInitValues()[0]);
        if (!init_opt) {
            MPS_LOG_ERROR("stablehlo.reduce_window: init value not found\n");
            return false;
        }

        std::vector<std::pair<int, int>> padWidth(rank);
        for (int64_t i = 0; i < rank; i++) {
            padWidth[i] = {static_cast<int>(padLow[i]), static_cast<int>(padHigh[i])};
        }
        padded = mlx::core::pad(input, padWidth, init_opt->get());
    }

    // Compute the output shape from padded input, window, strides, dilation.
    auto paddedShape = padded.shape();
    std::vector<int> outShape(rank);
    for (int64_t i = 0; i < rank; i++) {
        // Effective window size with dilation.
        int64_t effWin = (windowDims[i] - 1) * winDil[i] + 1;
        outShape[i] = static_cast<int>((paddedShape[i] - effWin) / strides[i] + 1);
    }

    // Use as_strided to create a view with window dimensions appended.
    // Output shape: [out_0, out_1, ..., win_0, win_1, ...]
    // Then reduce over the window dimensions.
    mlx::core::Shape viewShape;
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(outShape[i]);
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(static_cast<int32_t>(windowDims[i]));

    // Compute strides for the as_strided view.
    // First compute element strides of the padded array.
    std::vector<int64_t> elemStrides(rank);
    elemStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        elemStrides[i] = elemStrides[i + 1] * paddedShape[i + 1];
    }

    mlx::core::Strides viewStrides;
    // Output dimension strides: move by stride[i] * elemStride[i].
    for (int64_t i = 0; i < rank; i++) {
        viewStrides.push_back(strides[i] * elemStrides[i]);
    }
    // Window dimension strides: move by winDil[i] * elemStride[i].
    for (int64_t i = 0; i < rank; i++) {
        viewStrides.push_back(winDil[i] * elemStrides[i]);
    }

    auto windowed = mlx::core::as_strided(padded, viewShape, viewStrides, 0);

    // Reduce over the window dimensions (axes rank..2*rank-1).
    std::vector<int> reduceAxes;
    for (int64_t i = rank; i < 2 * rank; i++) {
        reduceAxes.push_back(static_cast<int>(i));
    }

    mlx::core::array result = [&]() {
        if (reduceType == ReduceType::Max)
            return mlx::core::max(windowed, reduceAxes);
        if (reduceType == ReduceType::Min)
            return mlx::core::min(windowed, reduceAxes);
        return mlx::core::sum(windowed, reduceAxes);
    }();

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Detect scatter update type from the body region (used by both select_and_scatter and scatter).
enum class ScatterType { Update, Add, Sub, Mul, Min, Max, Unknown };

ScatterType DetectScatterType(mlir::Region& body) {
    if (body.empty())
        return ScatterType::Unknown;

    auto& block = body.front();
    for (auto& op : block.getOperations()) {
        auto opName = op.getName().getStringRef();
        // The body takes (current, update) and returns the result
        // For simple update: return update (the second arg)
        if (opName == "stablehlo.return") {
            // Check if it returns the second block argument directly
            if (op.getNumOperands() == 1) {
                auto returnVal = op.getOperand(0);
                // If the return value is the second block argument, it's an update
                if (returnVal == block.getArgument(1)) {
                    return ScatterType::Update;
                }
            }
        }
        if (opName == "stablehlo.add")
            return ScatterType::Add;
        if (opName == "stablehlo.subtract")
            return ScatterType::Sub;
        if (opName == "stablehlo.multiply")
            return ScatterType::Mul;
        if (opName == "stablehlo.minimum")
            return ScatterType::Min;
        if (opName == "stablehlo.maximum")
            return ScatterType::Max;
    }
    return ScatterType::Unknown;
}

// Handler for stablehlo.select_and_scatter (backward pass of max/min pooling)
bool HandleSelectAndScatter(mlir::Operation* op, ValueMap& values,
                            std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto ssOp = mlir::dyn_cast<mlir::stablehlo::SelectAndScatterOp>(op);
    if (!ssOp) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: failed to cast\n");
        return false;
    }

    auto operand_opt = GetValue(values, ssOp.getOperand());
    auto source_opt = GetValue(values, ssOp.getSource());
    auto init_opt = GetValue(values, ssOp.getInitValue());
    if (!operand_opt || !source_opt || !init_opt) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: operand not found\n");
        return false;
    }
    auto& operand = operand_opt->get();
    auto& source = source_opt->get();
    auto& initValue = init_opt->get();

    auto rank = static_cast<int64_t>(operand.ndim());

    // Parse window attributes (same pattern as reduce_window).
    auto windowDimsOpt = ssOp.getWindowDimensions();
    if (!windowDimsOpt) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: window_dimensions required\n");
        return false;
    }
    auto windowDims = *windowDimsOpt;

    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> padLow(rank, 0);
    std::vector<int64_t> padHigh(rank, 0);

    if (auto s = ssOp.getWindowStrides()) {
        for (int64_t i = 0; i < rank; i++)
            strides[i] = (*s)[i];
    }
    if (auto p = ssOp.getPaddingAttr()) {
        auto vals = p.getValues<int64_t>();
        for (int64_t i = 0; i < rank; i++) {
            padLow[i] = vals[{(uint64_t)i, 0}];
            padHigh[i] = vals[{(uint64_t)i, 1}];
        }
    }

    // Detect select type: GE/GT => max (argmax), LE/LT => min (argmin).
    bool selectMax = true;
    {
        auto& body = ssOp.getSelect();
        if (body.empty()) {
            MPS_LOG_ERROR("stablehlo.select_and_scatter: empty select body\n");
            return false;
        }
        bool found = false;
        for (auto& bodyOp : body.front().getOperations()) {
            if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(bodyOp)) {
                auto dir = cmpOp.getComparisonDirection();
                if (dir == mlir::stablehlo::ComparisonDirection::GE ||
                    dir == mlir::stablehlo::ComparisonDirection::GT) {
                    selectMax = true;
                } else {
                    selectMax = false;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            MPS_LOG_ERROR("stablehlo.select_and_scatter: no compare in select body\n");
            return false;
        }
    }

    // Detect scatter type (typically add for pool gradients).
    ScatterType scatterType = DetectScatterType(ssOp.getScatter());
    if (scatterType != ScatterType::Add) {
        MPS_LOG_ERROR("stablehlo.select_and_scatter: only add scatter supported (got %d)\n",
                      static_cast<int>(scatterType));
        return false;
    }

    // Pad operand so windows can cover edge positions.
    // Pad with -inf (max-select) or +inf (min-select) so padded elements are never selected.
    mlx::core::array padded = operand;
    bool needsPad = false;
    for (int64_t i = 0; i < rank; i++) {
        if (padLow[i] != 0 || padHigh[i] != 0) {
            needsPad = true;
            break;
        }
    }
    if (needsPad) {
        float padScalar = selectMax ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::infinity();
        auto padVal = mlx::core::full({}, padScalar, operand.dtype());
        std::vector<std::pair<int, int>> padWidth(rank);
        for (int64_t i = 0; i < rank; i++)
            padWidth[i] = {static_cast<int>(padLow[i]), static_cast<int>(padHigh[i])};
        padded = mlx::core::pad(operand, padWidth, padVal);
    }

    auto paddedShape = padded.shape();

    // Compute output shape (should match source shape).
    std::vector<int> outShape(rank);
    for (int64_t i = 0; i < rank; i++)
        outShape[i] = static_cast<int>((paddedShape[i] - windowDims[i]) / strides[i] + 1);

    // Create windowed view using as_strided: [out_dims..., win_dims...].
    mlx::core::Shape viewShape;
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(outShape[i]);
    for (int64_t i = 0; i < rank; i++)
        viewShape.push_back(static_cast<int32_t>(windowDims[i]));

    std::vector<int64_t> elemStrides(rank);
    elemStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--)
        elemStrides[i] = elemStrides[i + 1] * paddedShape[i + 1];

    mlx::core::Strides viewStrides;
    for (int64_t i = 0; i < rank; i++)
        viewStrides.push_back(strides[i] * elemStrides[i]);
    for (int64_t i = 0; i < rank; i++)
        viewStrides.push_back(elemStrides[i]);

    auto windowed = mlx::core::as_strided(padded, viewShape, viewStrides, 0);

    // Flatten window dimensions into one axis and find argmax/argmin.
    int64_t winTotal = 1;
    for (int64_t i = 0; i < rank; i++)
        winTotal *= windowDims[i];

    mlx::core::Shape flatWinShape;
    for (int64_t i = 0; i < rank; i++)
        flatWinShape.push_back(outShape[i]);
    flatWinShape.push_back(static_cast<int32_t>(winTotal));

    auto windowedFlat = mlx::core::reshape(windowed, flatWinShape);
    int flatAxis = static_cast<int>(rank);
    auto selectedIdx = selectMax ? mlx::core::argmax(windowedFlat, flatAxis)
                                 : mlx::core::argmin(windowedFlat, flatAxis);
    // argmax/argmin returns uint32; cast to int32 for arithmetic.
    selectedIdx = mlx::core::astype(selectedIdx, mlx::core::int32);

    // Convert flat window indices to linear indices in the (unpadded) operand.
    // Unravel flat index f into per-dimension window coords:
    //   win_coord[k] = (f / divisor[k]) % windowDims[k]
    // Then compute operand position:
    //   operand_pos[k] = out_coord[k] * stride[k] + win_coord[k] - padLow[k]
    // Linear index = sum(operand_pos[k] * operand_elem_stride[k])

    auto operandShape = operand.shape();
    std::vector<int64_t> opElemStrides(rank);
    opElemStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--)
        opElemStrides[i] = opElemStrides[i + 1] * operandShape[i + 1];

    // Divisors for unraveling: divisor[k] = product of windowDims[k+1..rank-1].
    std::vector<int64_t> winDivisors(rank);
    winDivisors[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--)
        winDivisors[i] = winDivisors[i + 1] * windowDims[i + 1];

    auto linearIdx = mlx::core::zeros(selectedIdx.shape(), mlx::core::int32);

    for (int64_t k = 0; k < rank; k++) {
        // Extract window coordinate for dimension k.
        // Note: mlx::core::divide promotes int32 to float32, so we cast back.
        auto quotient = mlx::core::astype(
            mlx::core::divide(selectedIdx, mlx::core::array(static_cast<int32_t>(winDivisors[k]),
                                                            mlx::core::int32)),
            mlx::core::int32);
        auto winCoord = mlx::core::remainder(
            quotient, mlx::core::array(static_cast<int32_t>(windowDims[k]), mlx::core::int32));

        // Create output coordinate array for dimension k (broadcasts with other dims).
        mlx::core::Shape coordShape(rank, 1);
        coordShape[k] = outShape[k];
        auto outCoord =
            mlx::core::reshape(mlx::core::arange(outShape[k], mlx::core::int32), coordShape);

        // operand_pos[k] = out_coord[k] * stride[k] + win_coord[k] - padLow[k]
        auto operandPos = mlx::core::add(
            mlx::core::add(
                mlx::core::multiply(
                    outCoord, mlx::core::array(static_cast<int32_t>(strides[k]), mlx::core::int32)),
                winCoord),
            mlx::core::array(static_cast<int32_t>(-padLow[k]), mlx::core::int32));

        // Accumulate into linear index.
        linearIdx = mlx::core::add(
            linearIdx,
            mlx::core::multiply(operandPos, mlx::core::array(static_cast<int32_t>(opElemStrides[k]),
                                                             mlx::core::int32)));
    }

    // Scatter source values into a flat output initialized with init_value.
    int32_t operandTotal = 1;
    for (int64_t i = 0; i < rank; i++)
        operandTotal *= operandShape[i];

    auto flatOutput = mlx::core::full({operandTotal}, initValue);
    auto flatSource = mlx::core::flatten(source);
    auto flatIdx = mlx::core::flatten(linearIdx);

    // Ensure indices are int32 (required by MLX scatter).
    if (flatIdx.dtype() != mlx::core::int32) {
        flatIdx = mlx::core::astype(flatIdx, mlx::core::int32);
    }

    // MLX scatter expects updates shape [idx_shape..., slice_shape...] where
    // slice_shape has size-1 at scatter axes. For 1D scatter along axis 0,
    // updates need shape [M, 1] and indices shape [M].
    int32_t sourceTotal = 1;
    for (auto d : source.shape())
        sourceTotal *= d;
    flatSource = mlx::core::reshape(flatSource, {sourceTotal, 1});

    auto result = mlx::core::scatter_add(flatOutput, {flatIdx}, flatSource, {0});
    result = mlx::core::reshape(result, operand.shape());

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.dot_general
bool HandleDotGeneral(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        MPS_LOG_ERROR("stablehlo.dot_general: failed to cast\n");
        return false;
    }

    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.dot_general: operand not found in value map\n");
        return false;
    }

    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();
    auto dimNumbers = dotOp.getDotDimensionNumbers();

    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    auto lhsRank = static_cast<int>(lhs.ndim());
    auto rhsRank = static_cast<int>(rhs.ndim());

    // Try einsum path if enabled
    if (UseEinsumForDotGeneral()) {
        std::string subscript = BuildEinsumSubscript(lhsRank, rhsRank, lhsBatchDims, rhsBatchDims,
                                                     lhsContractDims, rhsContractDims);
        MPS_LOG_DEBUG("dot_general einsum: %s\n", subscript.c_str());
        auto result = mlx::core::einsum(subscript, {lhs, rhs});
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Standard path: transpose -> reshape -> matmul -> reshape

    // Build sets for quick lookup
    std::unordered_set<int> lhsContractSet(lhsContractDims.begin(), lhsContractDims.end());
    std::unordered_set<int> rhsContractSet(rhsContractDims.begin(), rhsContractDims.end());
    std::unordered_set<int> lhsBatchSet(lhsBatchDims.begin(), lhsBatchDims.end());
    std::unordered_set<int> rhsBatchSet(rhsBatchDims.begin(), rhsBatchDims.end());

    // Find free dimensions (not batch, not contract)
    std::vector<int> lhsFreeDims;
    std::vector<int> rhsFreeDims;
    for (int i = 0; i < lhsRank; ++i) {
        if (lhsBatchSet.count(i) == 0 && lhsContractSet.count(i) == 0) {
            lhsFreeDims.push_back(i);
        }
    }
    for (int i = 0; i < rhsRank; ++i) {
        if (rhsBatchSet.count(i) == 0 && rhsContractSet.count(i) == 0) {
            rhsFreeDims.push_back(i);
        }
    }

    // For standard matmul, we need exactly one free dimension each
    // LHS: [batch..., M, K] -> free = M, contract = K
    // RHS: [batch..., K, N] -> free = N, contract = K
    // Result: [batch..., M, N]

    // Build permutation for LHS: [batch dims..., free dims..., contract dims...]
    std::vector<int> lhsPerm;
    for (int64_t d : lhsBatchDims)
        lhsPerm.push_back(static_cast<int>(d));
    for (int d : lhsFreeDims)
        lhsPerm.push_back(d);
    for (int64_t d : lhsContractDims)
        lhsPerm.push_back(static_cast<int>(d));

    // Build permutation for RHS: [batch dims..., contract dims..., free dims...]
    std::vector<int> rhsPerm;
    for (int64_t d : rhsBatchDims)
        rhsPerm.push_back(static_cast<int>(d));
    for (int64_t d : rhsContractDims)
        rhsPerm.push_back(static_cast<int>(d));
    for (int d : rhsFreeDims)
        rhsPerm.push_back(d);

    // Transpose to standard form
    auto lhsT = mlx::core::transpose(lhs, lhsPerm);
    auto rhsT = mlx::core::transpose(rhs, rhsPerm);

    // Get shapes for reshape
    int numBatch = static_cast<int>(lhsBatchDims.size());
    int numLhsFree = static_cast<int>(lhsFreeDims.size());
    int numContract = static_cast<int>(lhsContractDims.size());

    // Calculate combined dimensions
    int64_t batchSize = 1;
    for (int i = 0; i < numBatch; ++i) {
        batchSize *= lhsT.shape(i);
    }

    int64_t lhsFreeSize = 1;
    for (int i = numBatch; i < numBatch + numLhsFree; ++i) {
        lhsFreeSize *= lhsT.shape(i);
    }

    int64_t contractSize = 1;
    for (int i = numBatch + numLhsFree; i < static_cast<int>(lhsT.ndim()); ++i) {
        contractSize *= lhsT.shape(i);
    }

    int64_t rhsFreeSize = 1;
    for (int i = numBatch + numContract; i < static_cast<int>(rhsT.ndim()); ++i) {
        rhsFreeSize *= rhsT.shape(i);
    }

    // Save original batch/free shapes for final reshape
    std::vector<int> batchShape;
    batchShape.reserve(numBatch);
    for (int i = 0; i < numBatch; ++i) {
        batchShape.push_back(lhsT.shape(i));
    }
    std::vector<int> lhsFreeShape;
    lhsFreeShape.reserve(numLhsFree);
    for (int i = numBatch; i < numBatch + numLhsFree; ++i) {
        lhsFreeShape.push_back(lhsT.shape(i));
    }
    std::vector<int> rhsFreeShape;
    rhsFreeShape.reserve(static_cast<int>(rhsT.ndim()) - numBatch - numContract);
    for (int i = numBatch + numContract; i < static_cast<int>(rhsT.ndim()); ++i) {
        rhsFreeShape.push_back(rhsT.shape(i));
    }

    // Reshape to 3D for matmul: [batch, M, K] @ [batch, K, N]
    mlx::core::Shape lhsShape3d = {static_cast<int>(batchSize), static_cast<int>(lhsFreeSize),
                                   static_cast<int>(contractSize)};
    mlx::core::Shape rhsShape3d = {static_cast<int>(batchSize), static_cast<int>(contractSize),
                                   static_cast<int>(rhsFreeSize)};

    auto lhs3d = mlx::core::reshape(lhsT, lhsShape3d);
    auto rhs3d = mlx::core::reshape(rhsT, rhsShape3d);

    // Perform batched matmul
    auto result3d = mlx::core::matmul(lhs3d, rhs3d);

    // Reshape back to [batch..., lhsFree..., rhsFree...]
    mlx::core::Shape finalShape;
    for (int s : batchShape)
        finalShape.push_back(s);
    for (int s : lhsFreeShape)
        finalShape.push_back(s);
    for (int s : rhsFreeShape)
        finalShape.push_back(s);

    auto result = mlx::core::reshape(result3d, finalShape);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.convolution
bool HandleConvolution(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
    if (!convOp) {
        MPS_LOG_ERROR("stablehlo.convolution: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    auto kernel_opt = GetValue(values, op->getOperand(1));
    if (!input_opt || !kernel_opt) {
        MPS_LOG_ERROR("stablehlo.convolution: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto& kernel = kernel_opt->get();
    auto dimNumbers = convOp.getDimensionNumbers();

    // Get dimension mappings
    int64_t inputBatchDim = dimNumbers.getInputBatchDimension();
    int64_t inputFeatureDim = dimNumbers.getInputFeatureDimension();
    auto inputSpatialDims = dimNumbers.getInputSpatialDimensions();

    int64_t kernelInputFeatureDim = dimNumbers.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDim = dimNumbers.getKernelOutputFeatureDimension();
    auto kernelSpatialDims = dimNumbers.getKernelSpatialDimensions();

    int64_t outputBatchDim = dimNumbers.getOutputBatchDimension();
    int64_t outputFeatureDim = dimNumbers.getOutputFeatureDimension();
    auto outputSpatialDims = dimNumbers.getOutputSpatialDimensions();

    int numSpatialDims = static_cast<int>(inputSpatialDims.size());

    // MLX conv_general expects:
    // Input: [N, spatial..., C_in] (NHWC for 2D, NWC for 1D)
    // Weight: [C_out, spatial..., C_in] (OHWI for 2D)
    // Output: [N, spatial..., C_out]

    // Build input permutation to [N, spatial..., C_in] format
    std::vector<int> inputPerm(input.ndim());
    inputPerm[0] = static_cast<int>(inputBatchDim);
    for (int i = 0; i < numSpatialDims; ++i) {
        inputPerm[1 + i] = static_cast<int>(inputSpatialDims[i]);
    }
    inputPerm[1 + numSpatialDims] = static_cast<int>(inputFeatureDim);

    // Build kernel permutation to [C_out, spatial..., C_in] format
    // MLX weight format: (C_out, spatial..., C_in)
    std::vector<int> kernelPerm(kernel.ndim());
    kernelPerm[0] = static_cast<int>(kernelOutputFeatureDim);
    for (int i = 0; i < numSpatialDims; ++i) {
        kernelPerm[1 + i] = static_cast<int>(kernelSpatialDims[i]);
    }
    kernelPerm[1 + numSpatialDims] = static_cast<int>(kernelInputFeatureDim);

    // Transpose input and kernel if needed
    // Note: JAX uses HWIO kernel layout, MLX expects OHWI, so kernel transpose is typically needed
    auto inputT = IsIdentityPermutation(inputPerm) ? input : mlx::core::transpose(input, inputPerm);
    auto kernelT =
        IsIdentityPermutation(kernelPerm) ? kernel : mlx::core::transpose(kernel, kernelPerm);

    // Extract strides
    std::vector<int> strides;
    if (auto stridesAttr = convOp.getWindowStrides()) {
        for (int64_t s : *stridesAttr) {
            strides.push_back(static_cast<int>(s));
        }
    } else {
        strides.resize(numSpatialDims, 1);
    }

    // Extract padding
    std::vector<int> paddingLow(numSpatialDims, 0);
    std::vector<int> paddingHigh(numSpatialDims, 0);
    if (auto paddingAttr = convOp.getPadding()) {
        auto paddingValues = paddingAttr->getValues<int64_t>();
        auto it = paddingValues.begin();
        for (int i = 0; i < numSpatialDims; ++i) {
            paddingLow[i] = static_cast<int>(*it++);
            paddingHigh[i] = static_cast<int>(*it++);
        }
    }

    // Extract dilation
    std::vector<int> kernelDilation;
    if (auto dilationAttr = convOp.getRhsDilation()) {
        for (int64_t d : *dilationAttr) {
            kernelDilation.push_back(static_cast<int>(d));
        }
    } else {
        kernelDilation.resize(numSpatialDims, 1);
    }

    std::vector<int> inputDilation;
    if (auto dilationAttr = convOp.getLhsDilation()) {
        for (int64_t d : *dilationAttr) {
            inputDilation.push_back(static_cast<int>(d));
        }
    } else {
        inputDilation.resize(numSpatialDims, 1);
    }

    // Handle feature_group_count for grouped/depthwise convolutions
    auto featureGroupCount = static_cast<int>(convOp.getFeatureGroupCount());

    // batch_group_count is used for specialized grouped convolutions; not currently supported
    auto batchGroupCount = convOp.getBatchGroupCount();
    if (batchGroupCount != 1) {
        MPS_LOG_ERROR("stablehlo.convolution: batch_group_count != 1 not supported\n");
        return false;
    }

    // Call MLX conv_general
    auto convResult =
        mlx::core::conv_general(inputT, kernelT, strides, paddingLow, paddingHigh, kernelDilation,
                                inputDilation, featureGroupCount, false);

    // Build output permutation to convert from MLX format [N, spatial..., C_out] to StableHLO
    // format. outputPerm[i] = j means: position i in output gets data from MLX position j
    std::vector<int> outputPerm(convResult.ndim());
    outputPerm[outputBatchDim] = 0;
    for (int i = 0; i < numSpatialDims; ++i) {
        outputPerm[outputSpatialDims[i]] = 1 + i;
    }
    outputPerm[outputFeatureDim] = 1 + numSpatialDims;

    auto result = IsIdentityPermutation(outputPerm) ? convResult
                                                    : mlx::core::transpose(convResult, outputPerm);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.custom_call
bool HandleCustomCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        MPS_LOG_ERROR("stablehlo.custom_call: failed to cast\n");
        return false;
    }

    auto callTargetName = customCallOp.getCallTargetName().str();

    // Handle Sharding annotation - just pass input through
    if (callTargetName == "Sharding") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: expected 1 input and 1 output\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: operand not found\n");
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    // Handle SPMDFullToShardShape and SPMDShardToFullShape - also pass-through for single device
    if (callTargetName == "SPMDFullToShardShape" || callTargetName == "SPMDShardToFullShape") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: operand not found\n", callTargetName.c_str());
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    // Handle mhlo.erf - error function
    if (callTargetName == "mhlo.erf") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.erf: expected 1 input and 1 output\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.erf: operand not found\n");
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), mlx::core::erf(input_opt->get()));
        return true;
    }

    // Handle unary mhlo.* custom calls
    using UnaryFn = mlx::core::array (*)(const mlx::core::array&, mlx::core::StreamOrDevice);
    static const std::unordered_map<std::string, UnaryFn> unaryCustomCalls = {
        {"mhlo.sinh", mlx::core::sinh},      {"mhlo.cosh", mlx::core::cosh},
        {"mhlo.asin", mlx::core::arcsin},    {"mhlo.acos", mlx::core::arccos},
        {"mhlo.atan", mlx::core::arctan},    {"mhlo.asinh", mlx::core::arcsinh},
        {"mhlo.acosh", mlx::core::arccosh},  {"mhlo.atanh", mlx::core::arctanh},
        {"mhlo.erf_inv", mlx::core::erfinv},
    };

    auto unaryIt = unaryCustomCalls.find(callTargetName);
    if (unaryIt != unaryCustomCalls.end()) {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: operand not found\n", callTargetName.c_str());
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), unaryIt->second(input_opt->get(), {}));
        return true;
    }

    // Handle mhlo.topk - returns top k values and their indices (sorted descending)
    if (callTargetName == "mhlo.topk") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.topk: expected 1 input and 2 outputs\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.topk: operand not found\n");
            return false;
        }

        // Get k from the output shape (last dimension of result 0)
        auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        int k = static_cast<int>(resultType.getShape().back());
        int axis = static_cast<int>(input_opt->get().ndim()) - 1;  // topk always on last axis

        // Negate + argsort for descending order, then take first k
        auto input = mlx::core::contiguous(input_opt->get());
        auto sortedIndices = mlx::core::argsort(mlx::core::negative(input), axis);

        // Slice first k along the axis
        mlx::core::Shape starts(sortedIndices.ndim(), 0);
        mlx::core::Shape stops(sortedIndices.shape().begin(), sortedIndices.shape().end());
        stops[axis] = k;
        auto indices = mlx::core::slice(sortedIndices, starts, stops);

        auto topValues = mlx::core::take_along_axis(input, indices, axis);
        values.emplace(ToKey(op->getResult(0)), std::move(topValues));
        values.emplace(ToKey(op->getResult(1)), mlx::core::astype(indices, mlx::core::int32));
        return true;
    }

    MPS_LOG_ERROR("stablehlo.custom_call: unsupported target '%s'\n", callTargetName.c_str());
    return false;
}

// Handler for stablehlo.optimization_barrier
// This op is an identity — it passes operands through unchanged. Its purpose is to
// prevent optimization passes from reordering ops across it. At runtime, it's a no-op.
// Generated by jax.checkpoint/jax.remat for gradient rematerialization.
bool HandleOptimizationBarrier(mlir::Operation* op, ValueMap& values,
                               std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto operand_opt = GetValue(values, op->getOperand(i));
        if (!operand_opt) {
            MPS_LOG_ERROR("stablehlo.optimization_barrier: operand %u not found\n", i);
            return false;
        }
        values.emplace(ToKey(op->getResult(i)), operand_opt->get());
    }
    return true;
}

// Handler for func.call
bool HandleCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!callOp) {
        MPS_LOG_ERROR("func.call: failed to cast\n");
        return false;
    }

    // Find the callee function in the module
    auto calleeName = callOp.getCallee();
    auto calleeFunc = ctx.module.lookupSymbol<mlir::func::FuncOp>(calleeName);
    if (!calleeFunc) {
        MPS_LOG_ERROR("func.call: callee function '%s' not found\n", calleeName.str().c_str());
        return false;
    }

    // Gather input arrays
    std::vector<mlx::core::array> callInputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("func.call: operand not found in value map\n");
            return false;
        }
        callInputs.push_back(val_opt->get());
    }

    // Execute the callee function
    std::vector<mlx::core::array> callOutputs;
    if (!ExecuteFunction(calleeFunc, callInputs, callOutputs, ctx)) {
        MPS_LOG_ERROR("func.call: failed to execute callee '%s'\n", calleeName.str().c_str());
        return false;
    }

    // Map results to the call operation's results
    if (callOutputs.size() != op->getNumResults()) {
        MPS_LOG_ERROR("func.call: result count mismatch\n");
        return false;
    }

    for (size_t i = 0; i < callOutputs.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(callOutputs[i]));
    }

    return true;
}

// Handler for stablehlo.sort
bool HandleSort(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        MPS_LOG_ERROR("stablehlo.sort: failed to cast\n");
        return false;
    }

    int dimension = static_cast<int>(sortOp.getDimension());
    bool isStable = sortOp.getIsStable();
    (void)isStable;  // MLX sort is always stable

    // Analyze comparator to determine sort direction
    // The comparator takes pairs of elements and returns bool
    // We look for compare LT (ascending) or GT (descending)
    bool ascending = true;
    auto& comparator = sortOp.getComparator();
    if (!comparator.empty()) {
        auto& block = comparator.front();
        for (auto& compOp : block.getOperations()) {
            if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(compOp)) {
                auto dir = cmpOp.getComparisonDirection();
                if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                    dir == mlir::stablehlo::ComparisonDirection::GE) {
                    ascending = false;
                }
                break;  // Use the last compare before return
            }
        }
    }

    size_t numInputs = sortOp.getInputs().size();

    if (numInputs == 1) {
        // Simple sort of a single tensor
        auto input_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.sort: input not found\n");
            return false;
        }
        auto result = mlx::core::sort(input_opt->get(), dimension);
        if (!ascending) {
            // Reverse the sorted dimension
            auto shape = result.shape();
            int dimSize = shape[dimension];
            mlx::core::Shape starts(result.ndim(), 0);
            mlx::core::Shape stops(shape.begin(), shape.end());
            mlx::core::Shape steps(result.ndim(), 1);
            starts[dimension] = dimSize - 1;
            stops[dimension] = -dimSize - 1;
            steps[dimension] = -1;
            result = mlx::core::slice(result, starts, stops, steps);
        }
        values.emplace(ToKey(op->getResult(0)), std::move(result));
    } else {
        // Sort-by-key: sort first input, apply same permutation to all others
        auto keys_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!keys_opt) {
            MPS_LOG_ERROR("stablehlo.sort: keys not found\n");
            return false;
        }

        // For descending sort, negate keys so ascending argsort gives descending order
        auto sortKeys = ascending ? keys_opt->get() : mlx::core::negative(keys_opt->get());
        auto indices = mlx::core::argsort(sortKeys, dimension);

        // Apply permutation to all inputs
        for (size_t i = 0; i < numInputs; ++i) {
            auto input_opt = GetValue(values, sortOp.getInputs()[i]);
            if (!input_opt) {
                MPS_LOG_ERROR("stablehlo.sort: input %zu not found\n", i);
                return false;
            }
            auto sorted = mlx::core::take_along_axis(input_opt->get(), indices, dimension);
            values.emplace(ToKey(op->getResult(i)), std::move(sorted));
        }
    }

    return true;
}

// Handler for stablehlo.scatter
bool HandleScatter(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
    if (!scatterOp) {
        MPS_LOG_ERROR("stablehlo.scatter: failed to cast\n");
        return false;
    }

    // Only support single-input scatter for now
    if (scatterOp.getInputs().size() != 1) {
        MPS_LOG_ERROR("stablehlo.scatter: multi-input scatter not supported\n");
        return false;
    }

    auto operand_opt = GetValue(values, scatterOp.getInputs()[0]);
    auto indices_opt = GetValue(values, scatterOp.getScatterIndices());
    auto updates_opt = GetValue(values, scatterOp.getUpdates()[0]);
    if (!operand_opt || !indices_opt || !updates_opt) {
        MPS_LOG_ERROR("stablehlo.scatter: operand not found in value map\n");
        return false;
    }

    auto& operand = operand_opt->get();
    auto& scatterIndices = indices_opt->get();
    auto& updates = updates_opt->get();

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto indexVectorDim = static_cast<int>(dimNumbers.getIndexVectorDim());

    // Detect update type
    auto& body = scatterOp.getUpdateComputation();
    auto scatterType = DetectScatterType(body);

    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();

    // Batched scatter: each batch element scatters independently.
    // We use multi-axis scatter with iota indices for batch dims.
    if (!inputBatchingDims.empty()) {
        // Squeeze index_vector_dim if size 1
        auto indices = scatterIndices;
        if (indexVectorDim < static_cast<int>(indices.shape().size()) &&
            indices.shape(indexVectorDim) == 1) {
            indices = mlx::core::squeeze(indices, {indexVectorDim});
        }
        if (indices.dtype() != mlx::core::int32) {
            indices = mlx::core::astype(indices, mlx::core::int32);
        }

        // Build axes and index arrays: scatter dims + batch dims
        std::vector<int> axes;
        std::vector<mlx::core::array> idxVec;

        // Add scatter dim indices
        for (auto dim : scatterDimsToOperandDims) {
            axes.push_back(static_cast<int>(dim));
        }
        idxVec.push_back(indices);

        // Add batch dim iota indices
        for (size_t i = 0; i < inputBatchingDims.size(); ++i) {
            int batchAxis = static_cast<int>(inputBatchingDims[i]);
            axes.push_back(batchAxis);

            // Create iota for this batch dimension, broadcast to indices shape
            int batchSize = operand.shape(batchAxis);
            mlx::core::Shape iotaShape(indices.ndim(), 1);

            // Find which dim in indices corresponds to this batch dim
            int idxBatchDim = static_cast<int>(scatterIndicesBatchingDims[i]);
            // Adjust for squeezed index_vector_dim
            if (indexVectorDim < idxBatchDim) {
                --idxBatchDim;
            }
            iotaShape[idxBatchDim] = batchSize;

            auto batchIota = mlx::core::reshape(mlx::core::arange(batchSize), iotaShape);
            batchIota = mlx::core::broadcast_to(batchIota, indices.shape());
            idxVec.push_back(batchIota);
        }

        // Reshape updates: (idx_shape..., operand_shape_with_1_at_covered_dims...)
        // Covered dims = scatter dims + batch dims + inserted window dims
        // Updates dims are split into index dims (not in update_window_dims)
        // and window dims (in update_window_dims).
        auto updateWindowDims = dimNumbers.getUpdateWindowDims();
        std::set<int> coveredDims;
        for (auto d : scatterDimsToOperandDims) {
            coveredDims.insert(static_cast<int>(d));
        }
        for (auto d : inputBatchingDims) {
            coveredDims.insert(static_cast<int>(d));
        }
        for (auto d : insertedWindowDims) {
            coveredDims.insert(static_cast<int>(d));
        }

        auto updShape = updates.shape();
        int updNdim = static_cast<int>(updShape.size());
        std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());

        // Separate index dims and window dims, transpose if needed
        std::vector<int> transposeOrder;
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) == 0) {
                transposeOrder.push_back(i);
            }
        }
        int numIdxDims = static_cast<int>(transposeOrder.size());
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) != 0) {
                transposeOrder.push_back(i);
            }
        }
        auto transposedUpdates = updates;
        bool needsTranspose = false;
        for (int i = 0; i < updNdim; ++i) {
            if (transposeOrder[i] != i) {
                needsTranspose = true;
                break;
            }
        }
        if (needsTranspose) {
            transposedUpdates = mlx::core::transpose(updates, transposeOrder);
        }
        auto tShape = transposedUpdates.shape();

        // Build final shape: (idx_dims..., operand_dim_0, ...)
        mlx::core::Shape newShape;
        for (int i = 0; i < numIdxDims; ++i) {
            newShape.push_back(tShape[i]);
        }
        int windowIdx = 0;
        for (int dim = 0; dim < static_cast<int>(operand.ndim()); ++dim) {
            if (coveredDims.count(dim) != 0) {
                newShape.push_back(1);
            } else {
                newShape.push_back(tShape[numIdxDims + windowIdx]);
                ++windowIdx;
            }
        }
        auto reshapedUpdates = mlx::core::reshape(transposedUpdates, newShape);

        mlx::core::array result = operand;
        switch (scatterType) {
            case ScatterType::Update:
                result = mlx::core::scatter(operand, idxVec, reshapedUpdates, axes);
                break;
            case ScatterType::Add:
                result = mlx::core::scatter_add(operand, idxVec, reshapedUpdates, axes);
                break;
            case ScatterType::Sub:
                result = mlx::core::scatter_add(operand, idxVec,
                                                mlx::core::negative(reshapedUpdates), axes);
                break;
            case ScatterType::Mul:
                result = mlx::core::scatter_prod(operand, idxVec, reshapedUpdates, axes);
                break;
            case ScatterType::Min:
                result = mlx::core::scatter_min(operand, idxVec, reshapedUpdates, axes);
                break;
            case ScatterType::Max:
                result = mlx::core::scatter_max(operand, idxVec, reshapedUpdates, axes);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter update type (batched)\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Simple case: 1D scatter with single scatter dim
    if (scatterDimsToOperandDims.size() == 1 && insertedWindowDims.size() == 1) {
        int scatterDim = static_cast<int>(scatterDimsToOperandDims[0]);

        // Extract indices
        auto indices = scatterIndices;
        if (indexVectorDim < static_cast<int>(scatterIndices.shape().size()) &&
            scatterIndices.shape(indexVectorDim) == 1) {
            indices = mlx::core::squeeze(scatterIndices, {indexVectorDim});
        }

        // Ensure indices are int32 and at least 1D
        if (indices.dtype() != mlx::core::int32) {
            indices = mlx::core::astype(indices, mlx::core::int32);
        }
        if (indices.ndim() == 0) {
            indices = mlx::core::reshape(indices, {1});
        }

        // MLX scatter expects updates with shape [idx_shape..., slice_shape...]
        // where slice_shape includes all operand dims.
        // StableHLO updates have inserted_window_dims collapsed, so we need
        // to expand those dims back (insert size-1 dims).
        auto reshapedUpdates = updates;
        auto updateWindowDims = dimNumbers.getUpdateWindowDims();
        int insertedDim = static_cast<int>(insertedWindowDims[0]);

        // MLX scatter expects updates with shape [idx_shape..., slice_shape_per_operand_dim...]
        // where slice_shape has size-1 at scatter axes and original sizes elsewhere.
        // StableHLO updates have inserted_window_dims collapsed.
        // We need to expand updates to include idx_shape prefix and size-1 at inserted dims.
        {
            auto updShape = updates.shape();
            mlx::core::Shape newShape;

            // Batch dims come from indices shape
            auto idxShape = indices.shape();
            for (int dim : idxShape) {
                newShape.push_back(dim);
            }

            // Then operand dims, with size-1 at inserted positions
            if (updateWindowDims.empty()) {
                // All operand dims are inserted
                for (int i = 0; i < static_cast<int>(operand.ndim()); ++i) {
                    newShape.push_back(1);
                }
            } else {
                int windowIdx = static_cast<int>(updateWindowDims[0]);
                for (int operandDim = 0; operandDim < static_cast<int>(operand.ndim());
                     ++operandDim) {
                    if (operandDim == insertedDim) {
                        newShape.push_back(1);
                    } else {
                        newShape.push_back(updShape[windowIdx]);
                        ++windowIdx;
                    }
                }
            }
            reshapedUpdates = mlx::core::reshape(updates, newShape);
        }

        mlx::core::array result = operand;
        std::vector<mlx::core::array> idxVec = {indices};
        std::vector<int> axesVec = {scatterDim};

        switch (scatterType) {
            case ScatterType::Update:
                result = mlx::core::scatter(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Add:
                result = mlx::core::scatter_add(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Sub:
                result = mlx::core::scatter_add(operand, idxVec,
                                                mlx::core::negative(reshapedUpdates), axesVec);
                break;
            case ScatterType::Mul:
                result = mlx::core::scatter_prod(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Min:
                result = mlx::core::scatter_min(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Max:
                result = mlx::core::scatter_max(operand, idxVec, reshapedUpdates, axesVec);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter update type\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Multi-dimensional scatter: inserted_window_dims == scatter_dims_to_operand_dims
    // Handles both full index scatter (all operand dims are scatter targets)
    // and partial index scatter (some operand dims are window dims).
    if (scatterDimsToOperandDims.size() == insertedWindowDims.size()) {
        auto indices = scatterIndices;
        auto updateWindowDims = dimNumbers.getUpdateWindowDims();

        // Build per-axis index arrays
        std::vector<mlx::core::array> idxVec;
        std::vector<int> axesVec;
        for (size_t i = 0; i < scatterDimsToOperandDims.size(); ++i) {
            int axis = static_cast<int>(scatterDimsToOperandDims[i]);
            axesVec.push_back(axis);

            // Slice the index vector dim to get indices for this axis
            mlx::core::Shape starts(indices.ndim(), 0);
            mlx::core::Shape stops(indices.shape().begin(), indices.shape().end());
            starts[indexVectorDim] = static_cast<int>(i);
            stops[indexVectorDim] = static_cast<int>(i) + 1;
            auto axisIndices =
                mlx::core::squeeze(mlx::core::slice(indices, starts, stops), {indexVectorDim});
            if (axisIndices.dtype() != mlx::core::int32) {
                axisIndices = mlx::core::astype(axisIndices, mlx::core::int32);
            }
            idxVec.push_back(axisIndices);
        }

        // Reshape updates for MLX: (idx_shape..., operand_dims...)
        // where scatter target dims get size 1 and window dims keep their size.
        //
        // StableHLO updates dims are split into:
        //   - index dims: all dims NOT in update_window_dims
        //   - window dims: dims IN update_window_dims (correspond to non-inserted operand dims)
        // These can be interleaved, so we first transpose to put index dims first.
        std::set<int> insertedSet(insertedWindowDims.begin(), insertedWindowDims.end());
        std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());
        auto updShape = updates.shape();
        int updNdim = static_cast<int>(updShape.size());

        // Separate index dims and window dims in updates
        std::vector<int> transposeOrder;
        // Index dims first
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) == 0) {
                transposeOrder.push_back(i);
            }
        }
        int numIdxDims = static_cast<int>(transposeOrder.size());
        // Window dims second
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) != 0) {
                transposeOrder.push_back(i);
            }
        }

        // Transpose if needed
        auto transposedUpdates = updates;
        bool needsTranspose = false;
        for (int i = 0; i < updNdim; ++i) {
            if (transposeOrder[i] != i) {
                needsTranspose = true;
                break;
            }
        }
        if (needsTranspose) {
            transposedUpdates = mlx::core::transpose(updates, transposeOrder);
        }
        auto tShape = transposedUpdates.shape();

        // Build final shape: (idx_dims..., operand_dim_0, operand_dim_1, ...)
        mlx::core::Shape newShape;
        for (int i = 0; i < numIdxDims; ++i) {
            newShape.push_back(tShape[i]);
        }
        int windowIdx = 0;
        for (int operandDim = 0; operandDim < static_cast<int>(operand.ndim()); ++operandDim) {
            if (insertedSet.count(operandDim) != 0) {
                newShape.push_back(1);
            } else {
                newShape.push_back(tShape[numIdxDims + windowIdx]);
                ++windowIdx;
            }
        }
        auto reshapedUpdates = mlx::core::reshape(transposedUpdates, newShape);

        mlx::core::array result = operand;
        switch (scatterType) {
            case ScatterType::Update:
                result = mlx::core::scatter(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Add:
                result = mlx::core::scatter_add(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Sub:
                result = mlx::core::scatter_add(operand, idxVec,
                                                mlx::core::negative(reshapedUpdates), axesVec);
                break;
            case ScatterType::Mul:
                result = mlx::core::scatter_prod(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Min:
                result = mlx::core::scatter_min(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Max:
                result = mlx::core::scatter_max(operand, idxVec, reshapedUpdates, axesVec);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter update type\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Window scatter: scatter dim has window extent > 1 (not in insertedWindowDims).
    // We expand start indices to per-element indices within the window.
    if (scatterDimsToOperandDims.size() == 1 && insertedWindowDims.empty()) {
        int scatterDim = static_cast<int>(scatterDimsToOperandDims[0]);

        auto indices = scatterIndices;
        if (indexVectorDim < static_cast<int>(indices.shape().size()) &&
            indices.shape(indexVectorDim) == 1) {
            indices = mlx::core::squeeze(indices, {indexVectorDim});
        }
        if (indices.dtype() != mlx::core::int32) {
            indices = mlx::core::astype(indices, mlx::core::int32);
        }
        if (indices.ndim() == 0) {
            indices = mlx::core::reshape(indices, {1});
        }

        auto updateWindowDims = dimNumbers.getUpdateWindowDims();

        // Find the update dim that corresponds to the scatter operand dim.
        // updateWindowDims maps (in order) to operand dims not in insertedWindowDims.
        // Since insertedWindowDims is empty, operand dim i maps to updateWindowDims[i].
        int windowDimForScatterAxis = static_cast<int>(updateWindowDims[scatterDim]);
        int windowSize = static_cast<int>(updates.shape(windowDimForScatterAxis));

        // Expand start indices: for each start index, generate start + arange(windowSize)
        auto offsets = mlx::core::arange(windowSize, mlx::core::int32);

        // indices shape: (num_starts,...), offsets shape: (windowSize,)
        // Broadcast: indices[..., None] + offsets -> (..., windowSize)
        auto idxShape = indices.shape();
        mlx::core::Shape broadcastIdxShape = idxShape;
        broadcastIdxShape.push_back(1);
        auto reshapedIdx = mlx::core::reshape(indices, broadcastIdxShape);

        mlx::core::Shape broadcastOffsetShape(indices.ndim(), 1);
        broadcastOffsetShape.push_back(windowSize);
        auto reshapedOffsets = mlx::core::reshape(offsets, broadcastOffsetShape);

        auto expandedIndices = mlx::core::add(reshapedIdx, reshapedOffsets);
        // Flatten to 1D
        int totalIndices = 1;
        for (auto s : expandedIndices.shape())
            totalIndices *= static_cast<int>(s);
        expandedIndices = mlx::core::reshape(expandedIndices, {totalIndices});

        // Reshape updates for MLX: move scatter-axis window elements into idx_shape,
        // insert size-1 at the scatter axis position in slice_shape.
        std::set<int> windowDimSet(updateWindowDims.begin(), updateWindowDims.end());
        auto updShape = updates.shape();
        int updNdim = static_cast<int>(updShape.size());

        // Transpose: index dims first, then scatter-axis window dim, then other window dims
        std::vector<int> transposeOrder;
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) == 0)
                transposeOrder.push_back(i);
        }
        transposeOrder.push_back(windowDimForScatterAxis);
        for (int i = 0; i < updNdim; ++i) {
            if (windowDimSet.count(i) != 0 && i != windowDimForScatterAxis)
                transposeOrder.push_back(i);
        }

        auto transposedUpdates = updates;
        bool needsTranspose = false;
        for (int i = 0; i < updNdim; ++i) {
            if (transposeOrder[i] != i) {
                needsTranspose = true;
                break;
            }
        }
        if (needsTranspose) {
            transposedUpdates = mlx::core::transpose(updates, transposeOrder);
        }
        auto tShape = transposedUpdates.shape();

        // tShape: (idx_dims..., scatterWindowSize, other_window_dims...)
        int numIdxDims = updNdim - static_cast<int>(updateWindowDims.size());
        // Merge idx dims and scatter window dim into one flat idx dim
        int flatIdxSize = 1;
        for (int i = 0; i <= numIdxDims; ++i) {
            flatIdxSize *= static_cast<int>(tShape[i]);
        }

        // Build: (flatIdxSize, 1_at_scatter_axis, other_operand_dims...)
        mlx::core::Shape newShape;
        newShape.push_back(flatIdxSize);
        int otherWindowIdx = numIdxDims + 1;
        for (int operandDim = 0; operandDim < static_cast<int>(operand.ndim()); ++operandDim) {
            if (operandDim == scatterDim) {
                newShape.push_back(1);
            } else {
                newShape.push_back(static_cast<int>(tShape[otherWindowIdx]));
                ++otherWindowIdx;
            }
        }
        auto reshapedUpdates = mlx::core::reshape(transposedUpdates, newShape);

        std::vector<mlx::core::array> idxVec = {expandedIndices};
        std::vector<int> axesVec = {scatterDim};

        mlx::core::array result = operand;
        switch (scatterType) {
            case ScatterType::Update:
                result = mlx::core::scatter(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Add:
                result = mlx::core::scatter_add(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Sub:
                result = mlx::core::scatter_add(operand, idxVec,
                                                mlx::core::negative(reshapedUpdates), axesVec);
                break;
            case ScatterType::Mul:
                result = mlx::core::scatter_prod(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Min:
                result = mlx::core::scatter_min(operand, idxVec, reshapedUpdates, axesVec);
                break;
            case ScatterType::Max:
                result = mlx::core::scatter_max(operand, idxVec, reshapedUpdates, axesVec);
                break;
            default:
                MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter update type (window)\n");
                return false;
        }

        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    MPS_LOG_ERROR("stablehlo.scatter: unsupported scatter pattern "
                  "(scatterDimsToOperandDims.size=%zu, insertedWindowDims.size=%zu)\n",
                  scatterDimsToOperandDims.size(), insertedWindowDims.size());
    return false;
}

// Handler for stablehlo.fft
bool HandleFft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
    if (!fftOp) {
        MPS_LOG_ERROR("stablehlo.fft: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.fft: operand not found\n");
        return false;
    }

    auto& input = input_opt->get();
    auto fftType = fftOp.getFftType();
    auto fftLength = fftOp.getFftLength();

    // Convert fft_length to vector
    std::vector<int> axes;
    mlx::core::Shape lengths;
    int ndim = static_cast<int>(input.ndim());
    for (size_t i = 0; i < fftLength.size(); ++i) {
        axes.push_back(ndim - static_cast<int>(fftLength.size()) + static_cast<int>(i));
        lengths.push_back(static_cast<int>(fftLength[i]));
    }

    mlx::core::array result = input;
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            result = mlx::core::fft::fftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IFFT:
            result = mlx::core::fft::ifftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::RFFT:
            result = mlx::core::fft::rfftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IRFFT:
            result = mlx::core::fft::irfftn(input, lengths, axes);
            break;
        default:
            MPS_LOG_ERROR("stablehlo.fft: unsupported fft type\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.complex (combine real + imag into complex)
bool HandleComplex(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto real_opt = GetValue(values, op->getOperand(0));
    auto imag_opt = GetValue(values, op->getOperand(1));
    if (!real_opt || !imag_opt) {
        MPS_LOG_ERROR("stablehlo.complex: operand not found\n");
        return false;
    }
    // MLX: create complex from real and imaginary parts
    // complex(re, im) = re + im * 1j
    auto imag_unit = mlx::core::array(std::complex<float>(0.0F, 1.0F));
    auto result = mlx::core::add(
        mlx::core::astype(real_opt->get(), mlx::core::complex64),
        mlx::core::multiply(mlx::core::astype(imag_opt->get(), mlx::core::complex64), imag_unit));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.real
bool HandleReal(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.real: operand not found\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::real(input_opt->get()));
    return true;
}

// Handler for stablehlo.imag
bool HandleImag(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.imag: operand not found\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::imag(input_opt->get()));
    return true;
}

// Handler for stablehlo.cholesky
bool HandleCholesky(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto cholOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!cholOp) {
        MPS_LOG_ERROR("stablehlo.cholesky: failed to cast\n");
        return false;
    }
    auto input_opt = GetValue(values, cholOp.getA());
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cholesky: operand not found in value map\n");
        return false;
    }
    auto& a = input_opt->get();
    bool lower = cholOp.getLower();
    // MLX cholesky uses upper=!lower convention
    auto result = mlx::core::linalg::cholesky(a, /*upper=*/!lower);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.triangular_solve
bool HandleTriangularSolve(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: failed to cast\n");
        return false;
    }
    auto a_opt = GetValue(values, triSolveOp.getA());
    auto b_opt = GetValue(values, triSolveOp.getB());
    if (!a_opt || !b_opt) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: operand not found in value map\n");
        return false;
    }
    auto& a = a_opt->get();
    auto& b = b_opt->get();

    bool left_side = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unit_diagonal = triSolveOp.getUnitDiagonal();
    auto transpose = triSolveOp.getTransposeA();

    auto a_solve = a;
    auto b_solve = b;

    if (unit_diagonal) {
        // Replace diagonal with 1s: zero out diagonal, add identity.
        int n = a_solve.shape()[a_solve.ndim() - 1];
        auto eye = mlx::core::eye(n, a_solve.dtype());
        a_solve = a_solve * (1.0F - eye) + eye;
    }

    // Guard against singular triangular matrices to prevent LAPACK abort().
    // Done after unit_diagonal handling so unit-diagonal matrices are never flagged.
    // Detect zero diagonals per-batch, replace them with epsilon to allow solve,
    // then use where() to NaN-out results for singular batches. This is purely
    // functional (no eval) so it works inside mlx::core::compile() tracing.
    int n_dim = a_solve.shape()[a_solve.ndim() - 1];
    auto diag = mlx::core::diagonal(a_solve, 0, -2, -1);
    auto zero_mask = mlx::core::equal(diag, mlx::core::zeros_like(diag));
    // per-batch: any zero on diagonal? Shape: (*batch_dims)
    auto batch_singular = mlx::core::any(zero_mask, /* axis= */ std::vector<int>{-1},
                                         /* keepdims= */ false);

    // Replace zero diagonal entries with epsilon so LAPACK won't abort
    auto eye_mat = mlx::core::eye(n_dim, a_solve.dtype());
    auto eps = mlx::core::array(1e-30F, a_solve.dtype());
    a_solve = a_solve + mlx::core::expand_dims(zero_mask, -1) * eye_mat * eps;

    if (transpose == mlir::stablehlo::Transpose::TRANSPOSE ||
        transpose == mlir::stablehlo::Transpose::ADJOINT) {
        // Transpose A: swap last two dimensions.
        auto ndim = a_solve.ndim();
        std::vector<int> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[ndim - 2], perm[ndim - 1]);
        a_solve = mlx::core::transpose(a_solve, perm);
        // Transposing flips lower/upper
        lower = !lower;
    }

    // Helper to replace results with NaN for singular batches
    auto nan_guard = [&](mlx::core::array result) -> mlx::core::array {
        auto nan_val = mlx::core::full(result.shape(), std::numeric_limits<float>::quiet_NaN(),
                                       result.dtype());
        // Broadcast batch_singular to result shape: expand dims for matrix dims
        auto mask = batch_singular;
        for (int i = 0; i < 2; ++i) {
            mask = mlx::core::expand_dims(mask, -1);
        }
        return mlx::core::where(mask, nan_val, result);
    };

    try {
        if (!left_side) {
            // Right-side solve: X * A = B
            // Equivalent to: A^T * X^T = B^T (left-side solve)
            auto ndim_a = a_solve.ndim();
            std::vector<int> perm_a(ndim_a);
            std::iota(perm_a.begin(), perm_a.end(), 0);
            std::swap(perm_a[ndim_a - 2], perm_a[ndim_a - 1]);
            a_solve = mlx::core::transpose(a_solve, perm_a);
            lower = !lower;

            auto ndim_b = b_solve.ndim();
            std::vector<int> perm_b(ndim_b);
            std::iota(perm_b.begin(), perm_b.end(), 0);
            std::swap(perm_b[ndim_b - 2], perm_b[ndim_b - 1]);
            b_solve = mlx::core::transpose(b_solve, perm_b);

            auto x_t = mlx::core::linalg::solve_triangular(a_solve, b_solve, /*upper=*/!lower,
                                                           mlx::core::Device::cpu);
            auto result = nan_guard(mlx::core::transpose(x_t, perm_b));
            values.emplace(ToKey(op->getResult(0)), std::move(result));
            return true;
        }

        // MLX solve_triangular uses upper=!lower
        auto result = nan_guard(mlx::core::linalg::solve_triangular(a_solve, b_solve,
                                                                    /*upper=*/!lower,
                                                                    mlx::core::Device::cpu));
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: %s\n", e.what());
        return false;
    }
}

// Op dispatch table - initialized once
const std::unordered_map<std::string, OpHandler>& GetOpHandlers() {
    static const std::unordered_map<std::string, OpHandler> handlers = {
        // Arithmetic
        {"stablehlo.add", HandleAdd},
        {"stablehlo.subtract", HandleSubtract},
        {"stablehlo.multiply", HandleMultiply},
        {"stablehlo.negate", HandleNegate},
        {"stablehlo.abs", HandleAbs},
        {"stablehlo.exponential", HandleExp},
        {"stablehlo.log", HandleLog},
        {"stablehlo.sqrt", HandleSqrt},
        {"stablehlo.rsqrt", HandleRsqrt},
        {"stablehlo.log_plus_one", HandleLogPlusOne},
        {"stablehlo.maximum", HandleMaximum},
        {"stablehlo.minimum", HandleMinimum},
        {"stablehlo.divide", HandleDivide},
        {"stablehlo.floor", HandleFloor},
        {"stablehlo.sine", HandleSine},
        {"stablehlo.cosine", HandleCosine},
        {"stablehlo.clamp", HandleClamp},
        {"stablehlo.power", HandlePower},
        {"stablehlo.tanh", HandleTanh},
        {"stablehlo.tan", HandleTan},
        {"stablehlo.sign", HandleSign},
        {"stablehlo.remainder", HandleRemainder},
        {"stablehlo.ceil", HandleCeil},
        {"stablehlo.round_nearest_even", HandleRoundNearestEven},
        {"stablehlo.is_finite", HandleIsFinite},
        {"stablehlo.exponential_minus_one", HandleExpm1},
        {"stablehlo.cbrt", HandleCbrt},
        {"stablehlo.atan2", HandleAtan2},
        // Bitwise
        {"stablehlo.and", HandleAnd},
        {"stablehlo.or", HandleOr},
        {"stablehlo.xor", HandleXor},
        {"stablehlo.not", HandleNot},
        {"stablehlo.shift_left", HandleShiftLeft},
        {"stablehlo.shift_right_logical", HandleShiftRightLogical},
        {"stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic},
        {"stablehlo.popcnt", HandlePopcount},
        // Comparison/selection
        {"stablehlo.compare", HandleCompare},
        {"stablehlo.select", HandleSelect},
        // Type/shape
        {"stablehlo.constant", HandleConstant},
        {"stablehlo.convert", HandleConvert},
        {"stablehlo.bitcast_convert", HandleBitcastConvert},
        {"stablehlo.reshape", HandleReshape},
        {"stablehlo.broadcast_in_dim", HandleBroadcastInDim},
        {"stablehlo.concatenate", HandleConcatenate},
        {"stablehlo.transpose", HandleTranspose},
        {"stablehlo.reverse", HandleReverse},
        {"stablehlo.iota", HandleIota},
        {"stablehlo.slice", HandleSlice},
        {"stablehlo.dynamic_slice", HandleDynamicSlice},
        {"stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice},
        {"stablehlo.pad", HandlePad},
        {"stablehlo.gather", HandleGather},
        {"stablehlo.scatter", HandleScatter},
        // Linear algebra
        {"stablehlo.dot_general", HandleDotGeneral},
        {"stablehlo.convolution", HandleConvolution},
        {"stablehlo.cholesky", HandleCholesky},
        {"stablehlo.triangular_solve", HandleTriangularSolve},
        // Reduction
        {"stablehlo.reduce", HandleReduce},
        {"stablehlo.reduce_window", HandleReduceWindow},
        {"stablehlo.select_and_scatter", HandleSelectAndScatter},
        // Control flow
        {"func.return", HandleReturn},
        {"func.call", HandleCall},
        {"stablehlo.custom_call", HandleCustomCall},
        {"stablehlo.optimization_barrier", HandleOptimizationBarrier},
        {"stablehlo.while", HandleWhile},
        {"stablehlo.case", HandleCase},
        // Sort
        {"stablehlo.sort", HandleSort},
        // FFT
        {"stablehlo.fft", HandleFft},
        // Complex
        {"stablehlo.complex", HandleComplex},
        {"stablehlo.real", HandleReal},
        {"stablehlo.imag", HandleImag},
        {"stablehlo.return", HandleStablehloReturn},
    };
    return handlers;
}

// Dispatch a single operation using the handler table
bool DispatchOp(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    std::string opName = op->getName().getStringRef().str();

    MPS_LOG_DEBUG("Dispatching op: %s\n", opName.c_str());

    const auto& handlers = GetOpHandlers();
    auto it = handlers.find(opName);
    if (it == handlers.end()) {
        MPS_LOG_ERROR("Unsupported op: %s\n", opName.c_str());
        return false;
    }

    try {
        if (IsProfilingEnabled()) {
            auto start = Clock::now();
            bool result = it->second(op, values, outputs, ctx);
            auto end = Clock::now();
            GetProfilingState().RecordOp(opName, Duration(end - start).count());
            return result;
        }
        return it->second(op, values, outputs, ctx);
    } catch (const CompileIncompatibleError&) {
        throw;  // Propagate without logging — compile() will catch and fall back
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("Exception dispatching %s: %s\n", opName.c_str(), e.what());
        return false;
    }
}

// Execute a region (for while loops, etc.) with given inputs
// parentValues allows inner regions to access values defined in outer scopes
// (e.g., constants hoisted out of while-loop regions by optimization passes)
bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx,
                   const ValueMap* parentValues) {
    if (region.empty()) {
        MPS_LOG_ERROR("ExecuteRegion: empty region\n");
        return false;
    }

    // Use a local ValueMap that falls back to parentValues for lookups.
    // We achieve this by wrapping GetValue lookups: local values first, then parent.
    // For simplicity, we create a combined map by inserting references to parent values.
    ValueMap values;

    // Insert parent values by reference (shallow copy of mlx::core::array is fine)
    if (parentValues) {
        for (const auto& kv : *parentValues) {
            values.emplace(kv.first, kv.second);
        }
    }

    auto& block = region.front();

    // Map block arguments to input arrays (may shadow parent values)
    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        if (argIdx >= args.size()) {
            MPS_LOG_ERROR("ExecuteRegion: not enough arguments\n");
            return false;
        }
        // Use insert_or_assign to shadow any parent value with the same key
        values.insert_or_assign(ToKey(arg), args[argIdx]);
        argIdx++;
    }

    // Walk operations and dispatch
    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, results, ctx)) {
            return false;
        }
    }

    return true;
}

// Execute a function with given inputs, collecting outputs
bool ExecuteFunction(mlir::func::FuncOp func, const std::vector<mlx::core::array>& inputs,
                     std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    ValueMap values;

    auto& block = func.front();
    size_t numArgs = block.getNumArguments();

    if (inputs.size() != numArgs) {
        MPS_LOG_ERROR("ExecuteFunction: input count mismatch: expected %zu, got %zu\n", numArgs,
                      inputs.size());
        return false;
    }

    // Map function arguments to input arrays
    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        values.emplace(ToKey(arg), inputs[argIdx]);
        argIdx++;
    }

    // Walk operations and dispatch
    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, outputs, ctx)) {
            return false;
        }
    }

    return true;
}

}  // namespace

std::unordered_set<std::string> GetSupportedOpNames() {
    const auto& handlers = GetOpHandlers();
    std::unordered_set<std::string> names;
    for (const auto& pair : handlers) {
        names.insert(pair.first);
    }
    return names;
}

std::unique_ptr<MlxExecutable> MlxExecutable::Create(mps::ParsedModule parsed_module) {
    auto executable = std::unique_ptr<MlxExecutable>(new MlxExecutable());

    if (!parsed_module.ok()) {
        executable->error_ = "Invalid parsed module";
        executable->valid_ = false;
        return executable;
    }

    // Check for unsupported ops
    if (!parsed_module.unsupported_ops.empty()) {
        executable->error_ = "Unsupported operations: ";
        for (size_t i = 0; i < parsed_module.unsupported_ops.size(); ++i) {
            if (i > 0)
                executable->error_ += ", ";
            executable->error_ += parsed_module.unsupported_ops[i];
        }
        executable->valid_ = false;
        return executable;
    }

    // Store the parsed module (keeps MLIR context alive)
    executable->parsed_module_ = std::move(parsed_module);

    // Extract output info from function return type
    auto funcType = executable->parsed_module_.entry_func.getFunctionType();
    executable->num_outputs_ = funcType.getNumResults();

    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        auto resultType = funcType.getResult(i);
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
            OutputInfo info;
            info.dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
            for (int64_t dim : tensorType.getShape()) {
                info.shape.push_back(dim);
            }
            executable->output_info_.push_back(info);
        } else {
            // Non-tensor output, use defaults
            OutputInfo info;
            info.dtype = PJRT_Buffer_Type_F32;
            executable->output_info_.push_back(info);
        }
    }

    executable->valid_ = true;
    MPS_LOG_DEBUG("Created MlxExecutable with %zu outputs\n", executable->num_outputs_);

    return executable;
}

bool MlxExecutable::IsValid() const {
    return valid_;
}

std::string MlxExecutable::error() const {
    return error_;
}

size_t MlxExecutable::num_outputs() const {
    return num_outputs_;
}

MlxExecuteResult MlxExecutable::Execute(const std::vector<MlxBuffer*>& inputs) {
    // NOTE: Thread safety is handled by GetPjrtGlobalMutex() at the PJRT API layer.
    MlxExecuteResult result;
    const bool profiling = IsProfilingEnabled();
    Clock::time_point exec_start;
    Clock::time_point dispatch_start;
    Clock::time_point dispatch_end;
    Clock::time_point eval_start;
    Clock::time_point eval_end;
    Clock::time_point exec_end;

    if (profiling) {
        exec_start = Clock::now();
        // Track time since last Execute() call
        auto& state = GetProfilingState();
        if (state.has_last_time) {
            state.cumulative_between_calls_ms +=
                Duration(exec_start - state.last_execute_end).count();
        }
    }

    if (!valid_) {
        MPS_LOG_ERROR("Cannot execute invalid executable: %s\n", error_.c_str());
        return result;
    }

    MPS_LOG_DEBUG("Executing with %zu inputs\n", inputs.size());

    // Validate inputs
    auto& block = parsed_module_.entry_func.front();
    size_t numArgs = block.getNumArguments();

    if (inputs.size() != numArgs) {
        MPS_LOG_ERROR("Input count mismatch: expected %zu, got %zu\n", numArgs, inputs.size());
        return result;
    }

    // Convert MlxBuffer inputs to mlx::core::array
    std::vector<mlx::core::array> inputArrays;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) {
            MPS_LOG_ERROR("Null input buffer at index %zu\n", i);
            return result;
        }
        inputArrays.push_back(inputs[i]->array());
    }

    // Set up execution context with module reference
    ExecContext ctx;
    ctx.module = *parsed_module_.module;

    // Execute the entry function (graph construction + op dispatch)
    if (profiling) {
        dispatch_start = Clock::now();
    }
    std::vector<mlx::core::array> outputs;

    // Try MLX compile() on first execution - it can fuse kernels for better performance
    // Can be disabled with MPS_NO_COMPILE=1 for debugging
    static bool disable_compile = std::getenv("MPS_NO_COMPILE") != nullptr;
    {
        // NOTE: Thread safety handled by GetPjrtGlobalMutex() at PJRT API layer.
        if (!compile_attempted_ && !disable_compile) {
            compile_attempted_ = true;

            // Create a function that we can compile
            // Note: capture 'this' only - ctx would be invalid on subsequent calls
            auto exec_fn =
                [this](
                    const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                std::vector<mlx::core::array> outs;
                ExecContext local_ctx;
                local_ctx.module = *parsed_module_.module;
                local_ctx.inside_compile = true;
                if (!ExecuteFunction(parsed_module_.entry_func, inputs, outs, local_ctx)) {
                    return {};
                }
                return outs;
            };

            try {
                // Try to compile the function
                compiled_fn_ = mlx::core::compile(exec_fn);

                // Test that compile works by running it once
                auto test_outputs = compiled_fn_(inputArrays);
                if (!test_outputs.empty()) {
                    mlx::core::eval(test_outputs);
                    compile_succeeded_ = true;
                    outputs = std::move(test_outputs);
                    MPS_LOG_INFO("MLX compile() succeeded - using compiled execution path\n");
                }
            } catch (const std::exception& e) {
                MPS_LOG_INFO("MLX compile() failed (%s), using direct path\n", e.what());
                compile_succeeded_ = false;
            }
        }
    }

    // Use compiled function if available, otherwise fall back to direct execution
    if (compile_succeeded_ && outputs.empty()) {
        outputs = compiled_fn_(inputArrays);
    } else if (outputs.empty()) {
        if (!ExecuteFunction(parsed_module_.entry_func, inputArrays, outputs, ctx)) {
            MPS_LOG_ERROR("Failed to execute entry function\n");
            return result;
        }
    }

    if (profiling) {
        dispatch_end = Clock::now();
    }

    // Validate output count
    if (outputs.size() != num_outputs_) {
        MPS_LOG_ERROR("Output count mismatch: expected %zu, got %zu\n", num_outputs_,
                      outputs.size());
        return result;
    }

    // Evaluate all outputs. We cannot defer eval to ToHostBuffer() because
    // PJRT_Buffer_ReadyEvent currently always returns ready=true, so
    // block_until_ready() would not synchronize and benchmarks/correctness
    // would be affected.
    if (profiling) {
        eval_start = Clock::now();
    }
    if (!outputs.empty()) {
        try {
            mlx::core::eval(outputs);
        } catch (const std::exception& e) {
            MPS_LOG_ERROR("MLX evaluation failed: %s\n", e.what());
            return result;
        }
    }
    if (profiling) {
        eval_end = Clock::now();
    }

    // Wrap outputs in MlxBuffer
    for (auto& arr : outputs) {
        result.buffers.push_back(MlxBuffer::FromArray(std::move(arr)));
    }

    // Record profiling stats
    if (profiling) {
        exec_end = Clock::now();
        auto& state = GetProfilingState();
        state.dispatch_overhead_ms = Duration(dispatch_end - dispatch_start).count();
        state.eval_time_ms = Duration(eval_end - eval_start).count();
        state.total_execution_ms = Duration(exec_end - exec_start).count();
        state.last_execute_end = exec_end;
        state.has_last_time = true;

        // Track slow executions (likely forward/backward passes)
        static size_t slow_count = 0;
        static double slow_total_ms = 0;
        if (state.total_execution_ms > 100.0) {
            slow_count++;
            slow_total_ms += state.total_execution_ms;
            fprintf(stderr, "[COMPUTE #%zu] %.0f ms (eval: %.0f ms)\n", slow_count,
                    state.total_execution_ms, state.eval_time_ms);
        }

        state.PrintSummary();
    }

    MPS_LOG_DEBUG("Execution complete with %zu outputs\n", result.buffers.size());

    return result;
}

}  // namespace jax_mps
