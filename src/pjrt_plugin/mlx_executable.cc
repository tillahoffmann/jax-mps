// MLX executable implementation with op dispatch

#include "pjrt_plugin/mlx_executable.h"

#include <mlx/compile.h>
#include <mlx/compile_impl.h>
#include <mlx/memory.h>
#include <mlx/mlx.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/ops/handler_utils.h"
#include "pjrt_plugin/type_utils.h"

namespace jax_mps {

// --- Shared utility function definitions (declared in handler_utils.h) ---

void* ToKey(mlir::Value v) {
    return v.getAsOpaquePointer();
}

std::optional<std::reference_wrapper<mlx::core::array>> GetValue(ValueMap& values, mlir::Value v) {
    auto it = values.find(ToKey(v));
    if (it == values.end()) {
        return std::nullopt;
    }
    return std::ref(it->second);
}

mlx::core::Dtype MlirTypeToMlxDtype(mlir::Type type) {
    int pjrt_dtype = MlirTypeToPjrtDtype(type);
    if (pjrt_dtype == -1) {
        MPS_LOG_ERROR("Unknown MLIR type, defaulting to float32\n");
        return mlx::core::float32;
    }
    if (type.isF64()) {
        MPS_LOG_WARN("MLX doesn't support float64, downcasting to float32\n");
        return mlx::core::float32;
    }
    return PjrtDtypeToMlx(pjrt_dtype);
}

mlx::core::Shape GetShape(mlir::RankedTensorType type) {
    mlx::core::Shape shape;
    for (int64_t dim : type.getShape()) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}

std::optional<mlx::core::array> CreateArrayWithTypedPtr(const void* data,
                                                        const mlx::core::Shape& shape,
                                                        mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::bool_: {
            // MLIR i1 splat data may store true as 0xFF (-1). Normalize to 0/1
            // so downstream ops (cumsum, etc.) see correct integer values.
            bool val = *reinterpret_cast<const uint8_t*>(data) != 0;
            return mlx::core::array(&val, shape, dtype);
        }
        case mlx::core::int8:
            return mlx::core::array(reinterpret_cast<const int8_t*>(data), shape, dtype);
        case mlx::core::int16:
            return mlx::core::array(reinterpret_cast<const int16_t*>(data), shape, dtype);
        case mlx::core::int32:
            return mlx::core::array(reinterpret_cast<const int32_t*>(data), shape, dtype);
        case mlx::core::int64:
            return mlx::core::array(reinterpret_cast<const int64_t*>(data), shape, dtype);
        case mlx::core::uint8:
            return mlx::core::array(reinterpret_cast<const uint8_t*>(data), shape, dtype);
        case mlx::core::uint16:
            return mlx::core::array(reinterpret_cast<const uint16_t*>(data), shape, dtype);
        case mlx::core::uint32:
            return mlx::core::array(reinterpret_cast<const uint32_t*>(data), shape, dtype);
        case mlx::core::uint64:
            return mlx::core::array(reinterpret_cast<const uint64_t*>(data), shape, dtype);
        case mlx::core::float16:
            return mlx::core::array(reinterpret_cast<const mlx::core::float16_t*>(data), shape,
                                    dtype);
        case mlx::core::bfloat16:
            return mlx::core::array(reinterpret_cast<const mlx::core::bfloat16_t*>(data), shape,
                                    dtype);
        case mlx::core::float32:
            return mlx::core::array(reinterpret_cast<const float*>(data), shape, dtype);
        case mlx::core::complex64:
            return mlx::core::array(reinterpret_cast<const mlx::core::complex64_t*>(data), shape,
                                    dtype);
        default:
            return std::nullopt;
    }
}

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

    // Handle splat constants (single value broadcast to shape)
    if (attr.isSplat()) {
        auto scalar_opt = CreateArrayWithTypedPtr(rawData.data(), {}, mlxDtype);
        if (!scalar_opt) {
            MPS_LOG_ERROR("Unsupported dtype %d for splat constant\n",
                          static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
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

    // MLIR stores i1 (boolean) data as bit-packed: 1 bit per element.
    if (mlxDtype == mlx::core::bool_) {
        size_t expectedBitPackedSize = (numElements + 7) / 8;
        if (rawData.size() < expectedBitPackedSize) {
            MPS_LOG_ERROR(
                "Boolean constant data size mismatch: got %zu bytes, expected %zu (bit-packed for "
                "%zu elements)\n",
                rawData.size(), expectedBitPackedSize, numElements);
            return std::nullopt;
        }
        std::vector<uint8_t> unpacked(numElements);
        const uint8_t* bits = reinterpret_cast<const uint8_t*>(rawData.data());
        for (size_t i = 0; i < numElements; ++i) {
            unpacked[i] = (bits[i / 8] >> (i % 8)) & 1;
        }
        auto arr = mlx::core::array(unpacked.data(), shape, mlx::core::uint8);
        return mlx::core::astype(arr, mlx::core::bool_);
    }

    if (rawData.size() < expectedSize) {
        MPS_LOG_ERROR("Constant data size mismatch: got %zu bytes, expected %zu\n", rawData.size(),
                      expectedSize);
        return std::nullopt;
    }

    auto result = CreateArrayWithTypedPtr(rawData.data(), shape, mlxDtype);
    if (!result) {
        MPS_LOG_ERROR("Unsupported dtype %d for constant\n",
                      static_cast<int>(static_cast<mlx::core::Dtype::Val>(mlxDtype)));
    }
    return result;
}

// --- Factory function definitions (declared in handler_utils.h) ---

OpHandler MakeUnaryHandler(const char* opName, UnaryMlxFn fn) {
    return [opName, fn](mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) -> bool {
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), fn(input_opt->get(), {}));
        return true;
    };
}

OpHandler MakeBinaryHandler(const char* opName, BinaryMlxFn fn) {
    return [opName, fn](mlir::Operation* op, ValueMap& values,
                        std::vector<mlx::core::array>& outputs, ExecContext& ctx) -> bool {
        auto lhs_opt = GetValue(values, op->getOperand(0));
        auto rhs_opt = GetValue(values, op->getOperand(1));
        if (!lhs_opt || !rhs_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), fn(lhs_opt->get(), rhs_opt->get(), {}));
        return true;
    };
}

OpHandler MakeLogicalShiftHandler(const char* opName, BinaryMlxFn shiftFn) {
    return [opName, shiftFn](mlir::Operation* op, ValueMap& values,
                             std::vector<mlx::core::array>& outputs, ExecContext& ctx) -> bool {
        auto lhs_opt = GetValue(values, op->getOperand(0));
        auto rhs_opt = GetValue(values, op->getOperand(1));
        if (!lhs_opt || !rhs_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        auto& lhs = lhs_opt->get();
        auto& rhs = rhs_opt->get();
        int bit_width = static_cast<int>(GetDtypeSize(lhs.dtype()) * 8);
        auto zero = mlx::core::zeros_like(lhs);
        auto oob = mlx::core::logical_or(
            mlx::core::less(rhs, mlx::core::array(0, rhs.dtype())),
            mlx::core::greater_equal(rhs, mlx::core::array(bit_width, rhs.dtype())));
        auto shifted = shiftFn(lhs, mlx::core::maximum(rhs, mlx::core::array(0, rhs.dtype())), {});
        values.emplace(ToKey(op->getResult(0)), mlx::core::where(oob, zero, shifted));
        return true;
    };
}

namespace {

// Profiling infrastructure
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

bool IsProfilingEnabled() {
    static bool enabled = std::getenv("MPS_PROFILE") != nullptr;
    return enabled;
}

struct OpTimingStats {
    double total_ms = 0.0;
    size_t count = 0;
};

struct ProfilingState {
    std::unordered_map<std::string, OpTimingStats> op_times;
    double dispatch_overhead_ms = 0.0;
    double eval_time_ms = 0.0;
    double total_execution_ms = 0.0;
    size_t execution_count = 0;

    double cumulative_dispatch_ms = 0.0;
    double cumulative_eval_ms = 0.0;
    double cumulative_total_ms = 0.0;

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
        cumulative_dispatch_ms += dispatch_overhead_ms;
        cumulative_eval_ms += eval_time_ms;
        cumulative_total_ms += total_execution_ms;

        if (execution_count % 1000 != 0) {
            return;
        }

        fprintf(stderr, "\n=== MPS Final Summary (%zu executions) ===\n", execution_count);
        fprintf(stderr, "Total GPU time: %.0f ms (dispatch: %.0f ms, eval: %.0f ms)\n",
                cumulative_total_ms, cumulative_dispatch_ms, cumulative_eval_ms);

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

        size_t peak_mem = mlx::core::get_peak_memory();
        fprintf(stderr, "Peak memory: %.0f MB\n", static_cast<double>(peak_mem) / 1e6);
        fprintf(stderr, "=========================================\n");

        Reset();
    }
};

ProfilingState& GetProfilingState() {
    static ProfilingState state;
    return state;
}

// --- Op dispatch table ---

const std::unordered_map<std::string, OpHandler>& GetOpHandlers() {
    static auto handlers = [] {
        std::unordered_map<std::string, OpHandler> h;
        RegisterArithmeticHandlers(h);
        RegisterShapeHandlers(h);
        RegisterSliceHandlers(h);
        RegisterGatherScatterHandlers(h);
        RegisterReductionHandlers(h);
        RegisterLinalgHandlers(h);
        RegisterControlFlowHandlers(h);
        RegisterSortFftComplexHandlers(h);
        return h;
    }();
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

}  // namespace

// --- Cross-TU function definitions (declared in handler_utils.h) ---

bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx,
                   const ValueMap* parentValues) {
    if (region.empty()) {
        MPS_LOG_ERROR("ExecuteRegion: empty region\n");
        return false;
    }

    ValueMap values;

    if (parentValues) {
        for (const auto& kv : *parentValues) {
            values.emplace(kv.first, kv.second);
        }
    }

    auto& block = region.front();

    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        if (argIdx >= args.size()) {
            MPS_LOG_ERROR("ExecuteRegion: not enough arguments\n");
            return false;
        }
        values.insert_or_assign(ToKey(arg), args[argIdx]);
        argIdx++;
    }

    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, results, ctx)) {
            return false;
        }
    }

    return true;
}

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

    size_t argIdx = 0;
    for (auto arg : block.getArguments()) {
        values.emplace(ToKey(arg), inputs[argIdx]);
        argIdx++;
    }

    for (auto& op : block.getOperations()) {
        if (!DispatchOp(&op, values, outputs, ctx)) {
            return false;
        }
    }

    return true;
}

// --- Public API ---

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

    executable->parsed_module_ = std::move(parsed_module);

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
            OutputInfo info;
            info.dtype = PJRT_Buffer_Type_F32;
            executable->output_info_.push_back(info);
        }
    }

    executable->valid_ = true;
    MPS_LOG_DEBUG("Created MlxExecutable with %zu outputs\n", executable->num_outputs_);

    return executable;
}

MlxExecutable::~MlxExecutable() {
    // Drop the MLX process-global compiler-cache entry keyed by `this`. Without
    // this, the cache holds a CacheEntry — including the full traced tape and
    // any captured constants — for the lifetime of the process, even after JAX
    // evicts this executable.
    //
    // Skip during process shutdown: the static CompilerCache may already be
    // destroyed, and calling compile_erase on it would SIGSEGV / double-free.
    if (compile_attempted_ && !IsProcessShuttingDown()) {
        mlx::core::detail::compile_erase(reinterpret_cast<std::uintptr_t>(this));
    }
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

    auto& block = parsed_module_.entry_func.front();
    size_t numArgs = block.getNumArguments();

    if (inputs.size() != numArgs) {
        MPS_LOG_ERROR("Input count mismatch: expected %zu, got %zu\n", numArgs, inputs.size());
        return result;
    }

    std::vector<mlx::core::array> inputArrays;
    inputArrays.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) {
            MPS_LOG_ERROR("Null input buffer at index %zu\n", i);
            return result;
        }
        inputArrays.push_back(inputs[i]->array());
    }

    ExecContext ctx;
    ctx.module = *parsed_module_.module;

    if (profiling) {
        dispatch_start = Clock::now();
    }
    std::vector<mlx::core::array> outputs;

    static bool disable_compile = std::getenv("MPS_NO_COMPILE") != nullptr;
    {
        if (!compile_attempted_ && !disable_compile) {
            compile_attempted_ = true;

            auto exec_fn =
                [this](
                    const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                std::vector<mlx::core::array> outs;
                ExecContext local_ctx;
                local_ctx.module = *parsed_module_.module;
                local_ctx.inside_compile = true;
                local_ctx.allow_while_primitive = true;
                if (!ExecuteFunction(parsed_module_.entry_func, inputs, outs, local_ctx)) {
                    return {};
                }
                return outs;
            };

            try {
                // Use `this` as the fun_id so we can erase this executable's
                // entry from MLX's global compiler cache in our destructor.
                // The public compile() overload can't derive a stable id from a
                // capturing lambda and would otherwise accumulate entries under
                // id=0 that are never released.
                compiled_fn_ =
                    mlx::core::detail::compile(exec_fn, reinterpret_cast<std::uintptr_t>(this));

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

    if (outputs.size() != num_outputs_) {
        MPS_LOG_ERROR("Output count mismatch: expected %zu, got %zu\n", num_outputs_,
                      outputs.size());
        return result;
    }

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

    for (auto& arr : outputs) {
        result.buffers.push_back(MlxBuffer::FromArray(std::move(arr)));
    }

    if (profiling) {
        exec_end = Clock::now();
        auto& state = GetProfilingState();
        state.dispatch_overhead_ms = Duration(dispatch_end - dispatch_start).count();
        state.eval_time_ms = Duration(eval_end - eval_start).count();
        state.total_execution_ms = Duration(exec_end - exec_start).count();
        state.last_execute_end = exec_end;
        state.has_last_time = true;

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
