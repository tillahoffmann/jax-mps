// MLX executable implementation with op dispatch

#include "pjrt_plugin/mlx_executable.h"

#include <mlx/compile.h>
#include <mlx/memory.h>
#include <mlx/mlx.h>
#include <mlx/primitives.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "llvm/Support/raw_ostream.h"
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

// jax-mps#170: a non-finite float constant (the NaN/inf guard that jnp.std/var
// and numpyro grads bake in) materialized as a plain leaf array becomes a
// never-written buffer under mlx::core::compile(), so the compiled kernel reads
// garbage (intermittent zero / wrong gradients). Materialize such constants
// instead via a bitcast of their integer bit-pattern: that is a *computed* value
// (written by a kernel before it is read) and compiles correctly. Returns
// nullopt for finite values / non-float dtypes so callers use the normal copy
// path. `numElements` is the number of float elements addressable at `data`
// (1 for a splat). `data` (DenseElementsAttr::getRawData()) is byte-oriented and
// not guaranteed to be element-aligned, so the bytes are memcpy'd into aligned
// storage before being read as integers.
std::optional<mlx::core::array> MaybeBitcastNonFiniteFloat(const void* data,
                                                           const mlx::core::Shape& shape,
                                                           mlx::core::Dtype dtype,
                                                           size_t numElements) {
    // Non-finite (inf/nan) iff the exponent field is all ones.
    if (dtype == mlx::core::float32) {
        std::vector<uint32_t> bits(numElements);
        std::memcpy(bits.data(), data, numElements * sizeof(uint32_t));
        bool nonFinite = false;
        for (uint32_t b : bits) {
            if ((b & 0x7F800000U) == 0x7F800000U) {
                nonFinite = true;
                break;
            }
        }
        if (!nonFinite)
            return std::nullopt;
        return mlx::core::view(mlx::core::array(bits.data(), shape, mlx::core::uint32),
                               mlx::core::float32);
    }
    if (dtype == mlx::core::float16 || dtype == mlx::core::bfloat16) {
        const uint16_t expMask = dtype == mlx::core::float16 ? 0x7C00U : 0x7F80U;
        std::vector<uint16_t> bits(numElements);
        std::memcpy(bits.data(), data, numElements * sizeof(uint16_t));
        bool nonFinite = false;
        for (uint16_t b : bits) {
            if ((b & expMask) == expMask) {
                nonFinite = true;
                break;
            }
        }
        if (!nonFinite)
            return std::nullopt;
        return mlx::core::view(mlx::core::array(bits.data(), shape, mlx::core::uint16), dtype);
    }
    return std::nullopt;
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
        // jax-mps#170: route non-finite floats through a bitcast (computed value).
        auto scalar_opt = MaybeBitcastNonFiniteFloat(rawData.data(), {}, mlxDtype, 1);
        if (!scalar_opt)
            scalar_opt = CreateArrayWithTypedPtr(rawData.data(), {}, mlxDtype);
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

    // MLIR's DenseElementsAttr stores i1 either bit-packed (1 bit per element,
    // legacy / aggressively-folded constants) or byte-per-element (modern, what
    // StableHLO 1.16 / the bytecode shipped with jaxlib 0.10 produces).
    // Disambiguate by comparing the raw data size against both encodings.
    if (mlxDtype == mlx::core::bool_) {
        const size_t bitPackedSize = (numElements + 7) / 8;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(rawData.data());
        std::vector<uint8_t> unpacked(numElements);
        if (rawData.size() >= numElements) {
            // One byte per element — value is a non-zero byte for true.
            for (size_t i = 0; i < numElements; ++i) {
                unpacked[i] = bytes[i] != 0 ? 1 : 0;
            }
        } else if (rawData.size() >= bitPackedSize) {
            for (size_t i = 0; i < numElements; ++i) {
                unpacked[i] = (bytes[i / 8] >> (i % 8)) & 1;
            }
        } else {
            MPS_LOG_ERROR(
                "Boolean constant data size mismatch: got %zu bytes, expected %zu (bit-packed) or "
                "%zu (byte-per-element) for %zu elements\n",
                rawData.size(), bitPackedSize, numElements, numElements);
            return std::nullopt;
        }
        auto arr = mlx::core::array(unpacked.data(), shape, mlx::core::uint8);
        return mlx::core::astype(arr, mlx::core::bool_);
    }

    if (rawData.size() < expectedSize) {
        MPS_LOG_ERROR("Constant data size mismatch: got %zu bytes, expected %zu\n", rawData.size(),
                      expectedSize);
        return std::nullopt;
    }

    // jax-mps#170: route non-finite floats through a bitcast (computed value).
    auto result = MaybeBitcastNonFiniteFloat(rawData.data(), shape, mlxDtype, numElements);
    if (!result)
        result = CreateArrayWithTypedPtr(rawData.data(), shape, mlxDtype);
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

// Opt-in async dispatch (JAX_MPS_ASYNC_DISPATCH). When enabled, the final
// materialization of an executable's outputs uses mlx::core::async_eval instead
// of the blocking mlx::core::eval, letting Execute() return before the GPU
// finishes so the caller can dispatch the next computation (CPU/GPU pipelining).
// PJRT completion events (see PJRT_Event) track real GPU completion, so
// block_until_ready() and host reads remain correct.
//
// The value is parsed, not just tested for presence: "1"/"true"/"yes"/"on"
// (case-insensitive) enable it; anything else — including "0" and unset —
// disables it, so JAX_MPS_ASYNC_DISPATCH=0 turns it off as expected. Declared in
// the header so the startup notice (pjrt_client.cc) uses the identical check.
bool IsAsyncDispatchEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("JAX_MPS_ASYNC_DISPATCH");
        if (value == nullptr) {
            return false;
        }
        auto iequals = [](const char* a, const char* b) {
            for (; *a != '\0' && *b != '\0'; ++a, ++b) {
                if (std::tolower(static_cast<unsigned char>(*a)) !=
                    std::tolower(static_cast<unsigned char>(*b))) {
                    return false;
                }
            }
            return *a == *b;
        };
        return iequals(value, "1") || iequals(value, "true") || iequals(value, "yes") ||
               iequals(value, "on");
    }();
    return enabled;
}

namespace {

// Profiling infrastructure
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

bool IsProfilingEnabled() {
    static bool enabled = std::getenv("MPS_PROFILE") != nullptr;
    return enabled;
}

// Collect the outputs of control-flow primitives (while/case) reachable from
// `roots` in the lazy graph. Those primitives run a host-controlled loop via a
// nested synchronous mlx::core::eval(); that re-entrant eval is only safe under
// a *synchronous* outer pass. Under async dispatch we therefore resolve their
// dependency cones with a synchronous eval() first (see Execute), then
// async_eval() the remainder so independent/downstream work — and the next
// dispatch — still pipeline. Traversal stops at each control-flow output: a
// synchronous eval() of it pulls in its inputs (and any nested/upstream
// control flow) transitively.
std::vector<mlx::core::array> CollectControlFlowOutputs(
    const std::vector<mlx::core::array>& roots) {
    std::vector<mlx::core::array> found;
    std::unordered_set<std::uintptr_t> visited;
    std::vector<mlx::core::array> stack(roots.begin(), roots.end());
    while (!stack.empty()) {
        mlx::core::array arr = std::move(stack.back());
        stack.pop_back();
        if (!visited.insert(arr.id()).second)
            continue;
        if (!arr.has_primitive())
            continue;
        const char* pname = arr.primitive().name();
        if (std::strcmp(pname, "WhileLoop") == 0 || std::strcmp(pname, "Case") == 0) {
            found.push_back(std::move(arr));
            continue;
        }
        for (const auto& in : arr.inputs())
            stack.push_back(in);
    }
    return found;
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

// Build a one-line fingerprint of an op for diagnostics: opName, identifying
// attributes (e.g. `stablehlo.custom_call`'s `call_target_name`, which is the
// only thing distinguishing ApproxTopK from Householder etc.), plus operand
// and result MLIR types. Used when a handler returns false without throwing —
// most such failures are dtype/shape rejections, and the fingerprint usually
// tells you what wasn't handled.
std::string FormatOpFingerprint(mlir::Operation* op) {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << op->getName().getStringRef();
    if (auto targetAttr = op->getAttrOfType<mlir::StringAttr>("call_target_name")) {
        os << "[" << targetAttr.getValue() << "]";
    }
    os << "(";
    bool first = true;
    for (auto operand : op->getOperands()) {
        if (!first)
            os << ", ";
        first = false;
        operand.getType().print(os);
    }
    os << ") -> (";
    first = true;
    for (auto resType : op->getResultTypes()) {
        if (!first)
            os << ", ";
        first = false;
        resType.print(os);
    }
    os << ")";
    return out;
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

    bool result;
    try {
        if (IsProfilingEnabled()) {
            auto start = Clock::now();
            result = it->second(op, values, outputs, ctx);
            auto end = Clock::now();
            GetProfilingState().RecordOp(opName, Duration(end - start).count());
        } else {
            result = it->second(op, values, outputs, ctx);
        }
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("Exception dispatching %s: %s\n", opName.c_str(), e.what());
        if (ctx.error_message.empty()) {
            ctx.error_message = FormatOpFingerprint(op) + ": " + e.what();
        }
        return false;
    }
    if (!result && ctx.error_message.empty()) {
        // Handler returned false without setting a reason (validation reject,
        // unsupported dtype/shape combo, etc.). Synthesize a fingerprint from
        // the op's operand and result types so the failure is greppable.
        ctx.error_message = FormatOpFingerprint(op) + ": handler returned false";
    }
    return result;
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

MlxExecutable::~MlxExecutable() = default;

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
        // Safety net: if a func.call survived inlining (a multi-block / control-
        // flow callee the inliner can't clone), use the eager path instead of
        // compile() (jax-mps#170). Single-block callees are inlined and compile.
        if (!compile_attempted_ && !disable_compile && !parsed_module_.has_uninlined_call) {
            compile_attempted_ = true;

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
                // Remember whether this graph contains control-flow primitives,
                // so the async path can skip the control-flow walk when it
                // doesn't (set during the compile trace; replays reuse it).
                has_control_flow_ = local_ctx.produced_control_flow;
                return outs;
            };

            try {
                // Public compile() overload wraps capturing closures in a
                // shared_ptr whose deleter calls compile_erase when the
                // returned std::function dies — eviction is tied to
                // compiled_fn_'s lifetime, no manual erase needed.
                compiled_fn_ = mlx::core::compile(exec_fn);

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
            result.error_message = std::move(ctx.error_message);
            return result;
        }
    }

    if (profiling) {
        dispatch_end = Clock::now();
    }

    if (outputs.size() != num_outputs_) {
        MPS_LOG_ERROR("Output count mismatch: expected %zu, got %zu\n", num_outputs_,
                      outputs.size());
        if (result.error_message.empty()) {
            result.error_message = std::move(ctx.error_message);
        }
        return result;
    }

    if (profiling) {
        eval_start = Clock::now();
    }
    if (!outputs.empty()) {
        try {
            if (IsAsyncDispatchEnabled()) {
                // Resolve any control-flow primitives (while/case) synchronously
                // first: their internal re-entrant eval() deadlocks under an
                // async_eval outer pass (it waits cross-stream on a per-stream
                // completion event that the outer pass only signals at its end).
                // This syncs just the loop(s) and their dependency cone — which
                // a data-dependent loop forces to complete anyway — then the
                // async_eval below pipelines all independent/downstream work and
                // lets the next dispatch overlap.
                if (has_control_flow_) {
                    std::vector<mlx::core::array> cf_outputs = CollectControlFlowOutputs(outputs);
                    if (!cf_outputs.empty()) {
                        mlx::core::eval(cf_outputs);
                    }
                }
                mlx::core::async_eval(outputs);
            } else {
                mlx::core::eval(outputs);
            }
        } catch (const std::exception& e) {
            MPS_LOG_ERROR("MLX evaluation failed: %s\n", e.what());
            result.error_message = std::string("eval: ") + e.what();
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
