// Shared types, utilities, and registration declarations for op handlers.
#pragma once

#include <mlx/mlx.h>

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/type_utils.h"

namespace jax_mps {

// Value map type using void* as key (from mlir::Value's opaque pointer)
using ValueMap = std::unordered_map<void*, mlx::core::array>;

// Execution context passed to handlers
struct ExecContext {
    mlir::ModuleOp module;
    bool inside_compile = false;  // true when running inside mlx::core::compile()
};

// Exception thrown when an op is incompatible with mlx::core::compile() tracing.
class CompileIncompatibleError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Op handler function type
using OpHandler =
    std::function<bool(mlir::Operation*, ValueMap&, std::vector<mlx::core::array>&, ExecContext&)>;

// Function pointer types for trivial op handler factories
using UnaryMlxFn = mlx::core::array (*)(const mlx::core::array&, mlx::core::StreamOrDevice);
using BinaryMlxFn = mlx::core::array (*)(const mlx::core::array&, const mlx::core::array&,
                                         mlx::core::StreamOrDevice);

// --- Utility functions (defined in mlx_executable.cc) ---

void* ToKey(mlir::Value v);

std::optional<std::reference_wrapper<mlx::core::array>> GetValue(ValueMap& values, mlir::Value v);

mlx::core::Dtype MlirTypeToMlxDtype(mlir::Type type);

mlx::core::Shape GetShape(mlir::RankedTensorType type);

inline size_t GetDtypeSize(mlx::core::Dtype dtype) {
    return GetMlxDtypeSize(dtype);
}

std::optional<mlx::core::array> CreateArrayWithTypedPtr(const void* data,
                                                        const mlx::core::Shape& shape,
                                                        mlx::core::Dtype dtype);

std::optional<mlx::core::array> CreateArrayFromDenseAttr(mlir::DenseElementsAttr attr);

// --- Factory functions (defined in mlx_executable.cc) ---

OpHandler MakeUnaryHandler(const char* opName, UnaryMlxFn fn);
OpHandler MakeBinaryHandler(const char* opName, BinaryMlxFn fn);
OpHandler MakeLogicalShiftHandler(const char* opName, BinaryMlxFn shiftFn);

// --- Cross-TU functions (defined in mlx_executable.cc, called by control_flow.cc) ---

bool ExecuteRegion(mlir::Region& region, std::vector<mlx::core::array>& args,
                   std::vector<mlx::core::array>& results, ExecContext& ctx,
                   const ValueMap* parentValues = nullptr);

bool ExecuteFunction(mlir::func::FuncOp func, const std::vector<mlx::core::array>& inputs,
                     std::vector<mlx::core::array>& outputs, ExecContext& ctx);

// --- Shared enums/helpers (defined in gather_scatter.cc, used by reduction.cc) ---

enum class ScatterType { Update, Add, Sub, Mul, Min, Max, Unknown };

ScatterType DetectScatterType(mlir::Region& body);

std::optional<mlx::core::array> ApplyScatter(ScatterType scatterType,
                                             const mlx::core::array& operand,
                                             const std::vector<mlx::core::array>& idxVec,
                                             const mlx::core::array& updates,
                                             const std::vector<int>& axes);

// --- Registration functions (each defined in its own .cc file) ---

void RegisterArithmeticHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterShapeHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterSliceHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterGatherScatterHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterReductionHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterLinalgHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterControlFlowHandlers(std::unordered_map<std::string, OpHandler>& handlers);
void RegisterSortFftComplexHandlers(std::unordered_map<std::string, OpHandler>& handlers);

}  // namespace jax_mps
