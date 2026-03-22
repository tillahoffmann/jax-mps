// Shared types, utilities, and registration declarations for op handlers.
#pragma once

#include <mlx/mlx.h>

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

// --- Convenience helpers ---

// Like GetValue but logs an error and returns nullptr on failure.
inline mlx::core::array* RequireValue(ValueMap& values, mlir::Value v, const char* opName) {
    auto it = values.find(ToKey(v));
    if (it == values.end()) {
        MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
        return nullptr;
    }
    return &it->second;
}

// Cast mlir::Operation to a specific op type, logging on failure.
template <typename OpT>
OpT CastOp(mlir::Operation* op, const char* opName) {
    auto result = mlir::dyn_cast<OpT>(op);
    if (!result) {
        MPS_LOG_ERROR("%s: failed to cast\n", opName);
    }
    return result;
}

// Convert a range of int64_t-like values to std::vector<int>.
template <typename Range>
std::vector<int> ToIntVec(const Range& dims) {
    std::vector<int> result;
    for (auto d : dims)
        result.push_back(static_cast<int>(d));
    return result;
}

// Convert a range to mlx::core::Shape (SmallVector<int32_t>).
template <typename Range>
mlx::core::Shape ToShape(const Range& dims) {
    mlx::core::Shape result;
    for (auto d : dims)
        result.push_back(static_cast<int32_t>(d));
    return result;
}

// Extract the result shape from op->getResult(0), logging on failure.
inline std::optional<mlx::core::Shape> GetResultShape(mlir::Operation* op, const char* opName) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("%s: result type is not RankedTensorType\n", opName);
        return std::nullopt;
    }
    return GetShape(resultType);
}

// Extract the result dtype from op->getResult(0), logging on failure.
inline std::optional<mlx::core::Dtype> GetResultDtype(mlir::Operation* op, const char* opName) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        MPS_LOG_ERROR("%s: result type is not RankedTensorType\n", opName);
        return std::nullopt;
    }
    return MlirTypeToMlxDtype(resultType.getElementType());
}

// Check if a permutation is the identity (no transpose needed).
inline bool IsIdentityPermutation(const std::vector<int>& perm) {
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i))
            return false;
    }
    return true;
}

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

// --- Shared sort utilities (defined in sort_fft_complex.cc) ---

// Compute top-k values and indices along the last axis using ascending argsort.
// Returns (values, indices) where values are in descending order.
std::pair<mlx::core::array, mlx::core::array> TopKImpl(const mlx::core::array& input, int k);

// Reverse an array along a given axis using gather with reversed indices.
mlx::core::array ReverseAxis(const mlx::core::array& a, int axis);

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
