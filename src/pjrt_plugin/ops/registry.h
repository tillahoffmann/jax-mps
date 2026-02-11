#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/type_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

// Maps mlir::Value (via opaque pointer) to MPSGraphTensor*
using ValueMap = std::unordered_map<void*, MPSGraphTensor*>;

// Result type for op handlers - can be an error or success
struct ProcessResult {
    std::string error;
    std::vector<mlir::Value> return_values;

    bool ok() const {
        return error.empty();
    }
    static ProcessResult Error(const std::string& msg) {
        ProcessResult r;
        r.error = msg;
        return r;
    }
};

// ---------------------------------------------------------------------------
// Handler signatures
// ---------------------------------------------------------------------------

// Graph handler: operates on MPSGraph tensors, builds computation graph
using GraphOpHandler = ProcessResult (*)(MPSGraph*, mlir::Operation*, ValueMap&);

// Native handler: operates directly on Metal buffers via command buffer encoding
// Returns the output buffer (single-output ops only for now)
using NativeOpHandler = id<MTLBuffer> (*)(id<MTLDevice>, id<MTLCommandBuffer>, mlir::Operation*,
                                          const std::vector<id<MTLBuffer>>&);

// ---------------------------------------------------------------------------
// Tagged handler: unified representation for both execution models
// ---------------------------------------------------------------------------

struct OpHandler {
    enum class Kind {
        GRAPH,   // Normal graph-based ops using MPSGraph
        NATIVE,  // Native MPS kernel ops (e.g., Cholesky)
        SPECIAL  // Ops handled specially (control flow, custom_call)
    } kind;

    GraphOpHandler graph_handler = nullptr;
    NativeOpHandler native_handler = nullptr;

    static OpHandler Graph(GraphOpHandler h) {
        OpHandler handler;
        handler.kind = Kind::GRAPH;
        handler.graph_handler = h;
        return handler;
    }

    static OpHandler Native(NativeOpHandler h) {
        OpHandler handler;
        handler.kind = Kind::NATIVE;
        handler.native_handler = h;
        return handler;
    }

    static OpHandler Special() {
        OpHandler handler;
        handler.kind = Kind::SPECIAL;
        return handler;
    }

    bool is_native() const {
        return kind == Kind::NATIVE;
    }
    bool is_graph() const {
        return kind == Kind::GRAPH;
    }
    bool is_special() const {
        return kind == Kind::SPECIAL;
    }
};

// Global op registry - ops register themselves at static init time
// NOTE: Do not create additional registries. Use this single registry for all ops.
class OpRegistry {
public:
    static bool Register(const char* name, OpHandler handler) {
        GetMutableHandlers()[name] = handler;
        return true;
    }

    // Returns pointer to handler if found, nullptr otherwise
    static const OpHandler* Find(const std::string& name) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(name);
        return it != handlers.end() ? &it->second : nullptr;
    }

    // Returns comma-separated list of all registered operation names
    static std::string ListRegistered() {
        auto& handlers = GetMutableHandlers();
        std::string result;
        bool first = true;
        for (const auto& pair : handlers) {
            if (!first)
                result += ", ";
            result += pair.first;
            first = false;
        }
        return result;
    }

    // Returns set of all registered operation names
    static std::unordered_set<std::string> GetRegisteredOps() {
        std::unordered_set<std::string> ops;
        for (const auto& pair : GetMutableHandlers()) {
            ops.insert(pair.first);
        }
        return ops;
    }

private:
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }
};

// Global custom-call-target registry - custom call targets register themselves at static init time
// NOTE: Do not create additional registries. Use this single registry for all custom calls.
class CustomCallRegistry {
public:
    static bool Register(const char* target, OpHandler handler) {
        GetMutableHandlers()[target] = handler;
        return true;
    }

    // Returns pointer to handler if found, nullptr otherwise
    static const OpHandler* Find(const std::string& target) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(target);
        return it != handlers.end() ? &it->second : nullptr;
    }

private:
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }
};

// Helper to get tensor by mlir::Value
inline MPSGraphTensor* GetTensor(ValueMap& values, mlir::Value value) {
    auto it = values.find(value.getAsOpaquePointer());
    return it != values.end() ? it->second : nullptr;
}

// Helper to get input tensor at index from an operation
inline MPSGraphTensor* GetInputTensor(ValueMap& values, mlir::Operation* op, unsigned index) {
    if (index >= op->getNumOperands()) {
        return nullptr;
    }
    return GetTensor(values, op->getOperand(index));
}

// Helper to set output tensor for an operation's result
inline void SetOutputTensor(ValueMap& values, mlir::Operation* op, MPSGraphTensor* tensor,
                            unsigned index = 0) {
    if (index < op->getNumResults()) {
        values[op->getResult(index).getAsOpaquePointer()] = tensor;
    }
}

// Get MPSDataType from operation's result type
inline MPSDataType GetResultMpsType(mlir::Operation* op, unsigned index = 0) {
    if (index >= op->getNumResults()) {
        return MPSDataTypeInvalid;
    }
    auto resultType = op->getResult(index).getType();
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        return MlirTypeToMps(tensorType.getElementType());
    }
    return MPSDataTypeInvalid;
}

// Get element type from an MLIR type
inline mlir::Type GetElementType(mlir::Type type) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        return tensorType.getElementType();
    }
    return type;
}

// Get output shape from operation's result type as NSArray
inline NSArray<NSNumber*>* GetOutputShape(mlir::Operation* op, unsigned resultIndex = 0) {
    if (resultIndex >= op->getNumResults()) {
        return nil;
    }
    auto resultType = op->getResult(resultIndex).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType) {
        return nil;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (int64_t dim : tensorType.getShape()) {
        [shape addObject:@(dim)];
    }
    return shape;
}

// Helper to cast tensor to Int32 if needed (for indices)
inline MPSGraphTensor* EnsureInt32(MPSGraph* g, MPSGraphTensor* tensor) {
    if (tensor.dataType != MPSDataTypeInt32) {
        return [g castTensor:tensor toType:MPSDataTypeInt32 name:nil];
    }
    return tensor;
}

// Helper to finalize a simple op: check result, set output, return success/error
// Use this at the end of handlers to reduce boilerplate:
//   return Result(values, op, result, "my_op");
inline ProcessResult Result(ValueMap& values, mlir::Operation* op, MPSGraphTensor* result,
                            const char* op_name) {
    if (!result)
        return ProcessResult::Error(std::string(op_name) + ": handler returned null");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}

// ---------------------------------------------------------------------------
// Registration macros
// ---------------------------------------------------------------------------

// Macro for registering graph ops - use in .mm files
// Use the full MLIR op name (e.g., "stablehlo.add")
#define REGISTER_MPS_OP(mlir_op_name, handler_fn) \
    static bool _reg_##handler_fn =               \
        ::jax_mps::OpRegistry::Register(mlir_op_name, ::jax_mps::OpHandler::Graph(handler_fn))

// Macro for registering native ops (ops that use Metal buffers directly)
#define REGISTER_NATIVE_MPS_OP(mlir_op_name, handler_fn) \
    static bool _reg_##handler_fn =                      \
        ::jax_mps::OpRegistry::Register(mlir_op_name, ::jax_mps::OpHandler::Native(handler_fn))

// Macro for registering special ops (handled specially in mps_executable.mm)
#define REGISTER_SPECIAL_MPS_OP(mlir_op_name, unique_suffix) \
    static bool _reg_special_##unique_suffix =               \
        ::jax_mps::OpRegistry::Register(mlir_op_name, ::jax_mps::OpHandler::Special())

// Convenience macro for simple binary ops
#define REGISTER_MLIR_BINARY_OP(mlir_op_name, mps_method, reg_suffix)                             \
    static ::jax_mps::ProcessResult HandleMlir##reg_suffix(MPSGraph* g, mlir::Operation* op,      \
                                                           ::jax_mps::ValueMap& values) {         \
        MPSGraphTensor* lhs = GetInputTensor(values, op, 0);                                      \
        MPSGraphTensor* rhs = GetInputTensor(values, op, 1);                                      \
        if (!lhs || !rhs)                                                                         \
            return ::jax_mps::ProcessResult::Error(#reg_suffix ": missing input tensor");         \
        MPSGraphTensor* out = [g mps_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil]; \
        return Result(values, op, out, #reg_suffix);                                              \
    }                                                                                             \
    REGISTER_MPS_OP(mlir_op_name, HandleMlir##reg_suffix)

// Convenience macro for simple unary ops
#define REGISTER_MLIR_UNARY_OP(mlir_op_name, mps_method, reg_suffix)                         \
    static ::jax_mps::ProcessResult HandleMlir##reg_suffix(MPSGraph* g, mlir::Operation* op, \
                                                           ::jax_mps::ValueMap& values) {    \
        MPSGraphTensor* input = GetInputTensor(values, op, 0);                               \
        if (!input)                                                                          \
            return ::jax_mps::ProcessResult::Error(#reg_suffix ": missing input tensor");    \
        MPSGraphTensor* out = [g mps_method##WithTensor:input name:nil];                     \
        return Result(values, op, out, #reg_suffix);                                         \
    }                                                                                        \
    REGISTER_MPS_OP(mlir_op_name, HandleMlir##reg_suffix)

// Macro for registering custom call targets (graph-based)
// Use unique_suffix to allow registering the same handler for multiple targets
#define REGISTER_CUSTOM_CALL(target_name, handler_fn, unique_suffix)               \
    static bool _cc_reg_##unique_suffix = ::jax_mps::CustomCallRegistry::Register( \
        target_name, ::jax_mps::OpHandler::Graph(handler_fn))

// Convenience macro for simple unary custom call targets
#define REGISTER_CUSTOM_CALL_UNARY_OP(target_name, mps_method, reg_suffix)                 \
    static ::jax_mps::ProcessResult HandleCc##reg_suffix(MPSGraph* g, mlir::Operation* op, \
                                                         ::jax_mps::ValueMap& values) {    \
        MPSGraphTensor* input = GetInputTensor(values, op, 0);                             \
        if (!input)                                                                        \
            return ::jax_mps::ProcessResult::Error(#reg_suffix ": missing input tensor");  \
        MPSGraphTensor* out = [g mps_method##WithTensor:input name:nil];                   \
        return Result(values, op, out, #reg_suffix);                                       \
    }                                                                                      \
    REGISTER_CUSTOM_CALL(target_name, HandleCc##reg_suffix, reg_suffix)

}  // namespace jax_mps
