#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

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

// Handler signature: takes MLIR operation directly
using OpHandler = MPSGraphTensor* (*)(MPSGraph*, mlir::Operation*, ValueMap&);

// Global op registry - ops register themselves at static init time
class OpRegistry {
public:
    static bool Register(const char* name, OpHandler handler) {
        GetMutableHandlers()[name] = handler;
        return true;
    }

    static OpHandler Find(const std::string& name) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(name);
        return it != handlers.end() ? it->second : nullptr;
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
class CustomCallRegistry {
public:
    static bool Register(const char* target, OpHandler handler) {
        GetMutableHandlers()[target] = handler;
        return true;
    }

    static OpHandler Find(const std::string& target) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(target);
        return it != handlers.end() ? it->second : nullptr;
    }

private:
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }
};

// Native op handler: encodes work to a command buffer, returns output MTLBuffer.
// Inputs are MTLBuffers corresponding to op->getOperands().
using NativeOpHandler = id<MTLBuffer> (*)(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                          mlir::Operation* op,
                                          const std::vector<id<MTLBuffer>>& inputs);

// Global native op registry - native ops register themselves at static init time
class NativeOpRegistry {
public:
    static bool Register(const char* name, NativeOpHandler handler) {
        GetMutableHandlers()[name] = handler;
        return true;
    }

    static NativeOpHandler Find(const std::string& name) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(name);
        return it != handlers.end() ? it->second : nullptr;
    }

    // Returns set of all registered native operation names
    static std::unordered_set<std::string> GetRegisteredOps() {
        std::unordered_set<std::string> ops;
        for (const auto& pair : GetMutableHandlers()) {
            ops.insert(pair.first);
        }
        return ops;
    }

private:
    static std::unordered_map<std::string, NativeOpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, NativeOpHandler> handlers;
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

// Macro for registering ops - use in .mm files
// Use the full MLIR op name (e.g., "stablehlo.add")
#define REGISTER_MPS_OP(mlir_op_name, handler_fn) \
    static bool _reg_##handler_fn = ::jax_mps::OpRegistry::Register(mlir_op_name, handler_fn)

// Convenience macro for simple binary ops
#define REGISTER_MLIR_BINARY_OP(mlir_op_name, mps_method, reg_suffix)                 \
    static MPSGraphTensor* Handle_mlir_##reg_suffix(MPSGraph* g, mlir::Operation* op, \
                                                    ::jax_mps::ValueMap& values) {    \
        MPSGraphTensor* lhs = GetInputTensor(values, op, 0);                          \
        MPSGraphTensor* rhs = GetInputTensor(values, op, 1);                          \
        if (!lhs || !rhs)                                                             \
            return nullptr;                                                           \
        return [g mps_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];    \
    }                                                                                 \
    REGISTER_MPS_OP(mlir_op_name, Handle_mlir_##reg_suffix)

// Convenience macro for simple unary ops
#define REGISTER_MLIR_UNARY_OP(mlir_op_name, mps_method, reg_suffix)                  \
    static MPSGraphTensor* Handle_mlir_##reg_suffix(MPSGraph* g, mlir::Operation* op, \
                                                    ::jax_mps::ValueMap& values) {    \
        MPSGraphTensor* input = GetInputTensor(values, op, 0);                        \
        if (!input)                                                                   \
            return nullptr;                                                           \
        return [g mps_method##WithTensor:input name:nil];                             \
    }                                                                                 \
    REGISTER_MPS_OP(mlir_op_name, Handle_mlir_##reg_suffix)

// Macro for registering custom call targets
#define REGISTER_CUSTOM_CALL_TARGET(target_name, handler_fn) \
    static bool _cc_reg_##handler_fn =                       \
        ::jax_mps::CustomCallRegistry::Register(target_name, handler_fn)

// Convenience macro for simple unary custom call targets
#define REGISTER_CUSTOM_CALL_UNARY_OP(target_name, mps_method, reg_suffix)          \
    static MPSGraphTensor* Handle_cc_##reg_suffix(MPSGraph* g, mlir::Operation* op, \
                                                  ::jax_mps::ValueMap& values) {    \
        MPSGraphTensor* input = GetInputTensor(values, op, 0);                      \
        if (!input)                                                                 \
            return nullptr;                                                         \
        return [g mps_method##WithTensor:input name:nil];                           \
    }                                                                               \
    static bool _cc_reg_##reg_suffix =                                              \
        ::jax_mps::CustomCallRegistry::Register(target_name, Handle_cc_##reg_suffix)

// Macro for registering native ops
#define REGISTER_NATIVE_MPS_OP(mlir_op_name, handler_fn) \
    static bool _native_reg_##handler_fn =               \
        ::jax_mps::NativeOpRegistry::Register(mlir_op_name, handler_fn)

}  // namespace jax_mps
