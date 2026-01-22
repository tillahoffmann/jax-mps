#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <string>
#include <unordered_map>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/type_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

// Maps mlir::Value (via opaque pointer) to MPSGraphTensor*
using ValueMap = std::unordered_map<void*, MPSGraphTensor*>;

// Handler signature: takes MLIR operation directly
using OpHandler = MPSGraphTensor* (*)(MPSGraph*, mlir::Operation*, ValueMap&, NSArray<NSNumber*>*);

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

// Macro for registering ops - use in .mm files
// Use the full MLIR op name (e.g., "stablehlo.add")
#define REGISTER_MPS_OP(mlir_op_name, handler_fn) \
    static bool _reg_##handler_fn = ::jax_mps::OpRegistry::Register(mlir_op_name, handler_fn)

// Convenience macro for simple binary ops
#define REGISTER_MLIR_BINARY_OP(mlir_op_name, mps_method, reg_suffix)                         \
    static MPSGraphTensor* Handle_mlir_##reg_suffix(                                          \
        MPSGraph* g, mlir::Operation* op, ::jax_mps::ValueMap& values, NSArray<NSNumber*>*) { \
        MPSGraphTensor* lhs = GetInputTensor(values, op, 0);                                  \
        MPSGraphTensor* rhs = GetInputTensor(values, op, 1);                                  \
        if (!lhs || !rhs)                                                                     \
            return nullptr;                                                                   \
        return [g mps_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];            \
    }                                                                                         \
    REGISTER_MPS_OP(mlir_op_name, Handle_mlir_##reg_suffix)

// Convenience macro for simple unary ops
#define REGISTER_MLIR_UNARY_OP(mlir_op_name, mps_method, reg_suffix)                          \
    static MPSGraphTensor* Handle_mlir_##reg_suffix(                                          \
        MPSGraph* g, mlir::Operation* op, ::jax_mps::ValueMap& values, NSArray<NSNumber*>*) { \
        MPSGraphTensor* input = GetInputTensor(values, op, 0);                                \
        if (!input)                                                                           \
            return nullptr;                                                                   \
        return [g mps_method##WithTensor:input name:nil];                                     \
    }                                                                                         \
    REGISTER_MPS_OP(mlir_op_name, Handle_mlir_##reg_suffix)

}  // namespace jax_mps
