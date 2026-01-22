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

// Convert MLIR type to MPSDataType
inline MPSDataType MlirTypeToMps(mlir::Type type) {
    if (type.isF32())
        return MPSDataTypeFloat32;
    if (type.isF16())
        return MPSDataTypeFloat16;
    if (type.isBF16())
        return MPSDataTypeBFloat16;

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();

        if (width == 1)
            return MPSDataTypeBool;
        if (width == 8)
            return isUnsigned ? MPSDataTypeUInt8 : MPSDataTypeInt8;
        if (width == 16)
            return isUnsigned ? MPSDataTypeUInt16 : MPSDataTypeInt16;
        if (width == 32)
            return isUnsigned ? MPSDataTypeUInt32 : MPSDataTypeInt32;
        if (width == 64)
            return isUnsigned ? MPSDataTypeUInt64 : MPSDataTypeInt64;
    }

    return MPSDataTypeInvalid;
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

// Map PJRT dtype to MPSDataType (for compatibility with buffer operations)
inline MPSDataType PjrtDtypeToMps(int dtype) {
    switch (dtype) {
        case 11:
            return MPSDataTypeFloat32;
        case 10:
            return MPSDataTypeFloat16;
        case 16:
            return MPSDataTypeBFloat16;
        case 4:
            return MPSDataTypeInt32;
        case 5:
            return MPSDataTypeInt64;
        case 8:
            return MPSDataTypeUInt32;
        case 9:
            return MPSDataTypeUInt64;
        case 2:
            return MPSDataTypeInt8;
        case 6:
            return MPSDataTypeUInt8;
        case 3:
            return MPSDataTypeInt16;
        case 7:
            return MPSDataTypeUInt16;
        case 1:
            return MPSDataTypeBool;
        default:
            return MPSDataTypeInvalid;
    }
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
