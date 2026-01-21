#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#import <string>
#import <unordered_map>

namespace jax_mps {

struct HloOp;

using TensorDict = NSMutableDictionary<NSString*, MPSGraphTensor*>*;
using OpHandler = MPSGraphTensor* (*)(MPSGraph*, TensorDict, const HloOp&, NSArray<NSNumber*>*);

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

// Helper to get tensor by name
inline MPSGraphTensor* GetTensor(TensorDict tensors, const std::string& name) {
    return tensors[[NSString stringWithUTF8String:name.c_str()]];
}

// Macro for registering ops - use in .mm files
#define REGISTER_OP(op_name, handler_fn) \
    static bool _reg_##op_name = ::jax_mps::OpRegistry::Register(#op_name, handler_fn)

// Convenience macro for simple binary ops
#define REGISTER_BINARY_OP(op_name, mps_method)                                                    \
    static MPSGraphTensor* Handle_##op_name(MPSGraph* g, TensorDict t, const ::jax_mps::HloOp& op, \
                                            NSArray<NSNumber*>*) {                                 \
        return [g mps_method##WithPrimaryTensor:GetTensor(t, op.inputs[0])                         \
                                secondaryTensor:GetTensor(t, op.inputs[1])                         \
                                           name:nil];                                              \
    }                                                                                              \
    REGISTER_OP(op_name, Handle_##op_name)

// Convenience macro for simple unary ops
#define REGISTER_UNARY_OP(op_name, mps_method)                                                     \
    static MPSGraphTensor* Handle_##op_name(MPSGraph* g, TensorDict t, const ::jax_mps::HloOp& op, \
                                            NSArray<NSNumber*>*) {                                 \
        return [g mps_method##WithTensor:GetTensor(t, op.inputs[0]) name:nil];                     \
    }                                                                                              \
    REGISTER_OP(op_name, Handle_##op_name)

// Map PJRT dtype to MPSDataType
// Returns MPSDataTypeInvalid (0) for unknown types - caller must check
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
        case 1:
            return MPSDataTypeBool;
        default:
            return MPSDataTypeInvalid;
    }
}

}  // namespace jax_mps
