// Unary operations: tanh, exp, log, negate, abs
// Also includes constant creation

#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_UNARY_OP(tanh, tanh);
REGISTER_UNARY_OP(exp, exponent);
REGISTER_UNARY_OP(log, logarithm);
REGISTER_UNARY_OP(negate, negative);
REGISTER_UNARY_OP(abs, absolute);

// Constant creation - creates a constant tensor from parsed StableHLO data
static MPSGraphTensor* Handle_constant(MPSGraph* g, TensorDict t, const HloOp& op,
                                       NSArray<NSNumber*>* shape) {
    MPSDataType dtype = PjrtDtypeToMps(op.dtype);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype %d for constant operation", op.dtype);
        return nullptr;
    }

    // Handle scalar/splat constants (most common case, e.g., 0.0 for relu)
    if (op.is_scalar_constant) {
        if (shape.count == 0) {
            // True scalar
            return [g constantWithScalar:op.constant_scalar dataType:dtype];
        } else {
            // Splat to shape
            return [g constantWithScalar:op.constant_scalar shape:shape dataType:dtype];
        }
    }

    // Handle dense array constants
    if (!op.constant_data.empty()) {
        // Create NSData from the constant values
        NSData* data = [NSData dataWithBytes:op.constant_data.data()
                                      length:op.constant_data.size() * sizeof(float)];
        return [g constantWithData:data shape:shape dataType:dtype];
    }

    // No constant data available - this shouldn't happen if parser worked correctly
    NSLog(@"ERROR: Constant operation has no data (is_scalar=%d, data_size=%zu)",
          op.is_scalar_constant, op.constant_data.size());
    return nullptr;
}
REGISTER_OP(constant, Handle_constant);

}  // namespace jax_mps
