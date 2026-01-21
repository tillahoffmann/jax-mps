// Unary operations: tanh, exp, log, negate, abs
// Also includes constant creation

#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/mps_executable.h"

namespace jax_mps {

REGISTER_UNARY_OP(tanh, tanh);
REGISTER_UNARY_OP(exp, exponent);
REGISTER_UNARY_OP(log, logarithm);
REGISTER_UNARY_OP(negate, negative);
REGISTER_UNARY_OP(abs, absolute);

// Constant creation - creates a zero tensor of the given shape
// TODO: Parse actual constant values from StableHLO
static MPSGraphTensor* Handle_constant(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>* shape) {
    MPSDataType dtype = PjrtDtypeToMps(op.dtype);
    // Create a constant tensor filled with zeros
    // For scalar constants (empty shape), create a scalar
    if (shape.count == 0) {
        return [g constantWithScalar:0.0 dataType:dtype];
    }
    return [g constantWithScalar:0.0 shape:shape dataType:dtype];
}
REGISTER_OP(constant, Handle_constant);

}  // namespace jax_mps
