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

// Constant creation - NOT YET IMPLEMENTED
// Returns nullptr to trigger a clear error message.
// TODO: Parse actual constant values from StableHLO bytecode.
// This requires extracting the constant data from the MLIR DenseElementsAttr.
static MPSGraphTensor* Handle_constant(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>* shape) {
    // Returning nullptr will cause Execute to fail with:
    // "Operation 'constant' handler returned null"
    // This is correct behavior - we don't support constants yet and shouldn't
    // silently return wrong values (zeros).
    //
    // Note: Many JAX programs use constants. Until this is implemented,
    // programs using constants will fail with a clear error.
    return nullptr;
}
REGISTER_OP(constant, Handle_constant);

}  // namespace jax_mps
