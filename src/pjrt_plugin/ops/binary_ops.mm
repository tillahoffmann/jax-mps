// Binary operations: add, subtract, multiply, divide, maximum, minimum, dot

#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_BINARY_OP(add, addition);
REGISTER_BINARY_OP(subtract, subtraction);
REGISTER_BINARY_OP(multiply, multiplication);
REGISTER_BINARY_OP(divide, division);
REGISTER_BINARY_OP(maximum, maximum);
REGISTER_BINARY_OP(minimum, minimum);

// Matrix multiplication
static MPSGraphTensor* Handle_dot(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>*) {
    return [g matrixMultiplicationWithPrimaryTensor:GetTensor(t, op.inputs[0])
                                    secondaryTensor:GetTensor(t, op.inputs[1])
                                               name:nil];
}
REGISTER_OP(dot, Handle_dot);
REGISTER_OP(dot_general, Handle_dot);

}  // namespace jax_mps
