// Bitwise operations: and, or, xor, shift_left, shift_right_logical
// Also includes concatenate which is needed for RNG

#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Bitwise AND
REGISTER_BINARY_OP(and, bitwiseAND);

// Bitwise OR
REGISTER_BINARY_OP(or, bitwiseOR);

// Bitwise XOR
REGISTER_BINARY_OP(xor, bitwiseXOR);

// Logical left shift
REGISTER_BINARY_OP(shift_left, bitwiseLeftShift);

// Logical right shift
REGISTER_BINARY_OP(shift_right_logical, bitwiseRightShift);

// Concatenate - joins tensors along a dimension
static MPSGraphTensor* Handle_concatenate(MPSGraph* g, TensorDict t, const HloOp& op,
                                          NSArray<NSNumber*>*) {
    // Gather all input tensors
    NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray array];
    for (const auto& input_name : op.inputs) {
        MPSGraphTensor* tensor = GetTensor(t, input_name);
        if (tensor) {
            [input_tensors addObject:tensor];
        }
    }

    if (input_tensors.count == 0) {
        NSLog(@"ERROR: Concatenate operation has no valid inputs");
        return nullptr;
    }

    // Use the concatenate dimension from the op
    NSInteger dimension = static_cast<NSInteger>(op.concatenate_dim);

    return [g concatTensors:input_tensors dimension:dimension name:nil];
}
REGISTER_OP(concatenate, Handle_concatenate);

}  // namespace jax_mps
