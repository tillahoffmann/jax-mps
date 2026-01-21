// Shape operations: broadcast, reshape, convert

#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static MPSGraphTensor* Handle_broadcast(MPSGraph* g, TensorDict t, const HloOp& op,
                                        NSArray<NSNumber*>* shape) {
    return [g broadcastTensor:GetTensor(t, op.inputs[0]) toShape:shape name:nil];
}
REGISTER_OP(broadcast, Handle_broadcast);

// broadcast_in_dim needs special handling for dimension mapping
// When broadcasting (5,) to (1,5) with broadcast_dimensions=(1,), we need to:
// 1. Reshape (5,) to (1,5) by inserting dims according to broadcast_dimensions
// 2. Then broadcast to final shape
static MPSGraphTensor* Handle_broadcast_in_dim(MPSGraph* g, TensorDict t, const HloOp& op,
                                               NSArray<NSNumber*>* outputShape) {
    MPSGraphTensor* input = GetTensor(t, op.inputs[0]);

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;
    NSUInteger outputRank = outputShape.count;

    // If broadcast_dims is empty, just broadcast directly
    if (op.broadcast_dims.empty()) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // If ranks already match, just broadcast
    if (inputRank == outputRank) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // Build intermediate shape: start with all 1s, then fill in from broadcast_dims
    NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
    for (NSUInteger i = 0; i < outputRank; i++) {
        [intermediateShape addObject:@1];
    }

    // Map input dimensions to output dimensions according to broadcast_dims
    for (size_t i = 0; i < op.broadcast_dims.size() && i < inputRank; i++) {
        int64_t outDim = op.broadcast_dims[i];
        if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
            intermediateShape[outDim] = inputShape[i];
        }
    }

    // Reshape input to intermediate shape (same rank as output)
    MPSGraphTensor* reshaped = [g reshapeTensor:input withShape:intermediateShape name:nil];

    // Now broadcast to final output shape
    return [g broadcastTensor:reshaped toShape:outputShape name:nil];
}
REGISTER_OP(broadcast_in_dim, Handle_broadcast_in_dim);

static MPSGraphTensor* Handle_reshape(MPSGraph* g, TensorDict t, const HloOp& op,
                                      NSArray<NSNumber*>* shape) {
    return [g reshapeTensor:GetTensor(t, op.inputs[0]) withShape:shape name:nil];
}
REGISTER_OP(reshape, Handle_reshape);

static MPSGraphTensor* Handle_convert(MPSGraph* g, TensorDict t, const HloOp& op,
                                      NSArray<NSNumber*>*) {
    return [g castTensor:GetTensor(t, op.inputs[0]) toType:PjrtDtypeToMps(op.dtype) name:nil];
}
REGISTER_OP(convert, Handle_convert);

}  // namespace jax_mps
