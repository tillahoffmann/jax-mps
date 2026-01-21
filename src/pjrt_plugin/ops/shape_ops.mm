// Shape operations: broadcast, reshape, convert

#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/mps_executable.h"

namespace jax_mps {

static MPSGraphTensor* Handle_broadcast(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>* shape) {
    return [g broadcastTensor:GetTensor(t, op.inputs[0]) toShape:shape name:nil];
}
REGISTER_OP(broadcast, Handle_broadcast);
REGISTER_OP(broadcast_in_dim, Handle_broadcast);

static MPSGraphTensor* Handle_reshape(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>* shape) {
    return [g reshapeTensor:GetTensor(t, op.inputs[0]) withShape:shape name:nil];
}
REGISTER_OP(reshape, Handle_reshape);

static MPSGraphTensor* Handle_convert(MPSGraph* g, TensorDict t, const HloOp& op, NSArray<NSNumber*>*) {
    return [g castTensor:GetTensor(t, op.inputs[0]) toType:PjrtDtypeToMps(op.dtype) name:nil];
}
REGISTER_OP(convert, Handle_convert);

}  // namespace jax_mps
