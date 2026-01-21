// Unary operations: tanh, exp, log, negate, abs

#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/mps_executable.h"

namespace jax_mps {

REGISTER_UNARY_OP(tanh, tanh);
REGISTER_UNARY_OP(exp, exponent);
REGISTER_UNARY_OP(log, logarithm);
REGISTER_UNARY_OP(negate, negative);
REGISTER_UNARY_OP(abs, absolute);

}  // namespace jax_mps
