#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers the RewritePattern that rewrites the standard RMSNorm
// decomposition (`x * rsqrt(mean(x^2) + eps) * weight`, normalized over the
// trailing axis) into
//     stablehlo.custom_call @mps.rms_norm(x, weight) {eps}
// which lowers to a single mlx::core::fast::rms_norm at execution time.
//
// Matches the form emitted by flax.linen.RMSNorm / nnx.RMSNorm with
// use_scale=True. Flax shares its normalization code with LayerNorm, so it
// emits a `subtract(x, broadcast(0))` centering no-op (mean is the constant 0
// for RMSNorm); the matcher tolerates that. Only fires over the last axis with
// a 1-D weight of size x.shape[-1] (the kernel's contract).
void populateFuseRmsNormPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
