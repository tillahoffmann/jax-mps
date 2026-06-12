#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers the RewritePattern that rewrites the standard affine
// LayerNorm decomposition (mean / variance over the trailing axis, then
// `(x - mean) * rsqrt(var + eps) * weight + bias`) into
//     stablehlo.custom_call @mps.layer_norm(x, weight, bias) {eps}
// which lowers to a single mlx::core::fast::layer_norm at execution time.
//
// Matches the form emitted by flax.linen.LayerNorm / nnx.LayerNorm with
// use_scale=use_bias=True, where the variance is computed as
// max(0, E[x^2] - E[x]^2). Only fires when normalization is over the last
// axis and weight/bias are 1-D of size x.shape[-1] (the contract of
// mlx::core::fast::layer_norm).
void populateFuseLayerNormPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
