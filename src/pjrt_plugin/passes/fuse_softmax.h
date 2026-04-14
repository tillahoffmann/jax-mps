#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers the RewritePattern that rewrites the canonical numerically-stable
// softmax pattern
//
//     max  = reduce_max(x, axis=-1)           // optional maximum(-inf, max)
//     exp  = exp(x - broadcast(max))
//     sum  = reduce_sum(exp, axis=-1)
//     out  = exp / broadcast(sum)
//
// into
//
//     stablehlo.custom_call @mps.softmax(x)
//
// which lowers to a single mlx::core::softmax kernel. Currently matches
// trailing-axis softmax only (by far the common case).
void populateFuseSoftmaxPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
