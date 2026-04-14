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
// which lowers to a single mlx::core::softmax kernel. Matches softmax over
// any single reduction axis (trailing, leading, or middle; negative axes
// are normalized in the runtime handler).
void populateFuseSoftmaxPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
