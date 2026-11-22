#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers the RewritePattern that rewrites hand-rolled scaled dot-product
// attention
//
//     s   = dot_general(q, k)            // contract head dim -> (B, N, T, S)
//     s'  = s * c  |  s / c  |  s        // optional constant scale
//     p   = custom_call @mps.softmax(s') // axis = 3 (keys), from fuse_softmax
//     out = dot_general(p, v)            // contract S (either operand order)
//
// into
//
//     stablehlo.custom_call @mps.sdpa(q', k', v', true) {scale = c | 1/c | 1}
//
// which lowers to mlx::core::fast::scaled_dot_product_attention. q/k/v may use
// any batching layout (e.g. BNTH or BTNH); transposes into BNTH are inserted
// as needed. Every intermediate must have a single use, so the pattern never
// fires when the scores or weights are needed elsewhere (e.g. as autodiff
// residuals).
void populateFuseSdpaPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
