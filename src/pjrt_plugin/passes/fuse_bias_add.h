#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers the RewritePattern that rewrites
//     add(dot_general(x, w), broadcast_in_dim(bias))
// into
//     stablehlo.custom_call @mps.addmm(x, w, bias)
// which lowers to a single mlx::core::addmm at execution time.
void populateFuseBiasAddPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
