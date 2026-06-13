#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mps {

// Registers canonicalization patterns that fold broadcasted arithmetic
// identities the upstream StableHLO aggressive-simplification pass leaves in
// place (it folds the scalar forms but not `broadcast_in_dim(splat)` ones):
//
//   add(x, broadcast(splat 0))       -> x   (either operand order)
//   subtract(x, broadcast(splat 0))  -> x   (RHS-zero ONLY; 0 - x is negation)
//   multiply(x, broadcast(splat 1))  -> x   (either operand order)
//
// Intentionally FLOAT-only: the splat constant must be a float type. The
// motivating cases (flax norm decompositions) are all float, and restricting
// to float keeps this orthogonal to the integer/shape canonicalization the
// upstream simplification + folder passes already handle. Extend to integer
// splats only if a concrete case needs it.
//
// Every fold is guarded on result-shape equality with x: when the splat
// operand is the larger one (x is being broadcast up), the op is a reshape of
// x, not an identity, and we must NOT fold. These are true algebraic
// identities under that guard, so unlike the fusion matchers they cannot
// change numerics. Running them before the mps.* fusion patterns lets the norm
// matchers see clean input (e.g. flax RMSNorm's `subtract(x, broadcast(0))`
// centering no-op disappears).
void populateCanonicalizeBroadcastIdentityPatterns(mlir::RewritePatternSet& patterns);

}  // namespace mps
