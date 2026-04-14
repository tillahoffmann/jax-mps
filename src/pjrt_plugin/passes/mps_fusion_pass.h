#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mps {

// Creates the MPS fusion pass: one nested FuncOp pass that bundles all
// high-level pattern rewrites (matmul+bias -> addmm, future residual+ln,
// etc.) into a single greedy-rewrite fixpoint.
//
// This symbol is defined in the pjrt_plugin_passes static lib, which is
// compiled -fno-rtti to match MLIR. The main (-frtti) plugin library just
// holds the returned unique_ptr<Pass> and hands it to its PassManager —
// same pattern as createStablehloAggressiveSimplificationPass.
std::unique_ptr<mlir::Pass> createMpsFusionPass();

}  // namespace mps
