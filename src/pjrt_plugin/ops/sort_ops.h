#pragma once

#include "pjrt_plugin/ops/control_flow_ops.h"
#include "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Handle multi-result stablehlo.reduce for argmax/argmin patterns
// This is called directly from mps_executable.mm because stablehlo.reduce
// can have single or multiple results, requiring special dispatch logic.
ProcessResult HandleMultiResultReduceOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values);

// HandleSortOp and HandleTopKOp are registered via MultiResultOpRegistry
// and don't need explicit declarations here.

}  // namespace jax_mps
