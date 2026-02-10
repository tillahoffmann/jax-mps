#pragma once

#include <string>
#include <vector>

// registry.h provides MPSGraph types, mlir types, ValueMap, and ProcessResult
#include "pjrt_plugin/ops/registry.h"

// Forward declarations for MLIR types used in signatures
namespace mlir {
class Block;
class ModuleOp;
}  // namespace mlir

namespace jax_mps {

// Block processor function type for recursive processing
using BlockProcessor = ProcessResult (*)(MPSGraph* graph, mlir::Block& block, ValueMap& values,
                                         mlir::ModuleOp module, int depth);

// Process stablehlo.while operation
ProcessResult HandleWhileOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values,
                            mlir::ModuleOp module, int depth, BlockProcessor processBlock);

// Process stablehlo.case operation
ProcessResult HandleCaseOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values,
                           mlir::ModuleOp module, int depth, BlockProcessor processBlock);

// Check if an operation is a control flow op handled specially
bool IsControlFlowOp(const std::string& op_name);

}  // namespace jax_mps
