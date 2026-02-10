#pragma once

#include <string>
#include <vector>

// registry.h provides MPSGraph types, mlir types, and ValueMap
#include "pjrt_plugin/ops/registry.h"

// Forward declarations for MLIR types used in signatures
namespace mlir {
class Block;
class ModuleOp;
}  // namespace mlir

namespace jax_mps {

// Result type for control flow operations - can be an error or return values
struct ProcessResult {
    std::string error;
    std::vector<mlir::Value> return_values;

    bool ok() const {
        return error.empty();
    }
    static ProcessResult Error(const std::string& msg) {
        ProcessResult r;
        r.error = msg;
        return r;
    }
};

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
