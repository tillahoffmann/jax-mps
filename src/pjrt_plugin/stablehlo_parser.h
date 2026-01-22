#ifndef STABLEHLO_PARSER_H
#define STABLEHLO_PARSER_H

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mps {

// Result of parsing a StableHLO bytecode/text module.
// Owns the MLIR context and module, keeping them alive for direct execution.
struct ParsedModule {
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp entry_func;

    // List of unsupported operations encountered during parsing
    std::vector<std::string> unsupported_ops;

    // Check if parsing was successful
    bool ok() const {
        return context && module && entry_func;
    }
};

// Parse MLIR bytecode (StableHLO portable artifact) into a module
// Returns a ParsedModule with ownership of context and module
ParsedModule parseStableHLOBytecode(const char* data, size_t size);

// Parse MLIR text format into a module
ParsedModule parseStableHLOText(const std::string& text);

// Get the text representation from bytecode (for debugging)
std::string bytecodeToText(const char* data, size_t size);

}  // namespace mps

#endif  // STABLEHLO_PARSER_H
