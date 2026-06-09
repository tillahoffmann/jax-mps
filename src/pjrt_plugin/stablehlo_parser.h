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

    // True if a reachable func.call survived inlining (e.g. a recursive or
    // otherwise non-inlinable callee). Such calls run via HandleCall, whose
    // compile() lowering is unsafe (jax-mps#170), so the executable falls back to
    // the eager path for them. Usually false: MLIR's inliner inlines the helper
    // calls jaxlib emits (jnp.std/var/numpyro @_where, ...) and SymbolDCE drops
    // the dead callees, so this flag flips true only for the rare residual call.
    bool has_uninlined_call = false;

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
