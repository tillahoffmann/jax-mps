// MLIR-based StableHLO parser
// Parses StableHLO bytecode/text and keeps MLIR alive for direct execution

#include "pjrt_plugin/stablehlo_parser.h"

#include <unordered_set>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mps {

namespace {

// Set of supported operation names
const std::unordered_set<std::string>& getSupportedOps() {
    static const std::unordered_set<std::string> supported = {
        // Binary ops
        "stablehlo.add",
        "stablehlo.multiply",
        "stablehlo.subtract",
        "stablehlo.divide",
        "stablehlo.maximum",
        "stablehlo.minimum",
        // Unary ops
        "stablehlo.tanh",
        "stablehlo.exponential",
        "stablehlo.log",
        "stablehlo.negate",
        "stablehlo.abs",
        "stablehlo.sqrt",
        "stablehlo.erf",
        "stablehlo.log_plus_one",
        // Comparison and selection
        "stablehlo.compare",
        "stablehlo.select",
        "stablehlo.clamp",
        // Matrix ops
        "stablehlo.dot",
        "stablehlo.dot_general",
        // Convolution ops
        "stablehlo.convolution",
        // Shape ops
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.broadcast",
        "stablehlo.broadcast_in_dim",
        "stablehlo.convert",
        "stablehlo.constant",
        // Bitwise ops (for RNG)
        "stablehlo.and",
        "stablehlo.or",
        "stablehlo.xor",
        "stablehlo.shift_right_logical",
        "stablehlo.shift_left",
        // Other ops
        "stablehlo.concatenate",
        "stablehlo.slice",
        "stablehlo.dynamic_slice",
        "stablehlo.iota",
        "stablehlo.bitcast_convert",
        "stablehlo.custom_call",
        // CHLO (Client HLO) ops
        "chlo.erf_inv",
        "chlo.next_after",
        // Control flow
        "func.return",
        "func.call",
    };
    return supported;
}

// Register all dialects needed for StableHLO parsing
void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    // Allow unknown dialects (e.g., sdy/Shardy for sharding) to pass through
    context.allowUnregisteredDialects();
}

// Run the inliner pass to inline all func.call operations
bool runInlinerPass(mlir::MLIRContext& context, mlir::ModuleOp module) {
    // Mark all non-main functions as private so they can be inlined
    // The MLIR inliner only inlines private functions by default
    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() != "main" && funcOp.isPublic()) {
            funcOp.setPrivate();
        }
    });

    // Run a single inliner pass - we handle remaining func.call at runtime
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createInlinerPass());

    // Ignore errors from inliner - we'll handle func.call at runtime
    (void)pm.run(module);
    return true;
}

// Find the entry function (usually "main")
mlir::func::FuncOp findEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;

    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() == "main") {
            entry = funcOp;
        } else if (!entry) {
            // Fall back to first function if no "main"
            entry = funcOp;
        }
    });

    return entry;
}

// Check for unsupported operations and collect their names
// Walks ALL functions in the module, not just the entry function
std::vector<std::string> checkUnsupportedOps(mlir::ModuleOp module) {
    std::unordered_set<std::string> unsupported_set;
    const auto& supported = getSupportedOps();

    module.walk([&](mlir::Operation* op) {
        // Skip module and function ops themselves
        if (mlir::isa<mlir::ModuleOp>(op) || mlir::isa<mlir::func::FuncOp>(op)) {
            return;
        }

        std::string name = op->getName().getStringRef().str();
        if (supported.find(name) == supported.end()) {
            unsupported_set.insert(name);
        }
    });

    return std::vector<std::string>(unsupported_set.begin(), unsupported_set.end());
}

// Common parsing logic - takes an already-parsed module
ParsedModule finalizeModule(std::unique_ptr<mlir::MLIRContext> context,
                            mlir::OwningOpRef<mlir::ModuleOp> module) {
    ParsedModule result;

    if (!module) {
        return result;
    }

    // Run the inliner pass to inline all func.call operations
    if (!runInlinerPass(*context, *module)) {
        return result;
    }

    // Find the entry function
    mlir::func::FuncOp entry = findEntryFunction(*module);
    if (!entry) {
        return result;
    }

    // Check for unsupported operations across ALL functions in the module
    result.unsupported_ops = checkUnsupportedOps(*module);

    // Transfer ownership
    result.context = std::move(context);
    result.module = std::move(module);
    result.entry_func = entry;

    return result;
}

}  // namespace

ParsedModule parseStableHLOBytecode(const char* data, size_t size) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    // Create memory buffer from data
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                   /*BufferName=*/"stablehlo_bytecode",
                                                   /*RequiresNullTerminator=*/false);

    // Try to deserialize as portable artifact first (handles VHLO versioning)
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), context.get());

    if (!moduleOp) {
        // Fall back to regular bytecode reading
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    }

    return finalizeModule(std::move(context), std::move(moduleOp));
}

ParsedModule parseStableHLOText(const std::string& text) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    // Parse text format
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::parseSourceString<mlir::ModuleOp>(text, context.get());

    return finalizeModule(std::move(context), std::move(moduleOp));
}

std::string bytecodeToText(const char* data, size_t size) {
    mlir::MLIRContext context;
    registerDialects(context);

    // Create memory buffer
    auto buffer =
        llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);

    // Deserialize
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {
        // Try regular parsing
        llvm::SourceMgr sourceMgr;
        auto bufferCopy = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                           "stablehlo_bytecode", false);
        sourceMgr.AddNewSourceBuffer(std::move(bufferCopy), llvm::SMLoc());
        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!moduleOp) {
        return "";
    }

    // Print to string
    std::string result;
    llvm::raw_string_ostream os(result);
    moduleOp->print(os);
    return result;
}

}  // namespace mps
