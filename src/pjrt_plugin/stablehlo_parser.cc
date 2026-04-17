// MLIR-based StableHLO parser
// Parses StableHLO bytecode/text and keeps MLIR alive for direct execution

#include "pjrt_plugin/stablehlo_parser.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <system_error>
#include <unordered_set>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
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
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/ops/registry.h"
#include "pjrt_plugin/passes/mps_fusion_pass.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mps {

namespace {

// Set of supported operation names - derived from OpRegistry plus runtime-lowered ops
const std::unordered_set<std::string>& getSupportedOps() {
    static std::unordered_set<std::string> supported = []() {
        auto ops = jax_mps::OpRegistry::GetRegisteredOps();
        // Add ops handled directly in mlx_executable.cc (not via OpRegistry).
        ops.insert("func.return");
        ops.insert("func.call");
        return ops;
    }();
    return supported;
}

// Register all dialects needed for StableHLO parsing
void registerDialects(mlir::MLIRContext& context) {
    // Our StableHLO modules are tiny; MLIR's default per-context LLVM thread
    // pool (hardware_concurrency workers) would otherwise leak ~8 worker
    // threads per compile, since JAX caches the MlxExecutable — and thus its
    // MLIRContext — for the lifetime of the process.
    context.disableMultithreading();
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

// Run StableHLO algebraic simplification passes (x*1 -> x, x+0 -> x, etc.)
// Disable by setting JAX_MPS_NO_OPTIMIZE=1 if you encounter issues.
bool runOptimizationPasses(mlir::MLIRContext& context, mlir::ModuleOp module) {
    // Read every call so tests (and interactive bisection) can flip this
    // mid-process without restarting the plugin.
    const char* env = std::getenv("JAX_MPS_NO_OPTIMIZE");
    if (env && std::string(env) == "1")
        return true;

    mlir::PassManager pm(&context);
    // Algebraic simplification: x*1 -> x, x+0 -> x, etc.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createStablehloAggressiveSimplificationPass());
    // MPS-specific fusions (matmul+bias -> addmm, etc.). Run after upstream
    // simplification so patterns like x+0 are already cleaned up before we
    // try to match.
    pm.addNestedPass<mlir::func::FuncOp>(createMpsFusionPass());

    if (mlir::failed(pm.run(module))) {
        fprintf(stderr,
                "ERROR: StableHLO optimization pass failed. The module may be in a partially "
                "transformed state. Set JAX_MPS_NO_OPTIMIZE=1 to skip optimization passes.\n");
        return false;
    }

    return true;
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

    // If inliner fails, log but continue - we handle func.call at runtime
    if (mlir::failed(pm.run(module))) {
        MPS_LOG_DEBUG("Inliner pass failed, func.call ops will be handled at runtime\n");
    }
    return true;
}

// Find the entry function (usually "main")
mlir::func::FuncOp findEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;

    module.walk([&](mlir::func::FuncOp funcOp) {
        // Prefer "main", otherwise fall back to first function
        if (funcOp.getName() == "main" || !entry) {
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

    // Run StableHLO algebraic simplification passes after inlining so the passes
    // see fully-inlined IR without func.call ops. Fatal on failure because a
    // failed pass may leave the module partially transformed.
    if (!runOptimizationPasses(*context, *module)) {
        return result;
    }

    // Optional: dump the post-pass module to a file. Tests use this to inspect
    // the IR that our passes produced (see tests/test_fusion.py). Path is the
    // env var's value; one file per parsed module, named after a counter.
    if (const char* dumpPath = std::getenv("JAX_MPS_DUMP_OPTIMIZED_IR")) {
        static std::atomic<int> counter{0};
        int id = counter.fetch_add(1);
        std::error_code ec;
        if (auto dirEc = llvm::sys::fs::create_directories(dumpPath)) {
            MPS_LOG_WARN("JAX_MPS_DUMP_OPTIMIZED_IR: could not create %s: %s\n", dumpPath,
                         dirEc.message().c_str());
        } else {
            // PID-stamped so concurrent processes pointing at the same
            // directory don't overwrite each other's dumps.
            std::string filename = std::string(dumpPath) + "/module_" +
                                   std::to_string(llvm::sys::Process::getProcessId()) + "_" +
                                   std::to_string(id) + ".mlir";
            llvm::raw_fd_ostream os(filename, ec);
            if (ec) {
                MPS_LOG_WARN("JAX_MPS_DUMP_OPTIMIZED_IR: could not open %s: %s\n", filename.c_str(),
                             ec.message().c_str());
            } else {
                module->print(os);
                os.flush();
            }
        }
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
