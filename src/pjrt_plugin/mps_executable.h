#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "pjrt_plugin/mps_buffer.h"

namespace mps {
struct ParsedModule;
}

namespace jax_mps {

class MpsClient;
class MpsDevice;

// Execution result - either success with buffers or an error
struct ExecutionResult {
    std::vector<std::unique_ptr<MpsBuffer>> buffers;
    std::string error;

    bool ok() const {
        return error.empty();
    }
    static ExecutionResult Error(const std::string& msg) {
        ExecutionResult r;
        r.error = msg;
        return r;
    }
};

// Compiled executable for Metal
// Owns the MLIR context and module, executing operations directly from MLIR
class MpsExecutable {
public:
    // Takes ownership of the ParsedModule
    MpsExecutable(MpsClient* client, mps::ParsedModule module);

    ~MpsExecutable();

    // Check if compilation succeeded
    bool IsValid() const {
        return valid_;
    }

    // Get compilation error (if !IsValid())
    const std::string& error() const {
        return error_;
    }

    // Execution - returns result with error info
    ExecutionResult Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device);

    // Info
    const std::string& name() const {
        return name_;
    }
    int num_outputs() const {
        return num_outputs_;
    }

private:
    MpsClient* client_;
    std::string name_;
    std::string error_;
    int num_outputs_ = 1;
    bool valid_ = false;

    // MLIR ownership - keeps the module alive for execution
    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    mlir::func::FuncOp entry_func_;
};

}  // namespace jax_mps
