#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mps_buffer.h"

namespace mps {
struct StableHLOModule;
}

namespace jax_mps {

class MpsClient;
class MpsDevice;

// Parsed HLO operation
struct HloOp {
    std::string name;       // e.g., "add", "dot", "tanh"
    std::string output;     // e.g., "%2"
    std::vector<std::string> inputs;  // e.g., ["%0", "%1"]
    int dtype;              // Output dtype
    std::vector<int64_t> shape;  // Output shape
    std::vector<int64_t> broadcast_dims;  // For broadcast_in_dim
    std::vector<int64_t> permutation;     // For transpose

    // For constant operations
    std::vector<float> constant_data;     // Dense constant values
    float constant_scalar = 0.0f;         // Scalar constant value
    bool is_scalar_constant = false;      // True if constant is scalar/splat
};

// Parsed HLO computation
struct HloComputation {
    std::string name;
    std::vector<std::pair<std::string, std::vector<int64_t>>> parameters;  // name -> shape
    std::vector<HloOp> ops;
    std::string root_name;  // Which op is the root/output
};

// Execution result - either success with buffers or an error
struct ExecutionResult {
    std::vector<std::unique_ptr<MpsBuffer>> buffers;
    std::string error;

    bool ok() const { return error.empty(); }
    static ExecutionResult Error(const std::string& msg) {
        ExecutionResult r;
        r.error = msg;
        return r;
    }
};

// Compiled executable for Metal
class MpsExecutable {
public:
    MpsExecutable(MpsClient* client, const mps::StableHLOModule& module);

    ~MpsExecutable();

    // Check if compilation succeeded
    bool IsValid() const { return valid_; }

    // Get compilation error (if !IsValid())
    const std::string& error() const { return error_; }

    // Execution - returns result with error info
    ExecutionResult Execute(
        const std::vector<MpsBuffer*>& inputs,
        MpsDevice* device);

    // Info
    const std::string& name() const { return name_; }
    int num_outputs() const { return num_outputs_; }

private:
    void CompileFromStableHLO(const mps::StableHLOModule& module);

    MpsClient* client_;
    std::string name_;
    std::string error_;
    int num_outputs_ = 1;
    bool valid_ = false;
    HloComputation computation_;
    void* mps_graph_;  // MPSGraph*
    void* mps_executable_;  // MPSGraphExecutable*
};

}  // namespace jax_mps
