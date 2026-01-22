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
    std::string name;                     // e.g., "add", "dot", "tanh"
    std::string output;                   // e.g., "%2"
    std::vector<std::string> inputs;      // e.g., ["%0", "%1"]
    int dtype;                            // Output dtype
    std::vector<int64_t> shape;           // Output shape
    std::vector<int64_t> broadcast_dims;  // For broadcast_in_dim
    std::vector<int64_t> permutation;     // For transpose
    int64_t concatenate_dim = 0;          // For concatenate

    // For slice
    std::vector<int64_t> slice_starts;
    std::vector<int64_t> slice_limits;
    std::vector<int64_t> slice_strides;

    // For dynamic_slice
    std::vector<int64_t> slice_sizes;

    // For iota
    int64_t iota_dim = 0;

    // For custom_call
    std::string custom_call_target;

    // For compare
    std::string compare_direction;  // "LT", "LE", "GT", "GE", "EQ", "NE"

    // For constant operations
    std::vector<float> constant_data;   // Dense constant values (floats only)
    std::vector<uint8_t> constant_raw;  // Raw byte data for non-float constants
    float constant_scalar = 0.0f;       // Scalar constant value
    uint64_t constant_scalar_raw = 0;   // Raw scalar value for integers
    bool is_scalar_constant = false;    // True if constant is scalar/splat
    bool uses_raw_data = false;         // True if constant_raw contains the data
};

// Parsed HLO computation
struct HloComputation {
    std::string name;
    std::vector<std::pair<std::string, std::vector<int64_t>>> parameters;  // name -> shape
    std::vector<HloOp> ops;
    std::string root_name;                   // Which op is the root/output (legacy, last output)
    std::vector<std::string> return_values;  // All return values for multi-output functions
};

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
class MpsExecutable {
public:
    MpsExecutable(MpsClient* client, const mps::StableHLOModule& module);

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
    void CompileFromStableHLO(const mps::StableHLOModule& module);

    MpsClient* client_;
    std::string name_;
    std::string error_;
    int num_outputs_ = 1;
    bool valid_ = false;
    HloComputation computation_;
    void* mps_graph_;       // MPSGraph*
    void* mps_executable_;  // MPSGraphExecutable*
};

}  // namespace jax_mps
