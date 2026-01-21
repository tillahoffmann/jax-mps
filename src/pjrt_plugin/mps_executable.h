#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mps {
struct StableHLOModule;
}

namespace jax_mps {

class MpsClient;
class MpsDevice;
class MpsBuffer;

// Parsed HLO operation
struct HloOp {
    std::string name;       // e.g., "add", "dot", "tanh"
    std::string output;     // e.g., "%2"
    std::vector<std::string> inputs;  // e.g., ["%0", "%1"]
    int dtype;              // Output dtype
    std::vector<int64_t> shape;  // Output shape
    std::vector<int64_t> broadcast_dims;  // For broadcast_in_dim
    std::vector<int64_t> permutation;     // For transpose
};

// Parsed HLO computation
struct HloComputation {
    std::string name;
    std::vector<std::pair<std::string, std::vector<int64_t>>> parameters;  // name -> shape
    std::vector<HloOp> ops;
    std::string root_name;  // Which op is the root/output
};

// Compiled executable for Metal
class MpsExecutable {
public:
    MpsExecutable(MpsClient* client, const mps::StableHLOModule& module);

    ~MpsExecutable();

    // Check if compilation succeeded
    bool IsValid() const { return valid_; }

    // Execution
    std::vector<std::unique_ptr<MpsBuffer>> Execute(
        const std::vector<MpsBuffer*>& inputs,
        MpsDevice* device);

    // Info
    const std::string& name() const { return name_; }
    int num_outputs() const { return num_outputs_; }

private:
    void CompileFromStableHLO(const mps::StableHLOModule& module);

    MpsClient* client_;
    std::string name_;
    int num_outputs_ = 1;
    bool valid_ = false;
    HloComputation computation_;
    void* mps_graph_;  // MPSGraph*
    void* mps_executable_;  // MPSGraphExecutable*
};

}  // namespace jax_mps
