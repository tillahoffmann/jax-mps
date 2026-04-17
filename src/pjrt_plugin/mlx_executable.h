#pragma once

#include <mlx/mlx.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

// Returns the set of op names that have handlers in the MLX executable
std::unordered_set<std::string> GetSupportedOpNames();

class MlxBuffer;

struct MlxExecuteResult {
    std::vector<std::unique_ptr<MlxBuffer>> buffers;
};

// Output metadata for a single output
struct OutputInfo {
    int dtype;
    std::vector<int64_t> shape;
};

class MlxExecutable {
public:
    // Factory method to create executable from parsed module
    static std::unique_ptr<MlxExecutable> Create(mps::ParsedModule parsed_module);

    ~MlxExecutable();

    bool IsValid() const;
    std::string error() const;
    size_t num_outputs() const;

    // Get output metadata (dtype and shape for each output)
    const std::vector<OutputInfo>& output_info() const {
        return output_info_;
    }

    MlxExecuteResult Execute(const std::vector<MlxBuffer*>& inputs);

private:
    MlxExecutable() = default;

    mps::ParsedModule parsed_module_;
    std::string error_;
    bool valid_ = false;
    size_t num_outputs_ = 0;
    std::vector<OutputInfo> output_info_;

    // MLX compile support (thread safety via GetPjrtGlobalMutex at PJRT layer)
    mutable bool compile_attempted_ = false;
    mutable bool compile_succeeded_ = false;
    mutable std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>
        compiled_fn_;
};

}  // namespace jax_mps
