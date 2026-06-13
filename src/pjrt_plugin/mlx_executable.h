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

// True when JAX_MPS_ASYNC_DISPATCH is set to an enabling value
// ("1"/"true"/"yes"/"on", case-insensitive). Shared so the executable and the
// startup notice agree. Evaluated once and cached.
bool IsAsyncDispatchEnabled();

class MlxBuffer;

struct MlxExecuteResult {
    std::vector<std::unique_ptr<MlxBuffer>> buffers;
    // Underlying reason if execution failed (handler exception, MLX eval
    // failure, etc.). Surfaced into the PJRT error message so Python sees
    // the real cause instead of just "Output count mismatch".
    std::string error_message;
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
    // True once the compiled graph is known to contain a control-flow primitive
    // (WhileLoop/Case). Gates the async-dispatch control-flow graph walk so pure
    // executables skip it entirely.
    mutable bool has_control_flow_ = false;
    mutable std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>
        compiled_fn_;
};

}  // namespace jax_mps
