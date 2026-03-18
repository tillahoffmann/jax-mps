// MLX op registry
// Single source of truth: ops are registered in GetOpHandlers() in mlx_executable.cc
// This file provides access to the op names for parsing/validation

#pragma once

#include <string>
#include <unordered_set>

namespace jax_mps {

// Forward declaration - implemented in mlx_executable.cc
// Returns the set of op names that have handlers
std::unordered_set<std::string> GetSupportedOpNames();

class OpRegistry {
public:
    // Returns the set of supported ops (delegates to GetSupportedOpNames)
    static std::unordered_set<std::string> GetRegisteredOps() {
        return GetSupportedOpNames();
    }
};

}  // namespace jax_mps
