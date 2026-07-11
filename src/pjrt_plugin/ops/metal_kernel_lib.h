// Dispatch a named kernel from a precompiled Metal library (.metallib).
//
// Two levels of use:
//   * Simple: leave `buffers` empty and inputs bind positionally to slots
//     0..N-1, outputs to N..N+M-1 (all row-contiguous).
//   * Advanced: pass an explicit `buffers` layout (input / output / raw-bytes at
//     arbitrary slots) plus `function_constants`, so a caller can drive a kernel
//     with a fixed compiled signature. All array buffers are made row-contiguous.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

// One buffer binding for the kernel's argument table.
struct MklBuffer {
  enum Kind { kInput, kOutput, kBytes };
  int slot = 0;                 // Metal [[buffer(slot)]] index
  Kind kind = kInput;
  int arg = 0;                  // operand index (kInput) / result index (kOutput)
  std::vector<uint8_t> bytes;   // payload for kBytes (e.g. a packed params struct)

  bool operator==(const MklBuffer& o) const {
    return slot == o.slot && kind == o.kind && arg == o.arg && bytes == o.bytes;
  }
};

// One Metal function constant used to specialize the pipeline.
struct MklConstant {
  enum Type { kBool, kInt, kUint, kFloat };
  int index = 0;
  Type type = kBool;
  std::vector<uint8_t> value;   // little-endian packed scalar

  bool operator==(const MklConstant& o) const {
    return index == o.index && type == o.type && value == o.value;
  }
};

// Run kernel `kname` from the metallib at `libpath`, producing outputs of the
// given shapes/dtypes. `hash_name` is the pipeline cache key (defaults to
// `kname` when empty). `threadgroup` is threads per group. When
// `by_threadgroups` is false, `grid` is the total thread count per dim (MLX
// dispatch_threads); when true, `grid` is the number of threadgroups per dim
// (dispatch_threadgroups, for kernels that index by threadgroup_position_in_grid).
// `buffers` empty => positional binding.
std::vector<array> metal_kernel_lib(
    const std::vector<array>& inputs,
    const std::vector<Shape>& out_shapes,
    const std::vector<Dtype>& out_dtypes,
    const std::string& libpath,
    const std::string& kname,
    const std::string& hash_name,
    std::array<int, 3> grid,
    std::array<int, 3> threadgroup,
    bool by_threadgroups,
    std::vector<MklBuffer> buffers,
    std::vector<MklConstant> constants,
    StreamOrDevice s = {});

} // namespace mlx::core
