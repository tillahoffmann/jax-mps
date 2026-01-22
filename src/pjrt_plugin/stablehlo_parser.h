#ifndef STABLEHLO_PARSER_H
#define STABLEHLO_PARSER_H

#include <string>
#include <variant>
#include <vector>

namespace mps {

// Represents a tensor shape
struct TensorType {
    std::vector<int64_t> shape;
    std::string element_type;  // "f32", "f64", "i32", etc.
};

// Represents an operand (argument reference)
struct Operand {
    std::string name;  // e.g., "%arg0", "%0"
};

// Supported StableHLO operations
enum class OpKind {
    Add,
    Multiply,
    Subtract,
    Divide,
    Maximum,
    Minimum,
    Tanh,
    Exp,
    Log,
    Negate,
    Abs,
    Sqrt,
    LogPlusOne,
    Compare,
    Select,
    Dot,
    DotGeneral,
    Reshape,
    Transpose,
    Broadcast,
    BroadcastInDim,
    Reduce,
    Convert,
    Constant,
    Return,
    Call,
    // Bitwise operations (needed for RNG)
    And,
    Or,
    Xor,
    ShiftRightLogical,
    ShiftLeft,
    // Other operations
    Concatenate,
    Slice,
    DynamicSlice,
    Iota,
    BitcastConvert,
    CustomCall,
    Unknown
};

// Represents a parsed StableHLO operation
struct StableHLOOp {
    OpKind kind;
    std::string name;  // Result name (e.g., "%0")
    std::string
        op_name;  // Original operation name (e.g., "stablehlo.gather") - set for Unknown ops
    std::vector<Operand> operands;
    TensorType result_type;

    // For constants
    std::vector<float> constant_data;   // Dense array constant data (floats)
    std::vector<uint8_t> constant_raw;  // Raw byte data for non-float constants
    float constant_scalar = 0.0f;       // Scalar constant value (when is_scalar_constant is true)
    uint64_t constant_scalar_raw = 0;   // Raw scalar value for integers
    bool is_scalar_constant = false;    // True if constant is a scalar or splat
    bool uses_raw_data = false;         // True if constant_raw contains the data

    // For dot_general
    std::vector<int64_t> lhs_batching_dims;
    std::vector<int64_t> rhs_batching_dims;
    std::vector<int64_t> lhs_contracting_dims;
    std::vector<int64_t> rhs_contracting_dims;

    // For transpose/broadcast
    std::vector<int64_t> permutation;
    std::vector<int64_t> broadcast_dimensions;

    // For concatenate
    int64_t concatenate_dimension = 0;

    // For slice
    std::vector<int64_t> slice_starts;
    std::vector<int64_t> slice_limits;
    std::vector<int64_t> slice_strides;

    // For dynamic_slice
    std::vector<int64_t> slice_sizes;

    // For iota
    int64_t iota_dimension = 0;

    // For custom_call
    std::string custom_call_target;

    // For compare
    std::string compare_direction;  // "LT", "LE", "GT", "GE", "EQ", "NE"

    // For func.call
    std::string call_target;  // Name of the function being called
};

// Represents a parsed function
struct StableHLOFunction {
    std::string name;
    std::vector<TensorType> arg_types;
    std::vector<TensorType> result_types;
    std::vector<StableHLOOp> ops;
};

// Represents a parsed module
struct StableHLOModule {
    std::vector<StableHLOFunction> functions;
    std::string entry_function;  // Usually "main"

    // List of unsupported operations encountered during parsing
    std::vector<std::string> unsupported_ops;
};

// Parse MLIR bytecode (StableHLO portable artifact) into a module
// Returns true on success, false on failure
bool parseStableHLOBytecode(const char* data, size_t size, StableHLOModule& module);

// Parse MLIR text format into a module
bool parseStableHLOText(const std::string& text, StableHLOModule& module);

// Get the text representation from bytecode (for debugging)
std::string bytecodeToText(const char* data, size_t size);

}  // namespace mps

#endif  // STABLEHLO_PARSER_H
