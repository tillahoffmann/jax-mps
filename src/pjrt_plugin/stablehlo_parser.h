#ifndef STABLEHLO_PARSER_H
#define STABLEHLO_PARSER_H

#include <string>
#include <vector>
#include <variant>

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
    Tanh,
    Exp,
    Log,
    Negate,
    Abs,
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
    Unknown
};

// Represents a parsed StableHLO operation
struct StableHLOOp {
    OpKind kind;
    std::string name;  // Result name (e.g., "%0")
    std::vector<Operand> operands;
    TensorType result_type;

    // For constants
    std::vector<float> constant_data;

    // For dot_general
    std::vector<int64_t> lhs_batching_dims;
    std::vector<int64_t> rhs_batching_dims;
    std::vector<int64_t> lhs_contracting_dims;
    std::vector<int64_t> rhs_contracting_dims;

    // For transpose/broadcast
    std::vector<int64_t> permutation;
    std::vector<int64_t> broadcast_dimensions;
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
