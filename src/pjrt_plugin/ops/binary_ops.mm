// Binary operations: add, subtract, multiply, divide, maximum, minimum, dot

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_MLIR_BINARY_OP("stablehlo.add", addition, add);
REGISTER_MLIR_BINARY_OP("stablehlo.subtract", subtraction, subtract);
REGISTER_MLIR_BINARY_OP("stablehlo.multiply", multiplication, multiply);
REGISTER_MLIR_BINARY_OP("stablehlo.divide", division, divide);
REGISTER_MLIR_BINARY_OP("stablehlo.maximum", maximum, maximum);
REGISTER_MLIR_BINARY_OP("stablehlo.minimum", minimum, minimum);
REGISTER_MLIR_BINARY_OP("stablehlo.remainder", modulo, remainder);

// Matrix multiplication (dot and dot_general)
static MPSGraphTensor* Handle_dot(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
static bool _reg_dot = ::jax_mps::OpRegistry::Register("stablehlo.dot", Handle_dot);
static bool _reg_dot_general = ::jax_mps::OpRegistry::Register("stablehlo.dot_general", Handle_dot);

}  // namespace jax_mps
