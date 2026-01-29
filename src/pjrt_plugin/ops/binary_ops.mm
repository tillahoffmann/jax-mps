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
REGISTER_MLIR_BINARY_OP("stablehlo.power", power, power);
REGISTER_MLIR_BINARY_OP("stablehlo.atan2", atan2, atan2);

// Matrix multiplication (dot)
static MPSGraphTensor* Handle_dot(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;
    return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
static bool _reg_dot = ::jax_mps::OpRegistry::Register("stablehlo.dot", Handle_dot);

// Generalized matrix multiplication (dot_general)
// Handles contracting dimensions and batch dimensions
static MPSGraphTensor* Handle_dot_general(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        MPS_LOG_ERROR(" Expected DotGeneralOp\n");
        return nullptr;
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    NSArray<NSNumber*>* lhsShape = lhs.shape;
    NSArray<NSNumber*>* rhsShape = rhs.shape;
    NSUInteger lhsRank = lhsShape.count;
    NSUInteger rhsRank = rhsShape.count;

    // Simple case: standard 2D matmul with contraction on last/first dims
    // LHS: (M, K), RHS: (K, N) -> (M, N)
    if (lhsBatchDims.empty() && rhsBatchDims.empty() && lhsRank == 2 && rhsRank == 2 &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
        int64_t lhsContractDim = lhsContractingDims[0];
        int64_t rhsContractDim = rhsContractingDims[0];

        // Standard matmul: LHS contracts on dim 1, RHS contracts on dim 0
        if (lhsContractDim == 1 && rhsContractDim == 0) {
            return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        }

        // LHS contracts on dim 0: need to transpose LHS
        // (K, M) @ (K, N) -> transpose LHS to (M, K), then (M, K) @ ...
        if (lhsContractDim == 0 && rhsContractDim == 0) {
            // Need: LHS^T @ RHS where inner dims match
            // LHS is (K, M), LHS^T is (M, K)
            // RHS is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* lhsT = [g transposeTensor:lhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhsT secondaryTensor:rhs name:nil];
        }

        // LHS contracts on dim 1, RHS contracts on dim 1: need to transpose RHS
        if (lhsContractDim == 1 && rhsContractDim == 1) {
            // LHS is (M, K), RHS is (N, K), RHS^T is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* rhsT = [g transposeTensor:rhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhsT name:nil];
        }

        // LHS contracts on dim 0, RHS contracts on dim 1: transpose both
        if (lhsContractDim == 0 && rhsContractDim == 1) {
            // LHS is (K, M), RHS is (N, K)
            // LHS^T is (M, K), RHS^T is (K, N)
            // (M, K) @ (K, N) = (M, N)
            MPSGraphTensor* lhsT = [g transposeTensor:lhs permutation:@[@1, @0] name:nil];
            MPSGraphTensor* rhsT = [g transposeTensor:rhs permutation:@[@1, @0] name:nil];
            return [g matrixMultiplicationWithPrimaryTensor:lhsT secondaryTensor:rhsT name:nil];
        }
    }

    // Fall back to simple matmul for unhandled cases
    MPS_LOG_WARN("dot_general with complex contracting/batch dims, falling back to simple matmul. "
                 "LHS contracting: %lld, RHS contracting: %lld\n",
                 lhsContractingDims.empty() ? -1 : lhsContractingDims[0],
                 rhsContractingDims.empty() ? -1 : rhsContractingDims[0]);
    return [g matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
}
static bool _reg_dot_general =
    ::jax_mps::OpRegistry::Register("stablehlo.dot_general", Handle_dot_general);

}  // namespace jax_mps
