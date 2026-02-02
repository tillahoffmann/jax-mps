// Linear algebra operations: cholesky, triangular_solve

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// stablehlo.cholesky - Cholesky decomposition
// Computes the lower (or upper) triangular Cholesky factor of a positive definite matrix.
// Uses a column-by-column algorithm implemented with MPS Graph operations.
// For an NxN matrix A, computes L such that A = L * L^T (lower=true) or A = U^T * U (lower=false).
static MPSGraphTensor* Handle_cholesky(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        MPS_LOG_ERROR("Expected CholeskyOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        MPS_LOG_ERROR("cholesky: input tensor not found\n");
        return nullptr;
    }

    // Get the 'lower' attribute (default true)
    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    // Get matrix dimensions
    NSArray<NSNumber*>* shape = input.shape;
    NSUInteger rank = shape.count;
    if (rank < 2) {
        MPS_LOG_ERROR("cholesky: input must be at least 2D\n");
        return nullptr;
    }

    NSInteger n = [shape[rank - 1] integerValue];
    NSInteger m = [shape[rank - 2] integerValue];
    if (n != m) {
        MPS_LOG_ERROR("cholesky: input must be square, got %ld x %ld\n", (long)m, (long)n);
        return nullptr;
    }

    if (n > 64) {
        MPS_LOG_WARN("cholesky: large matrix (%ld x %ld) may be slow with column-by-column "
                     "decomposition\n",
                     (long)n, (long)n);
    }

    // Column-by-column Cholesky decomposition built as explicit graph operations.
    // Algorithm (lower triangular):
    // L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2 for k=0..j-1))
    // L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k=0..j-1)) / L[j,j]  for i > j

    MPSDataType dtype = input.dataType;
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dtype];

    // Initialize L as zeros with same shape as input
    MPSGraphTensor* L = [g constantWithScalar:0.0 shape:shape dataType:dtype];

    // Coordinate tensors for element-wise masking
    MPSGraphTensor* iota_row = [g coordinateAlongAxis:0 withShape:shape name:nil];
    MPSGraphTensor* iota_col = [g coordinateAlongAxis:1 withShape:shape name:nil];

    for (NSInteger j = 0; j < n; j++) {
        // Compute diagonal element L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2 for k<j))
        MPSGraphTensor* ajj = [g sliceTensor:input
                                      starts:@[@(j), @(j)]
                                        ends:@[@(j + 1), @(j + 1)]
                                     strides:@[@1, @1]
                                        name:nil];

        MPSGraphTensor* sumSq = zero;
        for (NSInteger k = 0; k < j; k++) {
            MPSGraphTensor* ljk = [g sliceTensor:L
                                          starts:@[@(j), @(k)]
                                            ends:@[@(j + 1), @(k + 1)]
                                         strides:@[@1, @1]
                                            name:nil];
            MPSGraphTensor* ljk_sq = [g multiplicationWithPrimaryTensor:ljk
                                                        secondaryTensor:ljk
                                                                   name:nil];
            sumSq = [g additionWithPrimaryTensor:sumSq secondaryTensor:ljk_sq name:nil];
        }

        MPSGraphTensor* diagVal = [g subtractionWithPrimaryTensor:ajj
                                                  secondaryTensor:sumSq
                                                             name:nil];
        diagVal = [g squareRootWithTensor:diagVal name:nil];

        // Compute off-diagonal elements L[i,j] for i > j
        for (NSInteger i = j + 1; i < n; i++) {
            MPSGraphTensor* aij = [g sliceTensor:input
                                          starts:@[@(i), @(j)]
                                            ends:@[@(i + 1), @(j + 1)]
                                         strides:@[@1, @1]
                                            name:nil];

            MPSGraphTensor* sumProd = zero;
            for (NSInteger k = 0; k < j; k++) {
                MPSGraphTensor* lik = [g sliceTensor:L
                                              starts:@[@(i), @(k)]
                                                ends:@[@(i + 1), @(k + 1)]
                                             strides:@[@1, @1]
                                                name:nil];
                MPSGraphTensor* ljk = [g sliceTensor:L
                                              starts:@[@(j), @(k)]
                                                ends:@[@(j + 1), @(k + 1)]
                                             strides:@[@1, @1]
                                                name:nil];
                MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:lik
                                                          secondaryTensor:ljk
                                                                     name:nil];
                sumProd = [g additionWithPrimaryTensor:sumProd secondaryTensor:prod name:nil];
            }

            MPSGraphTensor* offDiagVal = [g subtractionWithPrimaryTensor:aij
                                                         secondaryTensor:sumProd
                                                                    name:nil];
            offDiagVal = [g divisionWithPrimaryTensor:offDiagVal secondaryTensor:diagVal name:nil];

            // Create mask for position (i, j) and update L
            MPSGraphTensor* iConst = [g constantWithScalar:(double)i dataType:MPSDataTypeInt32];
            MPSGraphTensor* jConst = [g constantWithScalar:(double)j dataType:MPSDataTypeInt32];
            MPSGraphTensor* maskRow = [g equalWithPrimaryTensor:iota_row
                                                secondaryTensor:iConst
                                                           name:nil];
            MPSGraphTensor* maskCol = [g equalWithPrimaryTensor:iota_col
                                                secondaryTensor:jConst
                                                           name:nil];
            MPSGraphTensor* mask = [g logicalANDWithPrimaryTensor:maskRow
                                                  secondaryTensor:maskCol
                                                             name:nil];

            MPSGraphTensor* valBroadcast = [g broadcastTensor:offDiagVal toShape:shape name:nil];
            L = [g selectWithPredicateTensor:mask
                         truePredicateTensor:valBroadcast
                        falsePredicateTensor:L
                                        name:nil];
        }

        // Set diagonal element L[j,j]
        MPSGraphTensor* rowIdx = [g constantWithScalar:(double)j dataType:MPSDataTypeInt32];
        MPSGraphTensor* colIdx = [g constantWithScalar:(double)j dataType:MPSDataTypeInt32];
        MPSGraphTensor* maskDiagRow = [g equalWithPrimaryTensor:iota_row
                                                secondaryTensor:rowIdx
                                                           name:nil];
        MPSGraphTensor* maskDiagCol = [g equalWithPrimaryTensor:iota_col
                                                secondaryTensor:colIdx
                                                           name:nil];
        MPSGraphTensor* maskDiag = [g logicalANDWithPrimaryTensor:maskDiagRow
                                                  secondaryTensor:maskDiagCol
                                                             name:nil];
        MPSGraphTensor* diagBroadcast = [g broadcastTensor:diagVal toShape:shape name:nil];
        L = [g selectWithPredicateTensor:maskDiag
                     truePredicateTensor:diagBroadcast
                    falsePredicateTensor:L
                                    name:nil];
    }

    if (!lower) {
        // Transpose to get upper triangular
        L = [g transposeTensor:L dimension:rank - 2 withDimension:rank - 1 name:nil];
    }

    return L;
}
REGISTER_MPS_OP("stablehlo.cholesky", Handle_cholesky);

// stablehlo.triangular_solve - Solve triangular linear systems
// Solves op(A) * X = B or X * op(A) = B where A is triangular.
// We implement this using A_inv = inverse(A), then X = A_inv * B (or B * A_inv).
static MPSGraphTensor* Handle_triangular_solve(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        MPS_LOG_ERROR("Expected TriangularSolveOp\n");
        return nullptr;
    }

    MPSGraphTensor* A = GetInputTensor(values, op, 0);
    MPSGraphTensor* B = GetInputTensor(values, op, 1);
    if (!A || !B) {
        MPS_LOG_ERROR("triangular_solve: input tensors not found\n");
        return nullptr;
    }

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();

    // Ensure A is properly triangular by zeroing out the other triangle
    NSArray<NSNumber*>* aShape = A.shape;
    NSUInteger rank = aShape.count;

    // Create triangular mask using coordinate tensors
    MPSGraphTensor* rowCoords = [g coordinateAlongAxis:(NSInteger)(rank - 2)
                                             withShape:aShape
                                                  name:nil];
    MPSGraphTensor* colCoords = [g coordinateAlongAxis:(NSInteger)(rank - 1)
                                             withShape:aShape
                                                  name:nil];

    MPSGraphTensor* triMask;
    if (lower) {
        // Lower triangular: row >= col
        triMask = [g greaterThanOrEqualToWithPrimaryTensor:rowCoords
                                           secondaryTensor:colCoords
                                                      name:nil];
    } else {
        // Upper triangular: row <= col
        triMask = [g lessThanOrEqualToWithPrimaryTensor:rowCoords
                                        secondaryTensor:colCoords
                                                   name:nil];
    }

    MPSGraphTensor* zeroTensor = [g constantWithScalar:0.0 dataType:A.dataType];
    MPSGraphTensor* zeroFull = [g broadcastTensor:zeroTensor toShape:aShape name:nil];
    A = [g selectWithPredicateTensor:triMask
                 truePredicateTensor:A
                falsePredicateTensor:zeroFull
                                name:nil];

    // Handle unit diagonal: set diagonal to 1
    if (unitDiagonal) {
        MPSGraphTensor* diagMask = [g equalWithPrimaryTensor:rowCoords
                                             secondaryTensor:colCoords
                                                        name:nil];
        MPSGraphTensor* oneTensor = [g constantWithScalar:1.0 dataType:A.dataType];
        MPSGraphTensor* oneFull = [g broadcastTensor:oneTensor toShape:aShape name:nil];
        A = [g selectWithPredicateTensor:diagMask
                     truePredicateTensor:oneFull
                    falsePredicateTensor:A
                                    name:nil];
    }

    // Apply transpose if needed
    if (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
        transposeA == mlir::stablehlo::Transpose::ADJOINT) {
        A = [g transposeTensor:A dimension:rank - 2 withDimension:rank - 1 name:nil];
    }

    // Compute A_inv using MPS inverseOfTensor
    MPSGraphTensor* A_inv = [g inverseOfTensor:A name:nil];

    // Solve: if left_side, X = A_inv * B; if right_side, X = B * A_inv
    MPSGraphTensor* result;
    if (leftSide) {
        result = [g matrixMultiplicationWithPrimaryTensor:A_inv secondaryTensor:B name:nil];
    } else {
        result = [g matrixMultiplicationWithPrimaryTensor:B secondaryTensor:A_inv name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.triangular_solve", Handle_triangular_solve);

}  // namespace jax_mps
