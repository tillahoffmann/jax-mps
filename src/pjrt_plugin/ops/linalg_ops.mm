// Linear algebra operations: cholesky, triangular_solve
// Uses native MPS kernels (MPSMatrixDecompositionCholesky, MPSMatrixSolveTriangular)
// via the NativeOpRegistry.

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// ---------------------------------------------------------------------------
// stablehlo.cholesky – native MPSMatrixDecompositionCholesky
// ---------------------------------------------------------------------------

static id<MTLBuffer> NativeHandle_cholesky(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                           mlir::Operation* op,
                                           const std::vector<id<MTLBuffer>>& inputs) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        MPS_LOG_ERROR("cholesky: expected CholeskyOp\n");
        return nil;
    }

    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    int64_t n = shape[shape.size() - 1];

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger rowBytes = (NSUInteger)(n * (int64_t)elem_size);
    size_t byte_size = (size_t)(n * n) * elem_size;

    // Source matrix
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                      columns:(NSUInteger)n
                                                                     rowBytes:rowBytes
                                                                     dataType:mps_dtype];
    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:inputs[0] descriptor:desc];

    // Output buffer – zero-initialised so the unused triangle is clean.
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:byte_size
                                                     options:MTLResourceStorageModeShared];
    memset(outputBuffer.contents, 0, byte_size);
    MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:desc];

    // Status buffer
    id<MTLBuffer> statusBuffer = [device newBufferWithLength:sizeof(int32_t)
                                                     options:MTLResourceStorageModeShared];

    MPSMatrixDecompositionCholesky* cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:lower
                                                         order:(NSUInteger)n];

    [cholesky encodeToCommandBuffer:cmdBuf
                       sourceMatrix:sourceMatrix
                       resultMatrix:resultMatrix
                             status:statusBuffer];

    return outputBuffer;
}

REGISTER_NATIVE_MPS_OP("stablehlo.cholesky", NativeHandle_cholesky);

// ---------------------------------------------------------------------------
// stablehlo.triangular_solve – native MPSMatrixSolveTriangular
// ---------------------------------------------------------------------------

static id<MTLBuffer> NativeHandle_triangular_solve(id<MTLDevice> device,
                                                   id<MTLCommandBuffer> cmdBuf, mlir::Operation* op,
                                                   const std::vector<id<MTLBuffer>>& inputs) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        MPS_LOG_ERROR("triangular_solve: expected TriangularSolveOp\n");
        return nil;
    }

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool transpose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
                      transposeA == mlir::stablehlo::Transpose::ADJOINT);

    auto aType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto bShape = bType.getShape();

    int64_t n = aType.getShape().back();
    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];

    MPSDataType mps_dtype = MlirTypeToMps(bType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(bType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);

    // Number of right-hand sides depends on whether A is on the left or right.
    NSUInteger nrhs = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;

    // Source matrix A (NxN)
    NSUInteger aRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                       columns:(NSUInteger)n
                                                                      rowBytes:aRowBytes
                                                                      dataType:mps_dtype];
    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:inputs[0] descriptor:aDesc];

    // RHS matrix B and solution X (same shape)
    NSUInteger bRowBytes = (NSUInteger)(bCols * (int64_t)elem_size);
    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                       columns:(NSUInteger)bCols
                                                                      rowBytes:bRowBytes
                                                                      dataType:mps_dtype];
    MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:inputs[1] descriptor:bDesc];

    size_t byte_size = (size_t)(bRows * bCols) * elem_size;
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:byte_size
                                                     options:MTLResourceStorageModeShared];
    MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:bDesc];

    MPSMatrixSolveTriangular* solver =
        [[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                   right:!leftSide
                                                   upper:!lower
                                               transpose:transpose
                                                    unit:unitDiagonal
                                                   order:(NSUInteger)n
                                  numberOfRightHandSides:nrhs
                                                   alpha:1.0];

    [solver encodeToCommandBuffer:cmdBuf
                     sourceMatrix:sourceMatrix
              rightHandSideMatrix:rhsMatrix
                   solutionMatrix:solutionMatrix];

    return outputBuffer;
}

REGISTER_NATIVE_MPS_OP("stablehlo.triangular_solve", NativeHandle_triangular_solve);

}  // namespace jax_mps
