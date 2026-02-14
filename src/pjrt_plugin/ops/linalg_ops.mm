// Linear algebra operations: cholesky, triangular_solve
// Uses native MPS kernels (MPSMatrixDecompositionCholesky, MPSMatrixSolveTriangular)
// via the NativeOpRegistry.

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// ---------------------------------------------------------------------------
// Helpers for MPS row-byte alignment.
// MPS requires 16-byte-aligned row strides; rowBytesFromColumns returns the
// recommended value (e.g. 16 for a 2-column float32 matrix instead of 8).
// When the data stride differs we blit rows into an aligned staging buffer
// before calling the MPS kernel and blit back afterwards.
// ---------------------------------------------------------------------------

/// Blit rows between buffers with different row strides on the command buffer.
static void BlitRows(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, NSUInteger srcRowBytes,
                     id<MTLBuffer> dst, NSUInteger dstRowBytes, int64_t rows,
                     NSUInteger copyBytes) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    for (int64_t r = 0; r < rows; r++) {
        [blit copyFromBuffer:src
                 sourceOffset:(NSUInteger)r * srcRowBytes
                     toBuffer:dst
            destinationOffset:(NSUInteger)r * dstRowBytes
                         size:copyBytes];
    }
    [blit endEncoding];
}

/// Allocate a zero-filled staging buffer and blit contiguous rows into it.
/// Returns the staging buffer, or the original buffer if no padding is needed.
static id<MTLBuffer> PadBuffer(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src,
                               int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes)
        return src;
    id<MTLBuffer> padded = [device newBufferWithLength:mpsRowBytes * (NSUInteger)rows
                                               options:MTLResourceStorageModeShared];
    memset(padded.contents, 0, mpsRowBytes * (NSUInteger)rows);
    BlitRows(cmdBuf, src, dataRowBytes, padded, mpsRowBytes, rows, dataRowBytes);
    return padded;
}

/// Blit rows from a padded staging buffer into a new contiguous output buffer.
/// Returns the padded buffer directly if no padding was needed.
static id<MTLBuffer> UnpadBuffer(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                 id<MTLBuffer> padded, int64_t rows, NSUInteger dataRowBytes,
                                 NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes)
        return padded;
    size_t outSize = (size_t)rows * dataRowBytes;
    id<MTLBuffer> out = [device newBufferWithLength:outSize options:MTLResourceStorageModeShared];
    BlitRows(cmdBuf, padded, mpsRowBytes, out, dataRowBytes, rows, dataRowBytes);
    return out;
}

// ---------------------------------------------------------------------------
// stablehlo.cholesky – native MPSMatrixDecompositionCholesky
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
// ---------------------------------------------------------------------------

/// Fill a buffer with zeros using blit command encoder.
static void FillBufferWithZeros(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buffer, size_t size) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, size) value:0];
    [blit endEncoding];
}

static NativeResult NativeHandle_cholesky(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                          mlir::Operation* op,
                                          const std::vector<id<MTLBuffer>>& inputs) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        return NativeResult::Error("cholesky: expected CholeskyOp");
    }

    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    if (shape.size() < 2) {
        return NativeResult::Error("cholesky: expected at least rank 2 (got rank " +
                                   std::to_string(shape.size()) + ")");
    }
    int64_t n = shape[shape.size() - 1];
    int64_t m = shape[shape.size() - 2];
    if (n != m) {
        return NativeResult::Error("cholesky: expected square matrix (got " + std::to_string(m) +
                                   " x " + std::to_string(n) + ")");
    }

    // Compute batch size (product of all dimensions except last two).
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        batchSize *= shape[i];
    }

    if (!resultType.getElementType().isF32()) {
        return NativeResult::Error("cholesky: only float32 is supported");
    }

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger dataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger mpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                             dataType:mps_dtype];
    size_t matrixDataSize = (size_t)(n * n) * elem_size;  // Size of one matrix in input.
    size_t matrixMpsSize = (size_t)n * mpsRowBytes;       // Size of one matrix with MPS alignment.

    // Allocate output buffer for all batches.
    size_t totalOutSize = (size_t)batchSize * matrixDataSize;
    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    // Compile the verification shader once.
    static id<MTLComputePipelineState> verifyPipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void cholesky_verify(\n"
                          "    device float *L [[buffer(0)]],\n"
                          "    constant uint &n [[buffer(1)]],\n"
                          "    constant uint &stride [[buffer(2)]],\n"
                          "    uint tid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    if (tid != 0) return;\n"
                          "    for (uint j = 0; j < n; j++) {\n"
                          "        if (L[j * stride + j] <= 0.0f) {\n"
                          "            for (uint r = 0; r < n; r++)\n"
                          "                for (uint c = 0; c < n; c++)\n"
                          "                    L[r * stride + c] = NAN;\n"
                          "            return;\n"
                          "        }\n"
                          "    }\n"
                          "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> func = [lib newFunctionWithName:@"cholesky_verify"];
          verifyPipeline = [device newComputePipelineStateWithFunction:func error:&error];
      }
      if (!verifyPipeline) {
          MPS_LOG_ERROR("cholesky: failed to compile verify shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });

    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                      columns:(NSUInteger)n
                                                                     rowBytes:mpsRowBytes
                                                                     dataType:mps_dtype];

    MPSMatrixDecompositionCholesky* cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:lower
                                                         order:(NSUInteger)n];

    // Process each matrix in the batch.
    for (int64_t b = 0; b < batchSize; b++) {
        // Blit this matrix slice from input buffer using GPU commands.
        size_t srcOffset = (size_t)b * matrixDataSize;
        id<MTLBuffer> srcSlice = [device newBufferWithLength:matrixDataSize
                                                     options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> blitIn = [cmdBuf blitCommandEncoder];
        [blitIn copyFromBuffer:inputs[0]
                  sourceOffset:srcOffset
                      toBuffer:srcSlice
             destinationOffset:0
                          size:matrixDataSize];
        [blitIn endEncoding];

        // Pad source rows to MPS-recommended alignment.
        id<MTLBuffer> srcBuf = PadBuffer(device, cmdBuf, srcSlice, n, dataRowBytes, mpsRowBytes);

        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:desc];

        // Result buffer with MPS alignment (zero-filled so unused triangle is clean).
        id<MTLBuffer> resultBuf = [device newBufferWithLength:matrixMpsSize
                                                      options:MTLResourceStorageModeShared];
        FillBufferWithZeros(cmdBuf, resultBuf, matrixMpsSize);
        MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuf descriptor:desc];

        // The status buffer is unreliable on Apple Silicon — it always writes 0
        // (success) regardless of whether the input is positive definite.
        // See https://developer.apple.com/forums/thread/736787
        [cholesky encodeToCommandBuffer:cmdBuf
                           sourceMatrix:sourceMatrix
                           resultMatrix:resultMatrix
                                 status:nil];

        // Verification kernel to check diagonal and fill with NaN if non-positive.
        if (verifyPipeline) {
            uint32_t n32 = (uint32_t)n;
            uint32_t lStride = (uint32_t)(mpsRowBytes / elem_size);
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:verifyPipeline];
            [enc setBuffer:resultBuf offset:0 atIndex:0];
            [enc setBytes:&n32 length:sizeof(n32) atIndex:1];
            [enc setBytes:&lStride length:sizeof(lStride) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        // Unpad result to contiguous layout and copy to output buffer.
        id<MTLBuffer> unpadded =
            UnpadBuffer(device, cmdBuf, resultBuf, n, dataRowBytes, mpsRowBytes);

        // Blit the unpadded result to the output buffer at the correct offset.
        size_t dstOffset = (size_t)b * matrixDataSize;
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:unpadded
                 sourceOffset:0
                     toBuffer:outBuf
            destinationOffset:dstOffset
                         size:matrixDataSize];
        [blit endEncoding];
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.cholesky", NativeHandle_cholesky);

// ---------------------------------------------------------------------------
// stablehlo.triangular_solve – native MPSMatrixSolveTriangular
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
// ---------------------------------------------------------------------------

static NativeResult NativeHandle_triangular_solve(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                                  mlir::Operation* op,
                                                  const std::vector<id<MTLBuffer>>& inputs) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        return NativeResult::Error("triangular_solve: expected TriangularSolveOp");
    }

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool transpose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
                      transposeA == mlir::stablehlo::Transpose::ADJOINT);

    auto aType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();

    if (aShape.size() < 2 || bShape.size() < 2) {
        return NativeResult::Error("triangular_solve: expected at least rank 2 (got ranks " +
                                   std::to_string(aShape.size()) + ", " +
                                   std::to_string(bShape.size()) + ")");
    }

    // A must be square.
    int64_t aRows = aShape[aShape.size() - 2];
    int64_t aCols = aShape[aShape.size() - 1];
    if (aRows != aCols) {
        return NativeResult::Error("triangular_solve: matrix A must be square (got " +
                                   std::to_string(aRows) + " x " + std::to_string(aCols) + ")");
    }

    // A and B must have matching rank and batch dimensions.
    if (aShape.size() != bShape.size()) {
        return NativeResult::Error("triangular_solve: A and B must have same rank (got " +
                                   std::to_string(aShape.size()) + " vs " +
                                   std::to_string(bShape.size()) + ")");
    }
    for (size_t i = 0; i < aShape.size() - 2; i++) {
        if (aShape[i] != bShape[i]) {
            return NativeResult::Error(
                "triangular_solve: A and B batch dimensions must match (dim " + std::to_string(i) +
                ": " + std::to_string(aShape[i]) + " vs " + std::to_string(bShape[i]) + ")");
        }
    }

    // Compute batch size (product of all dimensions except last two).
    // Both A and B must have the same batch dimensions.
    int64_t batchSize = 1;
    size_t batchRank = aShape.size() - 2;
    for (size_t i = 0; i < batchRank; i++) {
        batchSize *= aShape[i];
    }

    int64_t n = aShape[aShape.size() - 1];
    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];

    if (!bType.getElementType().isF32()) {
        return NativeResult::Error("triangular_solve: only float32 is supported");
    }

    MPSDataType mps_dtype = MlirTypeToMps(bType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(bType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);

    // Number of right-hand sides depends on whether A is on the left or right.
    NSUInteger nrhs = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;

    // Row strides: data-contiguous vs MPS-recommended.
    NSUInteger aDataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger aMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                              dataType:mps_dtype];
    NSUInteger bDataRowBytes = (NSUInteger)(bCols * (int64_t)elem_size);
    NSUInteger bMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)bCols
                                                              dataType:mps_dtype];

    size_t aMatrixDataSize = (size_t)(n * n) * elem_size;
    size_t bMatrixDataSize = (size_t)(bRows * bCols) * elem_size;

    // Allocate output buffer for all batches.
    size_t totalOutSize = (size_t)batchSize * bMatrixDataSize;
    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                       columns:(NSUInteger)n
                                                                      rowBytes:aMpsRowBytes
                                                                      dataType:mps_dtype];

    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                       columns:(NSUInteger)bCols
                                                                      rowBytes:bMpsRowBytes
                                                                      dataType:mps_dtype];

    MPSMatrixSolveTriangular* solver =
        [[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                   right:!leftSide
                                                   upper:!lower
                                               transpose:transpose
                                                    unit:unitDiagonal
                                                   order:(NSUInteger)n
                                  numberOfRightHandSides:nrhs
                                                   alpha:1.0];

    // Process each matrix in the batch.
    for (int64_t b = 0; b < batchSize; b++) {
        // Blit matrix slices from input buffers using GPU commands.
        size_t aOffset = (size_t)b * aMatrixDataSize;
        size_t bOffset = (size_t)b * bMatrixDataSize;

        id<MTLBuffer> aSlice = [device newBufferWithLength:aMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> blitA = [cmdBuf blitCommandEncoder];
        [blitA copyFromBuffer:inputs[0]
                 sourceOffset:aOffset
                     toBuffer:aSlice
            destinationOffset:0
                         size:aMatrixDataSize];
        [blitA endEncoding];

        id<MTLBuffer> bSlice = [device newBufferWithLength:bMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> blitB = [cmdBuf blitCommandEncoder];
        [blitB copyFromBuffer:inputs[1]
                 sourceOffset:bOffset
                     toBuffer:bSlice
            destinationOffset:0
                         size:bMatrixDataSize];
        [blitB endEncoding];

        // Pad inputs to MPS alignment.
        id<MTLBuffer> aBuf = PadBuffer(device, cmdBuf, aSlice, n, aDataRowBytes, aMpsRowBytes);
        id<MTLBuffer> bBuf = PadBuffer(device, cmdBuf, bSlice, bRows, bDataRowBytes, bMpsRowBytes);

        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];

        id<MTLBuffer> solBuf = [device newBufferWithLength:bMpsRowBytes * (NSUInteger)bRows
                                                   options:MTLResourceStorageModeShared];
        MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf descriptor:bDesc];

        [solver encodeToCommandBuffer:cmdBuf
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rhsMatrix
                       solutionMatrix:solutionMatrix];

        // Unpad solution to contiguous layout and copy to output buffer.
        id<MTLBuffer> unpadded =
            UnpadBuffer(device, cmdBuf, solBuf, bRows, bDataRowBytes, bMpsRowBytes);

        // Blit the unpadded result to the output buffer at the correct offset.
        size_t dstOffset = (size_t)b * bMatrixDataSize;
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:unpadded
                 sourceOffset:0
                     toBuffer:outBuf
            destinationOffset:dstOffset
                         size:bMatrixDataSize];
        [blit endEncoding];
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.triangular_solve", NativeHandle_triangular_solve);

}  // namespace jax_mps
