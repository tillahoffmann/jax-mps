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
    if (shape.size() != 2) {
        MPS_LOG_ERROR("cholesky: batched inputs not yet supported (got rank %zu)\n", shape.size());
        return nil;
    }
    int64_t n = shape[shape.size() - 1];

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger dataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger mpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                             dataType:mps_dtype];

    // Pad source rows to MPS-recommended alignment.
    id<MTLBuffer> srcBuf = PadBuffer(device, cmdBuf, inputs[0], n, dataRowBytes, mpsRowBytes);

    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                      columns:(NSUInteger)n
                                                                     rowBytes:mpsRowBytes
                                                                     dataType:mps_dtype];
    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:desc];

    // Result buffer with MPS alignment (zero-filled so unused triangle is clean).
    id<MTLBuffer> resultBuf = [device newBufferWithLength:mpsRowBytes * (NSUInteger)n
                                                  options:MTLResourceStorageModeShared];
    memset(resultBuf.contents, 0, mpsRowBytes * (NSUInteger)n);
    MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuf descriptor:desc];

    MPSMatrixDecompositionCholesky* cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:lower
                                                         order:(NSUInteger)n];

    // The status buffer is unreliable on Apple Silicon — it always writes 0
    // (success) regardless of whether the input is positive definite.
    // See https://developer.apple.com/forums/thread/736787
    [cholesky encodeToCommandBuffer:cmdBuf
                       sourceMatrix:sourceMatrix
                       resultMatrix:resultMatrix
                             status:nil];

    // MPSMatrixDecompositionCholesky copies the source to the result before
    // decomposing in-place.  On failure it stops, leaving partially computed
    // pivots on the diagonal.  The failing pivot is always non-positive (the
    // kernel computes d = A[j,j] - sum_k L[j,k]^2 and stops when d <= 0).
    // Encode a verification kernel that checks the diagonal and fills the
    // output with NaN if any pivot is non-positive, matching CPU behavior.
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

    // Unpad result to contiguous layout.
    return UnpadBuffer(device, cmdBuf, resultBuf, n, dataRowBytes, mpsRowBytes);
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
    if (aType.getShape().size() != 2 || bType.getShape().size() != 2) {
        MPS_LOG_ERROR("triangular_solve: batched inputs not yet supported (got ranks %zu, %zu)\n",
                      aType.getShape().size(), bType.getShape().size());
        return nil;
    }
    auto bShape = bType.getShape();

    int64_t n = aType.getShape().back();
    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];

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

    // Pad inputs to MPS alignment.
    id<MTLBuffer> aBuf = PadBuffer(device, cmdBuf, inputs[0], n, aDataRowBytes, aMpsRowBytes);
    id<MTLBuffer> bBuf = PadBuffer(device, cmdBuf, inputs[1], bRows, bDataRowBytes, bMpsRowBytes);

    // Source matrix A (NxN)
    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                       columns:(NSUInteger)n
                                                                      rowBytes:aMpsRowBytes
                                                                      dataType:mps_dtype];
    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];

    // RHS matrix B and solution X (same shape)
    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                       columns:(NSUInteger)bCols
                                                                      rowBytes:bMpsRowBytes
                                                                      dataType:mps_dtype];
    MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];

    id<MTLBuffer> solBuf = [device newBufferWithLength:bMpsRowBytes * (NSUInteger)bRows
                                               options:MTLResourceStorageModeShared];
    MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf descriptor:bDesc];

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

    // Unpad solution to contiguous layout.
    return UnpadBuffer(device, cmdBuf, solBuf, bRows, bDataRowBytes, bMpsRowBytes);
}

REGISTER_NATIVE_MPS_OP("stablehlo.triangular_solve", NativeHandle_triangular_solve);

}  // namespace jax_mps
