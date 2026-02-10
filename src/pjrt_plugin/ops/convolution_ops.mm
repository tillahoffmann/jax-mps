// Convolution operations for StableHLO

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Compute permutation to transpose from source layout to target layout.
// Both layouts are specified as arrays where layout[i] = position of dimension i in the tensor.
// For example, NHWC input layout: {batch=0, feature=3, spatial=[1,2]} means N at 0, H at 1, W at 2,
// C at 3. Returns nil if no transpose is needed (identity permutation).
static NSArray<NSNumber*>* computePermutation(const std::vector<int64_t>& srcLayout,
                                              const std::vector<int64_t>& dstLayout) {
    size_t rank = srcLayout.size();
    if (rank != dstLayout.size())
        return nil;

    // perm[dstPos] = srcPos where the dimension at dstPos in dst was at srcPos in src
    NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rank];
    bool isIdentity = true;
    for (size_t dstPos = 0; dstPos < rank; ++dstPos) {
        // Find which dimension should be at dstPos in the destination
        int64_t dim = -1;
        for (size_t d = 0; d < rank; ++d) {
            if (dstLayout[d] == static_cast<int64_t>(dstPos)) {
                dim = static_cast<int64_t>(d);
                break;
            }
        }
        // Find where that dimension was in the source
        int64_t srcPos = srcLayout[dim];
        [perm addObject:@(srcPos)];
        if (srcPos != (int64_t)dstPos)
            isIdentity = false;
    }
    return isIdentity ? nil : perm;
}

// Standard NHWC layout: batch=0, feature=3, H=1, W=2
static const std::vector<int64_t> LAYOUT_NHWC = {0, 3, 1, 2};

// Convolution parameters bundled together for clarity
struct ConvParams {
    int64_t strideH = 1, strideW = 1;
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    int64_t dilationH = 1, dilationW = 1;
    int64_t inputDilationH = 1, inputDilationW = 1;
    int64_t featureGroupCount = 1;
    bool isTransposed = false;
};

// Extract convolution parameters from StableHLO op, handling 1D vs 2D
static ConvParams extractConvParams(mlir::stablehlo::ConvolutionOp& convOp, bool is1D) {
    ConvParams p;
    p.featureGroupCount = static_cast<int64_t>(convOp.getFeatureGroupCount());

    auto windowStrides = convOp.getWindowStrides();
    if (windowStrides) {
        auto v = windowStrides.value();
        if (is1D && !v.empty()) {
            p.strideW = v[0];
        } else if (v.size() >= 2) {
            p.strideH = v[0];
            p.strideW = v[1];
        }
    }

    auto padding = convOp.getPadding();
    if (padding) {
        auto paddingAttr = padding.value();
        if (is1D && paddingAttr.getNumElements() >= 2) {
            auto pv = paddingAttr.getValues<int64_t>();
            p.padLeft = pv[{0, 0}];
            p.padRight = pv[{0, 1}];
        } else if (paddingAttr.getNumElements() >= 4) {
            auto pv = paddingAttr.getValues<int64_t>();
            p.padTop = pv[{0, 0}];
            p.padBottom = pv[{0, 1}];
            p.padLeft = pv[{1, 0}];
            p.padRight = pv[{1, 1}];
        }
    }

    auto rhsDilation = convOp.getRhsDilation();
    if (rhsDilation) {
        auto v = rhsDilation.value();
        if (is1D && !v.empty()) {
            p.dilationW = v[0];
        } else if (v.size() >= 2) {
            p.dilationH = v[0];
            p.dilationW = v[1];
        }
    }

    auto lhsDilation = convOp.getLhsDilation();
    if (lhsDilation) {
        auto v = lhsDilation.value();
        if (is1D && !v.empty()) {
            p.inputDilationW = v[0];
            p.isTransposed = (p.inputDilationW != 1);
        } else if (v.size() >= 2) {
            p.inputDilationH = v[0];
            p.inputDilationW = v[1];
            p.isTransposed = (p.inputDilationH != 1 || p.inputDilationW != 1);
        }
    }

    return p;
}

// Compute forward convolution padding from transposed convolution parameters.
// Forward padding = effective_kernel_size - 1 - transposed_padding, clamped to >= 0.
static int64_t forwardPadding(int64_t kernelSize, int64_t dilation, int64_t transposedPad) {
    int64_t effectiveK = 1 + (kernelSize - 1) * dilation;
    return std::max(0LL, effectiveK - 1 - transposedPad);
}

// Prepare kernel for DataGradient API: un-reverse spatial dimensions and swap I/O channels.
// Input kernel is in OIHW format (from StableHLO transposed conv, which pre-reverses the kernel).
// Returns kernel ready for convolution2DDataGradient.
static MPSGraphTensor* prepareKernelForDataGradient(MPSGraph* g, MPSGraphTensor* kernel,
                                                    NSArray<NSNumber*>* spatialAxes) {
    // Un-reverse kernel: StableHLO pre-reverses spatial dims for transposed conv
    MPSGraphTensor* unreversed = [g reverseTensor:kernel axes:spatialAxes name:nil];
    // DataGradient API has I/O semantically swapped; swap dims 0 and 1 (O <-> I in OIHW)
    return [g transposeTensor:unreversed permutation:@[@1, @0, @2, @3] name:nil];
}

// Lift a 1D (rank-3) tensor to 2D (rank-4) by reordering to canonical order [dim0, dim1, dim2]
// then reshaping to [dim0, dim1, 1, dim2] (inserting singleton at position 2).
static MPSGraphTensor* lift1DTo2D(MPSGraph* g, MPSGraphTensor* tensor, int64_t dim0, int64_t dim1,
                                  int64_t dim2) {
    MPSGraphTensor* reordered = tensor;
    if (!(dim0 == 0 && dim1 == 1 && dim2 == 2)) {
        reordered = [g transposeTensor:tensor permutation:@[@(dim0), @(dim1), @(dim2)] name:nil];
    }
    NSArray<NSNumber*>* shape = reordered.shape;
    if (!shape || shape.count != 3)
        return nullptr;
    return [g reshapeTensor:reordered withShape:@[shape[0], shape[1], @1, shape[2]] name:nil];
}

// Convert 2D output back to 1D layout: squeeze the singleton H dimension and transpose to target.
static MPSGraphTensor* convert2DOutputTo1D(MPSGraph* g, MPSGraphTensor* result, int64_t batchDim,
                                           int64_t featureDim, int64_t spatialDim) {
    // Input is NHWC [B, 1, W, C], transpose to NCHW [B, C, 1, W]
    result = [g transposeTensor:result permutation:@[@0, @3, @1, @2] name:nil];
    // Squeeze to [B, C, W]
    NSArray<NSNumber*>* shape4D = result.shape;
    result = [g reshapeTensor:result withShape:@[shape4D[0], shape4D[1], shape4D[3]] name:nil];
    // Transpose from [B, C, W] to target layout using computePermutation
    std::vector<int64_t> srcLayout = {0, 1, 2};  // B=0, C=1, W=2
    std::vector<int64_t> dstLayout = {batchDim, featureDim, spatialDim};
    NSArray<NSNumber*>* perm = computePermutation(srcLayout, dstLayout);
    if (perm) {
        result = [g transposeTensor:result permutation:perm name:nil];
    }
    return result;
}

// Handle stablehlo.convolution
// StableHLO convolution is highly general - supports arbitrary dimension layouts,
// dilations, padding, grouped convolutions, etc.
static ProcessResult Handle_convolution(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
    if (!convOp) {
        return ProcessResult::Error("convolution: expected ConvolutionOp");
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* kernel = GetInputTensor(values, op, 1);
    if (!input || !kernel)
        return ProcessResult::Error("convolution: missing input tensor");

    // Get dimension numbers
    auto dimNumbers = convOp.getDimensionNumbers();
    int64_t inputBatchDim = dimNumbers.getInputBatchDimension();
    int64_t inputFeatureDim = dimNumbers.getInputFeatureDimension();
    auto inputSpatialDims = dimNumbers.getInputSpatialDimensions();

    int64_t kernelInputFeatureDim = dimNumbers.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDim = dimNumbers.getKernelOutputFeatureDimension();
    auto kernelSpatialDims = dimNumbers.getKernelSpatialDimensions();

    int64_t outputBatchDim = dimNumbers.getOutputBatchDimension();
    int64_t outputFeatureDim = dimNumbers.getOutputFeatureDimension();
    auto outputSpatialDims = dimNumbers.getOutputSpatialDimensions();

    // Determine spatial rank and check batch group count
    size_t spatialRank = inputSpatialDims.size();
    if (convOp.getBatchGroupCount() != 1) {
        return ProcessResult::Error("convolution: batch_group_count != 1 not yet supported");
    }

    bool is1D = (spatialRank == 1);
    if (spatialRank > 2) {
        return ProcessResult::Error("convolution: only 1D/2D convolution is currently supported");
    }

    // For 1D: save spatial dim before lifting (batch/feature dims don't change)
    int64_t output1DSpatialDim = is1D ? outputSpatialDims[0] : 0;

    if (is1D) {
        // Lift 1D tensors to 2D for unified processing
        input = lift1DTo2D(g, input, inputBatchDim, inputFeatureDim, inputSpatialDims[0]);
        kernel = lift1DTo2D(g, kernel, kernelOutputFeatureDim, kernelInputFeatureDim,
                            kernelSpatialDims[0]);
        if (!input || !kernel) {
            return ProcessResult::Error(
                "convolution: 1D convolution expects rank-3 input and kernel");
        }
    }

    // === Normalize to 4D dimension numbers ===
    // After 1D lifting: input is NCHW (0,1,2,3), kernel is OIHW (0,1,2,3)
    // For 2D: use original dimension numbers
    int64_t inBatch = is1D ? 0 : inputBatchDim;
    int64_t inFeature = is1D ? 1 : inputFeatureDim;
    int64_t inSpatial0 = is1D ? 2 : inputSpatialDims[0];
    int64_t inSpatial1 = is1D ? 3 : inputSpatialDims[1];

    int64_t kOutput = is1D ? 0 : kernelOutputFeatureDim;
    int64_t kInput = is1D ? 1 : kernelInputFeatureDim;
    int64_t kSpatial0 = is1D ? 2 : kernelSpatialDims[0];
    int64_t kSpatial1 = is1D ? 3 : kernelSpatialDims[1];

    int64_t outSpatial0 = is1D ? 2 : outputSpatialDims[0];
    int64_t outSpatial1 = is1D ? 3 : outputSpatialDims[1];

    // Extract convolution parameters
    ConvParams p = extractConvParams(convOp, is1D);

    // Transpose input to NHWC and kernel to OIHW using normalized dims
    std::vector<int64_t> inputLayout = {inBatch, inFeature, inSpatial0, inSpatial1};
    std::vector<int64_t> kernelLayout = {kOutput, kInput, kSpatial0, kSpatial1};

    MPSGraphTensor* mpsKernel = kernel;
    NSArray<NSNumber*>* kernelPerm = computePermutation(kernelLayout, {0, 1, 2, 3});
    if (kernelPerm) {
        mpsKernel = [g transposeTensor:kernel permutation:kernelPerm name:nil];
    }

    // Transpose input to NHWC (MPS is more reliable with NHWC)
    NSArray<NSNumber*>* inputPerm = computePermutation(inputLayout, LAYOUT_NHWC);
    MPSGraphTensor* convInput = input;
    if (inputPerm) {
        convInput = [g transposeTensor:input permutation:inputPerm name:nil];
    }

    // Perform convolution (or transposed convolution)
    MPSGraphTensor* result;
    if (p.isTransposed) {
        // Transposed convolution using DataGradient API
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            return ProcessResult::Error(
                "convolution: could not get result type for transposed convolution");
        }

        // Get output shape in NHWC format
        // Note: resultShape uses original MLIR dims (3D for 1D conv, 4D for 2D)
        auto resultShape = resultType.getShape();
        int64_t outN = resultShape[outputBatchDim];
        int64_t outH = is1D ? 1 : resultShape[outputSpatialDims[0]];
        int64_t outW = resultShape[is1D ? output1DSpatialDim : outputSpatialDims[1]];
        int64_t outC = resultShape[outputFeatureDim];
        NSArray<NSNumber*>* outputShape = @[@(outN), @(outH), @(outW), @(outC)];

        // Compute forward padding from transposed conv parameters
        int64_t kH = [mpsKernel.shape[2] longLongValue];
        int64_t kW = [mpsKernel.shape[3] longLongValue];

        // Prepare kernel for DataGradient API
        MPSGraphTensor* preparedKernel = prepareKernelForDataGradient(g, mpsKernel, @[@2, @3]);

        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)p.inputDilationW
                          strideInY:(NSUInteger)p.inputDilationH
                    dilationRateInX:(NSUInteger)p.dilationW
                    dilationRateInY:(NSUInteger)p.dilationH
                             groups:(NSUInteger)p.featureGroupCount
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingTop = (NSUInteger)forwardPadding(kH, p.dilationH, p.padTop);
        desc.paddingBottom = (NSUInteger)forwardPadding(kH, p.dilationH, p.padBottom);
        desc.paddingLeft = (NSUInteger)forwardPadding(kW, p.dilationW, p.padLeft);
        desc.paddingRight = (NSUInteger)forwardPadding(kW, p.dilationW, p.padRight);

        result = [g convolution2DDataGradientWithIncomingGradientTensor:convInput
                                                          weightsTensor:preparedKernel
                                                            outputShape:outputShape
                                           forwardConvolutionDescriptor:desc
                                                                   name:nil];
    } else {
        // Normal convolution
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)p.strideW
                          strideInY:(NSUInteger)p.strideH
                    dilationRateInX:(NSUInteger)p.dilationW
                    dilationRateInY:(NSUInteger)p.dilationH
                             groups:(NSUInteger)p.featureGroupCount
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = (NSUInteger)p.padLeft;
        desc.paddingRight = (NSUInteger)p.padRight;
        desc.paddingTop = (NSUInteger)p.padTop;
        desc.paddingBottom = (NSUInteger)p.padBottom;

        result = [g convolution2DWithSourceTensor:convInput
                                    weightsTensor:mpsKernel
                                       descriptor:desc
                                             name:nil];
    }

    // Output is in NHWC format from MPS

    if (is1D) {
        result =
            convert2DOutputTo1D(g, result, outputBatchDim, outputFeatureDim, output1DSpatialDim);
    } else {
        // Transpose 2D output from NHWC to expected layout using normalized dims
        std::vector<int64_t> outputLayout = {outputBatchDim, outputFeatureDim, outSpatial0,
                                             outSpatial1};
        NSArray<NSNumber*>* outputPerm = computePermutation(LAYOUT_NHWC, outputLayout);
        if (outputPerm) {
            result = [g transposeTensor:result permutation:outputPerm name:nil];
        }
    }

    if (!result)
        return ProcessResult::Error("convolution: handler returned null");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.convolution", Handle_convolution);

}  // namespace jax_mps
