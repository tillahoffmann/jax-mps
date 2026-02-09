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
            if (dstLayout[d] == (int64_t)dstPos) {
                dim = d;
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

// Standard layouts for reference
static const std::vector<int64_t> LAYOUT_NHWC = {0, 3, 1, 2};  // batch=0, feature=3, H=1, W=2
static const std::vector<int64_t> LAYOUT_OIHW = {0, 1, 2, 3};  // O=0, I=1, H=2, W=3

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
    p.featureGroupCount = convOp.getFeatureGroupCount();

    auto windowStrides = convOp.getWindowStrides();
    if (windowStrides) {
        auto v = windowStrides.value();
        if (is1D && v.size() >= 1) {
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
        if (is1D && v.size() >= 1) {
            p.dilationW = v[0];
        } else if (v.size() >= 2) {
            p.dilationH = v[0];
            p.dilationW = v[1];
        }
    }

    auto lhsDilation = convOp.getLhsDilation();
    if (lhsDilation) {
        auto v = lhsDilation.value();
        if (is1D && v.size() >= 1) {
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

// Lift 1D input tensor to 2D by reordering to [B, C, W] then reshaping to [B, C, 1, W].
// Returns nullptr on error.
static MPSGraphTensor* lift1DInputTo2D(MPSGraph* g, MPSGraphTensor* input, int64_t batchDim,
                                       int64_t featureDim, int64_t spatialDim) {
    MPSGraphTensor* bcwInput = input;
    if (!(batchDim == 0 && featureDim == 1 && spatialDim == 2)) {
        NSArray<NSNumber*>* perm = @[@(batchDim), @(featureDim), @(spatialDim)];
        bcwInput = [g transposeTensor:input permutation:perm name:nil];
    }
    NSArray<NSNumber*>* shape = bcwInput.shape;
    if (!shape || shape.count != 3) {
        MPS_LOG_ERROR("1D convolution expects rank-3 input\n");
        return nullptr;
    }
    return [g reshapeTensor:bcwInput withShape:@[shape[0], shape[1], @1, shape[2]] name:nil];
}

// Lift 1D kernel tensor to 2D by reordering to [O, I, K] then reshaping to [O, I, 1, K].
// Returns nullptr on error.
static MPSGraphTensor* lift1DKernelTo2D(MPSGraph* g, MPSGraphTensor* kernel, int64_t outputDim,
                                        int64_t inputDim, int64_t spatialDim) {
    MPSGraphTensor* oikKernel = kernel;
    if (!(outputDim == 0 && inputDim == 1 && spatialDim == 2)) {
        NSArray<NSNumber*>* perm = @[@(outputDim), @(inputDim), @(spatialDim)];
        oikKernel = [g transposeTensor:kernel permutation:perm name:nil];
    }
    NSArray<NSNumber*>* shape = oikKernel.shape;
    if (!shape || shape.count != 3) {
        MPS_LOG_ERROR("1D convolution expects rank-3 kernel\n");
        return nullptr;
    }
    return [g reshapeTensor:oikKernel withShape:@[shape[0], shape[1], @1, shape[2]] name:nil];
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
static MPSGraphTensor* Handle_convolution(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
    if (!convOp) {
        MPS_LOG_ERROR("Expected ConvolutionOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* kernel = GetInputTensor(values, op, 1);
    if (!input || !kernel)
        return nullptr;

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
        MPS_LOG_ERROR("batch_group_count != 1 not yet supported\n");
        return nullptr;
    }

    // === 1D convolution: lift to 2D ===
    // Save original 1D output layout info for final conversion
    bool is1D = (spatialRank == 1);
    int64_t output1DBatchDim = outputBatchDim;
    int64_t output1DFeatureDim = outputFeatureDim;
    int64_t output1DSpatialDim = is1D ? outputSpatialDims[0] : 0;

    if (spatialRank > 2) {
        MPS_LOG_ERROR("Only 1D/2D convolution is currently supported, got %zu spatial dims\n",
                      spatialRank);
        return nullptr;
    }

    if (is1D) {
        // Lift 1D tensors to 2D for unified processing
        input = lift1DInputTo2D(g, input, inputBatchDim, inputFeatureDim, inputSpatialDims[0]);
        if (!input)
            return nullptr;
        inputBatchDim = 0;
        inputFeatureDim = 1;

        kernel = lift1DKernelTo2D(g, kernel, kernelOutputFeatureDim, kernelInputFeatureDim,
                                  kernelSpatialDims[0]);
        if (!kernel)
            return nullptr;
        kernelOutputFeatureDim = 0;
        kernelInputFeatureDim = 1;

        outputBatchDim = 0;
        outputFeatureDim = 1;
    }

    // === From here on, we have 2D tensors (or 1D lifted to 2D) ===

    // Extract convolution parameters
    ConvParams p = extractConvParams(convOp, is1D);

    // Build input layout for permutation computation
    // For 1D lifted to 2D, we already have NCHW (batch=0, feature=1, H=2, W=3)
    std::vector<int64_t> inputLayout;
    if (is1D) {
        inputLayout = {0, 1, 2, 3};  // NCHW
    } else {
        inputLayout = {inputBatchDim, inputFeatureDim, inputSpatialDims[0], inputSpatialDims[1]};
    }

    // Transpose kernel to OIHW format for MPS
    // Kernel layout: {O, I, H, W} positions
    MPSGraphTensor* mpsKernel = kernel;
    if (!is1D) {
        std::vector<int64_t> kernelLayout = {kernelOutputFeatureDim, kernelInputFeatureDim,
                                             kernelSpatialDims[0], kernelSpatialDims[1]};
        NSArray<NSNumber*>* kernelPerm = computePermutation(kernelLayout, LAYOUT_OIHW);
        if (kernelPerm) {
            mpsKernel = [g transposeTensor:kernel permutation:kernelPerm name:nil];
        }
    }
    // For 1D, kernel is already OIHW from the lifting step

    // Transpose input to NHWC (MPS is more reliable with NHWC)
    NSArray<NSNumber*>* inputPerm = computePermutation(inputLayout, LAYOUT_NHWC);
    MPSGraphTensor* convInput = input;
    if (inputPerm) {
        convInput = [g transposeTensor:input permutation:inputPerm name:nil];
    }

    // Create convolution descriptor (always use NHWC since we transpose above)
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

    // Perform convolution (or transposed convolution)
    MPSGraphTensor* result;
    if (p.isTransposed) {
        // Transposed convolution using DataGradient API
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            MPS_LOG_ERROR("Could not get result type for transposed convolution\n");
            return nullptr;
        }

        // Get output shape in NHWC format
        auto resultShape = resultType.getShape();
        NSArray<NSNumber*>* outputShape;
        if (is1D) {
            // For 1D, output shape uses original 1D dimension numbers
            int64_t outN = resultShape[output1DBatchDim];
            int64_t outW = resultShape[output1DSpatialDim];
            int64_t outC = resultShape[output1DFeatureDim];
            outputShape = @[@(outN), @1, @(outW), @(outC)];
        } else {
            int64_t outN = resultShape[outputBatchDim];
            int64_t outH = resultShape[outputSpatialDims[0]];
            int64_t outW = resultShape[outputSpatialDims[1]];
            int64_t outC = resultShape[outputFeatureDim];
            outputShape = @[@(outN), @(outH), @(outW), @(outC)];
        }

        // Compute forward padding from transposed conv parameters
        int64_t kH = [mpsKernel.shape[2] longLongValue];
        int64_t kW = [mpsKernel.shape[3] longLongValue];
        int64_t fwdPadTop = forwardPadding(kH, p.dilationH, p.padTop);
        int64_t fwdPadBottom = forwardPadding(kH, p.dilationH, p.padBottom);
        int64_t fwdPadLeft = forwardPadding(kW, p.dilationW, p.padLeft);
        int64_t fwdPadRight = forwardPadding(kW, p.dilationW, p.padRight);

        // Prepare kernel for DataGradient API
        MPSGraphTensor* preparedKernel = prepareKernelForDataGradient(g, mpsKernel, @[@2, @3]);

        // Create forward conv descriptor
        MPSGraphConvolution2DOpDescriptor* fwdDesc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)p.inputDilationW
                          strideInY:(NSUInteger)p.inputDilationH
                    dilationRateInX:(NSUInteger)p.dilationW
                    dilationRateInY:(NSUInteger)p.dilationH
                             groups:(NSUInteger)p.featureGroupCount
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        fwdDesc.paddingTop = (NSUInteger)fwdPadTop;
        fwdDesc.paddingBottom = (NSUInteger)fwdPadBottom;
        fwdDesc.paddingLeft = (NSUInteger)fwdPadLeft;
        fwdDesc.paddingRight = (NSUInteger)fwdPadRight;

        result = [g convolution2DDataGradientWithIncomingGradientTensor:convInput
                                                          weightsTensor:preparedKernel
                                                            outputShape:outputShape
                                           forwardConvolutionDescriptor:fwdDesc
                                                                   name:nil];
    } else {
        // Normal convolution
        result = [g convolution2DWithSourceTensor:convInput
                                    weightsTensor:mpsKernel
                                       descriptor:desc
                                             name:nil];
    }

    // Output is in NHWC format from MPS

    if (is1D) {
        result = convert2DOutputTo1D(g, result, output1DBatchDim, output1DFeatureDim,
                                     output1DSpatialDim);
    } else {
        // Transpose 2D output from NHWC to expected layout
        std::vector<int64_t> outputLayout = {outputBatchDim, outputFeatureDim, outputSpatialDims[0],
                                             outputSpatialDims[1]};
        NSArray<NSNumber*>* outputPerm = computePermutation(LAYOUT_NHWC, outputLayout);
        if (outputPerm) {
            result = [g transposeTensor:result permutation:outputPerm name:nil];
        }
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.convolution", Handle_convolution);

}  // namespace jax_mps
