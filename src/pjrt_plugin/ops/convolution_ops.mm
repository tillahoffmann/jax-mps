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

    // Build inverse of srcLayout: invSrc[pos] = which dimension is at position pos
    std::vector<int64_t> invSrc(rank);
    for (size_t dim = 0; dim < rank; ++dim) {
        invSrc[srcLayout[dim]] = dim;
    }

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

// Build layout vector for 4D tensor from dimension positions.
// Returns {batch, feature, spatial0, spatial1} positions.
static std::vector<int64_t> makeLayout4D(int64_t batchDim, int64_t featureDim, int64_t spatial0,
                                         int64_t spatial1) {
    return {batchDim, featureDim, spatial0, spatial1};
}

// Standard layouts for reference
static const std::vector<int64_t> LAYOUT_NHWC = {0, 3, 1, 2};  // batch=0, feature=3, H=1, W=2
static const std::vector<int64_t> LAYOUT_OIHW = {0, 1, 2, 3};  // O=0, I=1, H=2, W=3

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

    // Get convolution attributes
    auto windowStrides = convOp.getWindowStrides();
    auto padding = convOp.getPadding();
    auto lhsDilation = convOp.getLhsDilation();  // input dilation
    auto rhsDilation = convOp.getRhsDilation();  // kernel dilation
    int64_t featureGroupCount = convOp.getFeatureGroupCount();
    int64_t batchGroupCount = convOp.getBatchGroupCount();

    // Determine spatial rank
    size_t spatialRank = inputSpatialDims.size();

    // Check batch group count
    if (batchGroupCount != 1) {
        MPS_LOG_ERROR("batch_group_count != 1 not yet supported\n");
        return nullptr;
    }

    // === 1D convolution: lift to 2D ===
    // Save original 1D output layout info for final conversion
    bool is1D = (spatialRank == 1);
    int64_t output1DBatchDim = outputBatchDim;
    int64_t output1DFeatureDim = outputFeatureDim;
    int64_t output1DSpatialDim = is1D ? outputSpatialDims[0] : 0;

    if (is1D) {
        // Lift input from 3D to 4D: reorder to [B, C, W] then reshape to [B, C, 1, W]
        NSMutableArray<NSNumber*>* inputPerm = [NSMutableArray arrayWithCapacity:3];
        [inputPerm addObject:@(inputBatchDim)];
        [inputPerm addObject:@(inputFeatureDim)];
        [inputPerm addObject:@(inputSpatialDims[0])];

        MPSGraphTensor* bcwInput = input;
        if (!(inputBatchDim == 0 && inputFeatureDim == 1 && inputSpatialDims[0] == 2)) {
            bcwInput = [g transposeTensor:input permutation:inputPerm name:nil];
        }
        NSArray<NSNumber*>* bcwShape = bcwInput.shape;
        if (!bcwShape || bcwShape.count != 3) {
            MPS_LOG_ERROR("1D convolution expects rank-3 input\n");
            return nullptr;
        }
        input = [g reshapeTensor:bcwInput
                       withShape:@[bcwShape[0], bcwShape[1], @1, bcwShape[2]]
                            name:nil];
        // Input is now NCHW with H=1
        inputBatchDim = 0;
        inputFeatureDim = 1;

        // Lift kernel from 3D to 4D: reorder to [O, I, K] then reshape to [O, I, 1, K]
        NSMutableArray<NSNumber*>* kernelPerm = [NSMutableArray arrayWithCapacity:3];
        [kernelPerm addObject:@(kernelOutputFeatureDim)];
        [kernelPerm addObject:@(kernelInputFeatureDim)];
        [kernelPerm addObject:@(kernelSpatialDims[0])];

        MPSGraphTensor* oikKernel = kernel;
        if (!(kernelOutputFeatureDim == 0 && kernelInputFeatureDim == 1 &&
              kernelSpatialDims[0] == 2)) {
            oikKernel = [g transposeTensor:kernel permutation:kernelPerm name:nil];
        }
        NSArray<NSNumber*>* oikShape = oikKernel.shape;
        if (!oikShape || oikShape.count != 3) {
            MPS_LOG_ERROR("1D convolution expects rank-3 kernel\n");
            return nullptr;
        }
        kernel = [g reshapeTensor:oikKernel
                        withShape:@[oikShape[0], oikShape[1], @1, oikShape[2]]
                             name:nil];
        // Kernel is now OIHW with H=1
        kernelOutputFeatureDim = 0;
        kernelInputFeatureDim = 1;

        // Set output layout to NCHW (will squeeze H later)
        outputBatchDim = 0;
        outputFeatureDim = 1;
    }

    // === From here on, we have 2D tensors (or 1D lifted to 2D) ===

    if (spatialRank != 1 && spatialRank != 2) {
        MPS_LOG_ERROR("Only 1D/2D convolution is currently supported, got %zu spatial dims\n",
                      spatialRank);
        return nullptr;
    }

    // Extract strides (default to 1)
    int64_t strideH = 1, strideW = 1;
    if (windowStrides) {
        auto stridesVec = windowStrides.value();
        if (is1D && stridesVec.size() >= 1) {
            strideW = stridesVec[0];
        } else if (stridesVec.size() >= 2) {
            strideH = stridesVec[0];
            strideW = stridesVec[1];
        }
    }

    // Extract padding (default to 0)
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (padding) {
        auto paddingAttr = padding.value();
        if (is1D && paddingAttr.getNumElements() >= 2) {
            auto paddingValues = paddingAttr.getValues<int64_t>();
            padLeft = paddingValues[{0, 0}];
            padRight = paddingValues[{0, 1}];
        } else if (paddingAttr.getNumElements() >= 4) {
            auto paddingValues = paddingAttr.getValues<int64_t>();
            padTop = paddingValues[{0, 0}];
            padBottom = paddingValues[{0, 1}];
            padLeft = paddingValues[{1, 0}];
            padRight = paddingValues[{1, 1}];
        }
    }

    // Extract dilations (default to 1)
    int64_t dilationH = 1, dilationW = 1;
    if (rhsDilation) {
        auto dilationVec = rhsDilation.value();
        if (is1D && dilationVec.size() >= 1) {
            dilationW = dilationVec[0];
        } else if (dilationVec.size() >= 2) {
            dilationH = dilationVec[0];
            dilationW = dilationVec[1];
        }
    }

    // Extract input dilation (for transposed convolution)
    int64_t inputDilationH = 1, inputDilationW = 1;
    bool isTransposedConv = false;
    if (lhsDilation) {
        auto lhsDilationVec = lhsDilation.value();
        if (is1D && lhsDilationVec.size() >= 1) {
            inputDilationW = lhsDilationVec[0];
            isTransposedConv = (inputDilationW != 1);
        } else if (lhsDilationVec.size() >= 2) {
            inputDilationH = lhsDilationVec[0];
            inputDilationW = lhsDilationVec[1];
            isTransposedConv = (inputDilationH != 1 || inputDilationW != 1);
        }
    }

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
        descriptorWithStrideInX:(NSUInteger)strideW
                      strideInY:(NSUInteger)strideH
                dilationRateInX:(NSUInteger)dilationW
                dilationRateInY:(NSUInteger)dilationH
                         groups:(NSUInteger)featureGroupCount
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    desc.paddingLeft = (NSUInteger)padLeft;
    desc.paddingRight = (NSUInteger)padRight;
    desc.paddingTop = (NSUInteger)padTop;
    desc.paddingBottom = (NSUInteger)padBottom;

    // Perform convolution (or transposed convolution)
    MPSGraphTensor* result;
    if (isTransposedConv) {
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
        int64_t fwdPadTop = forwardPadding(kH, dilationH, padTop);
        int64_t fwdPadBottom = forwardPadding(kH, dilationH, padBottom);
        int64_t fwdPadLeft = forwardPadding(kW, dilationW, padLeft);
        int64_t fwdPadRight = forwardPadding(kW, dilationW, padRight);

        // Prepare kernel for DataGradient API
        MPSGraphTensor* preparedKernel = prepareKernelForDataGradient(g, mpsKernel, @[@2, @3]);

        // Create forward conv descriptor
        MPSGraphConvolution2DOpDescriptor* fwdDesc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)inputDilationW
                          strideInY:(NSUInteger)inputDilationH
                    dilationRateInX:(NSUInteger)dilationW
                    dilationRateInY:(NSUInteger)dilationH
                             groups:(NSUInteger)featureGroupCount
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
        // Convert from NHWC [B, 1, W, C] to target 1D layout
        // First transpose to NCHW [B, C, 1, W]
        result = [g transposeTensor:result permutation:@[@0, @3, @1, @2] name:nil];
        // Squeeze to [B, C, W]
        NSArray<NSNumber*>* shape4D = result.shape;
        result = [g reshapeTensor:result withShape:@[shape4D[0], shape4D[1], shape4D[3]] name:nil];
        // Transpose from [B, C, W] to target layout
        if (!(output1DBatchDim == 0 && output1DFeatureDim == 1 && output1DSpatialDim == 2)) {
            NSMutableArray<NSNumber*>* outPerm = [NSMutableArray arrayWithCapacity:3];
            for (int i = 0; i < 3; ++i)
                [outPerm addObject:@0];
            outPerm[(NSUInteger)output1DBatchDim] = @0;
            outPerm[(NSUInteger)output1DFeatureDim] = @1;
            outPerm[(NSUInteger)output1DSpatialDim] = @2;
            result = [g transposeTensor:result permutation:outPerm name:nil];
        }
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
