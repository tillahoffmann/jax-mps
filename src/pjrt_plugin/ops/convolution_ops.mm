// Convolution operations for StableHLO

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

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

    // 1D convolution support: lift to 2D with a singleton height dimension.
    if (spatialRank == 1) {
        if (batchGroupCount != 1) {
            MPS_LOG_ERROR("1D convolution: batch_group_count != 1 not yet supported\n");
            return nullptr;
        }

        int64_t strideW = 1;
        if (windowStrides && windowStrides->size() >= 1) {
            strideW = (*windowStrides)[0];
        }

        int64_t padLeft = 0, padRight = 0;
        if (padding) {
            auto paddingAttr = *padding;
            if (paddingAttr.getNumElements() >= 2) {
                auto paddingValues = paddingAttr.getValues<int64_t>();
                padLeft = paddingValues[{0, 0}];
                padRight = paddingValues[{0, 1}];
            }
        }

        int64_t dilationW = 1;
        if (rhsDilation && rhsDilation->size() >= 1) {
            dilationW = (*rhsDilation)[0];
        }

        int64_t inputDilationW = 1;
        if (lhsDilation && lhsDilation->size() >= 1) {
            inputDilationW = (*lhsDilation)[0];
        }
        bool is1DTransposedConv = (inputDilationW != 1);

        // Reorder input to [B, F, S], then reshape to NCHW [B, F, 1, S].
        NSMutableArray<NSNumber*>* inputPerm = [NSMutableArray arrayWithCapacity:3];
        [inputPerm addObject:@(inputBatchDim)];
        [inputPerm addObject:@(inputFeatureDim)];
        [inputPerm addObject:@(inputSpatialDims[0])];
        MPSGraphTensor* bfsInput = input;
        if (!(inputBatchDim == 0 && inputFeatureDim == 1 && inputSpatialDims[0] == 2)) {
            bfsInput = [g transposeTensor:input permutation:inputPerm name:nil];
        }
        NSArray<NSNumber*>* bfsShape = bfsInput.shape;
        if (!bfsShape || bfsShape.count != 3) {
            MPS_LOG_ERROR("1D convolution expects rank-3 input\n");
            return nullptr;
        }
        MPSGraphTensor* convInput = [g reshapeTensor:bfsInput
                                           withShape:@[bfsShape[0], bfsShape[1], @1, bfsShape[2]]
                                                name:nil];

        // Reorder kernel to [O, I, K], then reshape to OIHW [O, I, 1, K].
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
        MPSGraphTensor* mpsKernel = [g reshapeTensor:oikKernel
                                           withShape:@[oikShape[0], oikShape[1], @1, oikShape[2]]
                                                name:nil];

        MPSGraphTensor* convOut;
        if (is1DTransposedConv) {
            // 1D Transposed convolution using DataGradient API
            // Use NHWC layout for reliability with DataGradient API
            auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
            if (!resultType) {
                MPS_LOG_ERROR("Could not get result type for 1D transposed convolution\n");
                return nullptr;
            }

            // Convert input from NCHW [B, C, 1, W] to NHWC [B, 1, W, C]
            MPSGraphTensor* nhwcInput = [g transposeTensor:convInput
                                               permutation:@[@0, @2, @3, @1]
                                                      name:nil];

            // Get output shape in NHWC format [N, 1, W, C]
            auto resultShape = resultType.getShape();
            int64_t outN = resultShape[outputBatchDim];
            int64_t outW = resultShape[outputSpatialDims[0]];
            int64_t outC = resultShape[outputFeatureDim];
            NSArray<NSNumber*>* outputShape = @[@(outN), @1, @(outW), @(outC)];

            // Get kernel width from mpsKernel (OIHW format with H=1)
            NSArray<NSNumber*>* kernelShape = mpsKernel.shape;
            int64_t kW = [kernelShape[3] longLongValue];

            // Compute effective kernel size accounting for dilation
            int64_t effectiveKW = 1 + (kW - 1) * dilationW;

            // Compute forward padding: forward_pad = effective_kernel - 1 - transposed_pad
            int64_t fwdPadLeft = effectiveKW - 1 - padLeft;
            int64_t fwdPadRight = effectiveKW - 1 - padRight;

            // Clamp forward padding to non-negative values
            if (fwdPadLeft < 0)
                fwdPadLeft = 0;
            if (fwdPadRight < 0)
                fwdPadRight = 0;

            // Un-reverse kernel: flip W dimension (dim 3)
            MPSGraphTensor* unreversedKernel = [g reverseTensor:mpsKernel axes:@[@3] name:nil];

            // For gradient computation, the kernel has I/O dimensions semantically swapped.
            // DataGradient API needs weights in [C_out, C_in, H, W] format, so swap dims 0 and 1.
            unreversedKernel = [g transposeTensor:unreversedKernel
                                      permutation:@[@1, @0, @2, @3]
                                             name:nil];

            // Create forward conv descriptor with strideInX = inputDilationW
            MPSGraphConvolution2DOpDescriptor* fwdDesc = [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)inputDilationW
                              strideInY:1
                        dilationRateInX:(NSUInteger)dilationW
                        dilationRateInY:1
                                 groups:(NSUInteger)featureGroupCount
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

            fwdDesc.paddingTop = 0;
            fwdDesc.paddingBottom = 0;
            fwdDesc.paddingLeft = (NSUInteger)fwdPadLeft;
            fwdDesc.paddingRight = (NSUInteger)fwdPadRight;

            // Call DataGradient API (result is in NHWC [B, 1, W, C])
            MPSGraphTensor* nhwcOut =
                [g convolution2DDataGradientWithIncomingGradientTensor:nhwcInput
                                                         weightsTensor:unreversedKernel
                                                           outputShape:outputShape
                                          forwardConvolutionDescriptor:fwdDesc
                                                                  name:nil];

            // Convert back to NCHW [B, C, 1, W] from NHWC [B, 1, W, C]
            convOut = [g transposeTensor:nhwcOut permutation:@[@0, @3, @1, @2] name:nil];
        } else {
            // Normal 1D convolution
            MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)strideW
                              strideInY:1
                        dilationRateInX:(NSUInteger)dilationW
                        dilationRateInY:1
                                 groups:(NSUInteger)featureGroupCount
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            desc.paddingTop = 0;
            desc.paddingBottom = 0;
            desc.paddingLeft = (NSUInteger)padLeft;
            desc.paddingRight = (NSUInteger)padRight;

            convOut = [g convolution2DWithSourceTensor:convInput
                                         weightsTensor:mpsKernel
                                            descriptor:desc
                                                  name:nil];
        }
        if (!convOut) {
            MPS_LOG_ERROR("1D convolution lowering failed\n");
            return nullptr;
        }

        // Back to [B, F, S].
        NSArray<NSNumber*>* out4Shape = convOut.shape;
        if (!out4Shape || out4Shape.count != 4) {
            MPS_LOG_ERROR("1D convolution output rank mismatch\n");
            return nullptr;
        }
        MPSGraphTensor* bfsOut = [g reshapeTensor:convOut
                                        withShape:@[out4Shape[0], out4Shape[1], out4Shape[3]]
                                             name:nil];

        // Reorder [B, F, S] into the StableHLO output layout.
        NSMutableArray<NSNumber*>* outPerm = [NSMutableArray arrayWithCapacity:3];
        for (int i = 0; i < 3; ++i)
            [outPerm addObject:@0];
        outPerm[(NSUInteger)outputBatchDim] = @0;
        outPerm[(NSUInteger)outputFeatureDim] = @1;
        outPerm[(NSUInteger)outputSpatialDims[0]] = @2;

        MPSGraphTensor* out = bfsOut;
        if (!(outputBatchDim == 0 && outputFeatureDim == 1 && outputSpatialDims[0] == 2)) {
            out = [g transposeTensor:bfsOut permutation:outPerm name:nil];
        }
        return out;
    }

    // Currently only support 1D and 2D convolutions.
    if (spatialRank != 2) {
        MPS_LOG_ERROR("Only 1D/2D convolution is currently supported, got %zu spatial dims\n",
                      spatialRank);
        return nullptr;
    }

    // Check batch group count (used for gradient computations)
    if (batchGroupCount != 1) {
        MPS_LOG_ERROR("batch_group_count != 1 not yet supported\n");
        return nullptr;
    }

    // Extract strides (default to 1)
    int64_t strideH = 1, strideW = 1;
    if (windowStrides) {
        auto stridesVec = windowStrides.value();
        if (stridesVec.size() >= 2) {
            strideH = stridesVec[0];
            strideW = stridesVec[1];
        }
    }

    // Extract padding (default to 0)
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (padding) {
        auto paddingAttr = padding.value();
        // padding is a 2D array: [[pad_top, pad_bottom], [pad_left, pad_right]]
        if (paddingAttr.getNumElements() >= 4) {
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
        if (dilationVec.size() >= 2) {
            dilationH = dilationVec[0];
            dilationW = dilationVec[1];
        }
    }

    // Extract input dilation (for transposed convolution)
    int64_t inputDilationH = 1, inputDilationW = 1;
    bool isTransposedConv = false;
    if (lhsDilation) {
        auto lhsDilationVec = lhsDilation.value();
        if (lhsDilationVec.size() >= 2) {
            inputDilationH = lhsDilationVec[0];
            inputDilationW = lhsDilationVec[1];
            isTransposedConv = (inputDilationH != 1 || inputDilationW != 1);
        }
    }

    // Determine if we need to transpose input/kernel to match MPS expected layout
    // MPS expects NHWC for input and OHWI for kernel (output, height, width, input)
    // But it can also work with NCHW via dataLayout setting

    // Check if input is NHWC (batch=0, feature=3, spatial=[1,2])
    bool inputIsNHWC =
        (inputBatchDim == 0 && inputFeatureDim == 3 && inputSpatialDims.size() == 2 &&
         inputSpatialDims[0] == 1 && inputSpatialDims[1] == 2);

    // Check if input is NCHW (batch=0, feature=1, spatial=[2,3])
    bool inputIsNCHW =
        (inputBatchDim == 0 && inputFeatureDim == 1 && inputSpatialDims.size() == 2 &&
         inputSpatialDims[0] == 2 && inputSpatialDims[1] == 3);

    // Check if input is CHWN (batch=3, feature=0, spatial=[1,2]) - used in gradient computation
    bool inputIsCHWN =
        (inputBatchDim == 3 && inputFeatureDim == 0 && inputSpatialDims.size() == 2 &&
         inputSpatialDims[0] == 1 && inputSpatialDims[1] == 2);

    // Check if input is CNHW (batch=1, feature=0, spatial=[2,3]) - used in kernel gradient of NCHW
    // conv
    bool inputIsCNHW =
        (inputBatchDim == 1 && inputFeatureDim == 0 && inputSpatialDims.size() == 2 &&
         inputSpatialDims[0] == 2 && inputSpatialDims[1] == 3);

    // MPS kernel layout is OHWI (output_features, height, width, input_features)
    // StableHLO default is often HWIO (height, width, input_features, output_features)
    // Check kernel layout: HWIO means spatial=[0,1], input=2, output=3
    bool kernelIsHWIO =
        (kernelSpatialDims.size() == 2 && kernelSpatialDims[0] == 0 && kernelSpatialDims[1] == 1 &&
         kernelInputFeatureDim == 2 && kernelOutputFeatureDim == 3);

    // Also check for OIHW (common in some frameworks)
    bool kernelIsOIHW =
        (kernelOutputFeatureDim == 0 && kernelInputFeatureDim == 1 &&
         kernelSpatialDims.size() == 2 && kernelSpatialDims[0] == 2 && kernelSpatialDims[1] == 3);

    // Check for IHWO (used in gradient computation) - input=0, spatial=[1,2], output=3
    bool kernelIsIHWO =
        (kernelInputFeatureDim == 0 && kernelSpatialDims.size() == 2 && kernelSpatialDims[0] == 1 &&
         kernelSpatialDims[1] == 2 && kernelOutputFeatureDim == 3);

    // Check for HWOI (used in input gradient computation) - spatial=[0,1], output=2, input=3
    bool kernelIsHWOI =
        (kernelSpatialDims.size() == 2 && kernelSpatialDims[0] == 0 && kernelSpatialDims[1] == 1 &&
         kernelOutputFeatureDim == 2 && kernelInputFeatureDim == 3);

    // Check if it's already OHWI
    bool kernelIsOHWI =
        (kernelOutputFeatureDim == 0 && kernelSpatialDims.size() == 2 &&
         kernelSpatialDims[0] == 1 && kernelSpatialDims[1] == 2 && kernelInputFeatureDim == 3);

    // Check for IOHW (used in gradient computation of NCHW conv) - input=0, output=1, spatial=[2,3]
    bool kernelIsIOHW =
        (kernelInputFeatureDim == 0 && kernelOutputFeatureDim == 1 &&
         kernelSpatialDims.size() == 2 && kernelSpatialDims[0] == 2 && kernelSpatialDims[1] == 3);

    if (!inputIsNHWC && !inputIsNCHW && !inputIsCHWN && !inputIsCNHW) {
        MPS_LOG_ERROR(
            "Unsupported input layout. Expected NHWC, NCHW, CHWN, or CNHW. Got batch=%lld, "
            "feature=%lld, spatial=[%lld,%lld]\n",
            inputBatchDim, inputFeatureDim, inputSpatialDims[0], inputSpatialDims[1]);
        return nullptr;
    }

    // Transpose kernel directly to OIHW format for MPS
    // MPS weightsLayout OIHW expects [outputChannels, inputChannels/groups, kH, kW]
    // Fused transpose: combine intermediate OHWI step into single permutation
    MPSGraphTensor* mpsKernel = kernel;
    if (kernelIsHWIO) {
        // HWIO [H, W, I, O] -> OIHW [O, I, H, W] directly
        // Fused permutation: [3, 0, 1, 2] then [0, 3, 1, 2] = [3, 2, 0, 1]
        mpsKernel = [g transposeTensor:kernel permutation:@[@3, @2, @0, @1] name:nil];
    } else if (kernelIsOIHW) {
        // OIHW [O, I, H, W] -> OIHW: no transpose needed (identity)
        // Fused permutation: [0, 2, 3, 1] then [0, 3, 1, 2] = [0, 1, 2, 3]
        mpsKernel = kernel;
    } else if (kernelIsIHWO) {
        // IHWO [I, H, W, O] -> OIHW [O, I, H, W] directly
        // Fused permutation: [3, 1, 2, 0] then [0, 3, 1, 2] = [3, 0, 1, 2]
        mpsKernel = [g transposeTensor:kernel permutation:@[@3, @0, @1, @2] name:nil];
    } else if (kernelIsHWOI) {
        // HWOI [H, W, O, I] -> OIHW [O, I, H, W] directly
        // Fused permutation: [2, 0, 1, 3] then [0, 3, 1, 2] = [2, 3, 0, 1]
        mpsKernel = [g transposeTensor:kernel permutation:@[@2, @3, @0, @1] name:nil];
    } else if (kernelIsOHWI) {
        // OHWI [O, H, W, I] -> OIHW [O, I, H, W]
        // Permutation: [0, 3, 1, 2]
        mpsKernel = [g transposeTensor:kernel permutation:@[@0, @3, @1, @2] name:nil];
    } else if (kernelIsIOHW) {
        // IOHW [I, O, H, W] -> OIHW [O, I, H, W]
        // Swap dims 0 and 1
        mpsKernel = [g transposeTensor:kernel permutation:@[@1, @0, @2, @3] name:nil];
    } else {
        MPS_LOG_ERROR("Unsupported kernel layout. Got output=%lld, input=%lld, "
                      "spatial=[%lld,%lld]\n",
                      kernelOutputFeatureDim, kernelInputFeatureDim, kernelSpatialDims[0],
                      kernelSpatialDims[1]);
        return nullptr;
    }

    // Create convolution descriptor
    MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:(NSUInteger)strideW
                      strideInY:(NSUInteger)strideH
                dilationRateInX:(NSUInteger)dilationW
                dilationRateInY:(NSUInteger)dilationH
                         groups:(NSUInteger)featureGroupCount
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:inputIsNHWC ? MPSGraphTensorNamedDataLayoutNHWC
                                            : MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    // Set explicit padding
    desc.paddingLeft = (NSUInteger)padLeft;
    desc.paddingRight = (NSUInteger)padRight;
    desc.paddingTop = (NSUInteger)padTop;
    desc.paddingBottom = (NSUInteger)padBottom;

    // Handle input layout - transpose to NHWC for MPS
    MPSGraphTensor* convInput = input;
    if (inputIsNCHW) {
        // For NCHW, transpose to NHWC first since MPS is more reliable with NHWC
        // NCHW -> NHWC: [0, 2, 3, 1]
        convInput = [g transposeTensor:input permutation:@[@0, @2, @3, @1] name:nil];
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    } else if (inputIsCHWN) {
        // For CHWN (used in gradient computation), transpose to NHWC
        // CHWN [C, H, W, N] -> NHWC [N, H, W, C]: [3, 1, 2, 0]
        convInput = [g transposeTensor:input permutation:@[@3, @1, @2, @0] name:nil];
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    } else if (inputIsCNHW) {
        // For CNHW (used in kernel gradient of NCHW conv), transpose to NHWC
        // CNHW [C, N, H, W] -> NHWC [N, H, W, C]: [1, 2, 3, 0]
        convInput = [g transposeTensor:input permutation:@[@1, @2, @3, @0] name:nil];
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    }

    // Perform convolution (or transposed convolution)
    MPSGraphTensor* result;
    if (isTransposedConv) {
        // Transposed convolution (used in backward pass of strided conv)
        // Use convolution2DDataGradient API which correctly handles all padding configurations.
        // MPS's convolutionTranspose2D has bugs when pad_start < lhs_dilation.

        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            MPS_LOG_ERROR("Could not get result type for transposed convolution\n");
            return nullptr;
        }

        // Get output shape in NHWC format
        auto resultShape = resultType.getShape();
        NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];

        int64_t outN = resultShape[outputBatchDim];
        int64_t outH = resultShape[outputSpatialDims[0]];
        int64_t outW = resultShape[outputSpatialDims[1]];
        int64_t outC = resultShape[outputFeatureDim];

        [outputShape addObject:@(outN)];
        [outputShape addObject:@(outH)];
        [outputShape addObject:@(outW)];
        [outputShape addObject:@(outC)];

        // Get kernel spatial dimensions from mpsKernel (OIHW format)
        NSArray<NSNumber*>* kernelShape = mpsKernel.shape;
        int64_t kH = [kernelShape[2] longLongValue];
        int64_t kW = [kernelShape[3] longLongValue];

        // Compute effective kernel size accounting for dilation
        int64_t effectiveKH = 1 + (kH - 1) * dilationH;
        int64_t effectiveKW = 1 + (kW - 1) * dilationW;

        // Compute forward padding: forward_pad = effective_kernel - 1 - transposed_pad
        // This translates the transposed conv padding to the equivalent forward conv padding
        int64_t fwdPadTop = effectiveKH - 1 - padTop;
        int64_t fwdPadBottom = effectiveKH - 1 - padBottom;
        int64_t fwdPadLeft = effectiveKW - 1 - padLeft;
        int64_t fwdPadRight = effectiveKW - 1 - padRight;

        // Clamp forward padding to non-negative values.
        // The DataGradient API uses the outputShape parameter to determine the actual
        // output size, so we just need valid (non-negative) padding values.
        if (fwdPadTop < 0)
            fwdPadTop = 0;
        if (fwdPadBottom < 0)
            fwdPadBottom = 0;
        if (fwdPadLeft < 0)
            fwdPadLeft = 0;
        if (fwdPadRight < 0)
            fwdPadRight = 0;

        // Un-reverse kernel: StableHLO pre-reverses for transposed conv
        // Kernel is in OIHW format, flip H (dim 2) and W (dim 3)
        MPSGraphTensor* unreversedKernel = [g reverseTensor:mpsKernel axes:@[@2, @3] name:nil];

        // For gradient computation, the kernel has I/O dimensions semantically swapped.
        // The StableHLO uses layout like HWOI where O=C_in and I=C_out (opposite of forward).
        // DataGradient API needs weights in [C_out, C_in, H, W] format, so swap dims 0 and 1.
        unreversedKernel = [g transposeTensor:unreversedKernel
                                  permutation:@[@1, @0, @2, @3]
                                         name:nil];

        // Create FORWARD conv descriptor
        // For DataGradient, stride = inputDilation (upsampling factor)
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

        // Call DataGradient API - computes gradient of data with respect to loss
        result = [g convolution2DDataGradientWithIncomingGradientTensor:convInput
                                                          weightsTensor:unreversedKernel
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

    // Transpose output back to expected layout if needed
    // Output is currently in NHWC format from MPS
    // Note: For kernel gradient computations, "batch" dim holds input features and
    // "feature" dim holds output features, effectively making the output a kernel tensor.
    bool outputIsNHWC =
        (outputBatchDim == 0 && outputFeatureDim == 3 && outputSpatialDims.size() == 2 &&
         outputSpatialDims[0] == 1 && outputSpatialDims[1] == 2);
    bool outputIsNCHW =
        (outputBatchDim == 0 && outputFeatureDim == 1 && outputSpatialDims.size() == 2 &&
         outputSpatialDims[0] == 2 && outputSpatialDims[1] == 3);
    bool outputIsCHWN =
        (outputBatchDim == 3 && outputFeatureDim == 0 && outputSpatialDims.size() == 2 &&
         outputSpatialDims[0] == 1 && outputSpatialDims[1] == 2);
    // For kernel gradient: HWIO layout (spatial=[0,1], input=2, output=3)
    // The "batch" dim represents input features, "feature" dim represents output features
    bool outputIsHWIO = (outputSpatialDims.size() == 2 && outputSpatialDims[0] == 0 &&
                         outputSpatialDims[1] == 1 && outputBatchDim == 2 && outputFeatureDim == 3);
    // For kernel gradient of NCHW conv: CNHW layout (feature=0, batch=1, spatial=[2,3])
    // "feature" dim (0) holds output channels, "batch" dim (1) holds input channels
    bool outputIsCNHW =
        (outputFeatureDim == 0 && outputBatchDim == 1 && outputSpatialDims.size() == 2 &&
         outputSpatialDims[0] == 2 && outputSpatialDims[1] == 3);

    if (outputIsNCHW) {
        // NHWC -> NCHW: [0, 3, 1, 2]
        result = [g transposeTensor:result permutation:@[@0, @3, @1, @2] name:nil];
    } else if (outputIsCHWN) {
        // NHWC -> CHWN: [3, 1, 2, 0]
        result = [g transposeTensor:result permutation:@[@3, @1, @2, @0] name:nil];
    } else if (outputIsHWIO) {
        // NHWC [N, H, W, C] -> HWIO [H, W, I, O]
        // For kernel gradient, N=I (input features), C=O (output features)
        // So [N, H, W, C] -> [H, W, N, C] which is HWIO
        // Permutation: [1, 2, 0, 3]
        result = [g transposeTensor:result permutation:@[@1, @2, @0, @3] name:nil];
    } else if (outputIsCNHW) {
        // NHWC [N, H, W, C] -> CNHW [C, N, H, W]
        // For kernel gradient of NCHW conv, N=I (input features), C=O (output features)
        // Permutation: [3, 0, 1, 2]
        result = [g transposeTensor:result permutation:@[@3, @0, @1, @2] name:nil];
    } else if (!outputIsNHWC) {
        MPS_LOG_WARN("Unexpected output layout batch=%lld, feature=%lld, spatial=[%lld,%lld]\n",
                     outputBatchDim, outputFeatureDim, outputSpatialDims[0], outputSpatialDims[1]);
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.convolution", Handle_convolution);

}  // namespace jax_mps
