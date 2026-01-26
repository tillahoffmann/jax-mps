// Convolution operations for StableHLO

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Handle stablehlo.convolution
// StableHLO convolution is highly general - supports arbitrary dimension layouts,
// dilations, padding, grouped convolutions, etc.
static MPSGraphTensor* Handle_convolution(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
    if (!convOp) {
        NSLog(@"ERROR: Expected ConvolutionOp");
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

    // Currently only support 2D convolutions
    if (spatialRank != 2) {
        NSLog(@"ERROR: Only 2D convolution is currently supported, got %zu spatial dims",
              spatialRank);
        return nullptr;
    }

    // Check batch group count (used for gradient computations)
    if (batchGroupCount != 1) {
        NSLog(@"ERROR: batch_group_count != 1 not yet supported");
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

    if (!inputIsNHWC && !inputIsNCHW && !inputIsCHWN) {
        NSLog(@"ERROR: Unsupported input layout. Expected NHWC, NCHW, or CHWN. Got batch=%lld, "
              @"feature=%lld, spatial=[%lld,%lld]",
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
    } else {
        NSLog(@"ERROR: Unsupported kernel layout. Got output=%lld, input=%lld, "
              @"spatial=[%lld,%lld]",
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
    }

    // Perform convolution (or transposed convolution)
    MPSGraphTensor* result;
    if (isTransposedConv) {
        // Transposed convolution (used in backward pass of strided conv)
        // For transposed conv, MPS needs the output shape
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            NSLog(@"ERROR: Could not get result type for transposed convolution");
            return nullptr;
        }

        // Get output shape in NHWC format
        auto resultShape = resultType.getShape();
        NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];

        // The result shape is in the output dimension layout, need to convert to NHWC
        int64_t outN = resultShape[outputBatchDim];
        int64_t outH = resultShape[outputSpatialDims[0]];
        int64_t outW = resultShape[outputSpatialDims[1]];
        int64_t outC = resultShape[outputFeatureDim];

        [outputShape addObject:@(outN)];
        [outputShape addObject:@(outH)];
        [outputShape addObject:@(outW)];
        [outputShape addObject:@(outC)];

        // For transposed convolution, the kernel's I and O have swapped semantics:
        // - MPS transpose conv expects source_channels == kernel_O
        // - But StableHLO kernel has source_channels == kernel_I
        // So we need to swap I and O: OIHW [O, I, H, W] -> [I, O, H, W]
        MPSGraphTensor* transposeKernel = [g transposeTensor:mpsKernel
                                                 permutation:@[@1, @0, @2, @3]
                                                        name:nil];

        MPSGraphConvolution2DOpDescriptor* transposeDesc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1
                          strideInY:1
                    dilationRateInX:(NSUInteger)dilationW
                    dilationRateInY:(NSUInteger)dilationH
                             groups:(NSUInteger)featureGroupCount
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        transposeDesc.paddingLeft = (NSUInteger)padLeft;
        transposeDesc.paddingRight = (NSUInteger)padRight;
        transposeDesc.paddingTop = (NSUInteger)padTop;
        transposeDesc.paddingBottom = (NSUInteger)padBottom;

        result = [g convolutionTranspose2DWithSourceTensor:convInput
                                             weightsTensor:transposeKernel
                                               outputShape:outputShape
                                                descriptor:transposeDesc
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
    } else if (!outputIsNHWC) {
        NSLog(@"WARNING: Unexpected output layout batch=%lld, feature=%lld, spatial=[%lld,%lld]",
              outputBatchDim, outputFeatureDim, outputSpatialDims[0], outputSpatialDims[1]);
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.convolution", Handle_convolution);

}  // namespace jax_mps
