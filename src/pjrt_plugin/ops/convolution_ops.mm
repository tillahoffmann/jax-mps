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

    // Check input dilation (for transposed convolution)
    if (lhsDilation) {
        auto lhsDilationVec = lhsDilation.value();
        for (auto d : lhsDilationVec) {
            if (d != 1) {
                NSLog(@"ERROR: lhs_dilation (input dilation) != 1 not yet supported");
                return nullptr;
            }
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

    if (!inputIsNHWC && !inputIsNCHW) {
        NSLog(@"ERROR: Unsupported input layout. Expected NHWC or NCHW. Got batch=%lld, "
              @"feature=%lld, spatial=[%lld,%lld]",
              inputBatchDim, inputFeatureDim, inputSpatialDims[0], inputSpatialDims[1]);
        return nullptr;
    }

    // Transpose kernel from HWIO to OHWI if needed
    MPSGraphTensor* transposedKernel = kernel;
    if (kernelIsHWIO) {
        // HWIO [H, W, I, O] -> OHWI [O, H, W, I]
        // Permutation: [3, 0, 1, 2]
        transposedKernel = [g transposeTensor:kernel permutation:@[@3, @0, @1, @2] name:nil];
    } else if (kernelIsOIHW) {
        // OIHW [O, I, H, W] -> OHWI [O, H, W, I]
        // Permutation: [0, 2, 3, 1]
        transposedKernel = [g transposeTensor:kernel permutation:@[@0, @2, @3, @1] name:nil];
    } else {
        // Check if it's already OHWI
        bool kernelIsOHWI =
            (kernelOutputFeatureDim == 0 && kernelSpatialDims.size() == 2 &&
             kernelSpatialDims[0] == 1 && kernelSpatialDims[1] == 2 && kernelInputFeatureDim == 3);
        if (!kernelIsOHWI) {
            NSLog(@"ERROR: Unsupported kernel layout. Got output=%lld, input=%lld, "
                  @"spatial=[%lld,%lld]",
                  kernelOutputFeatureDim, kernelInputFeatureDim, kernelSpatialDims[0],
                  kernelSpatialDims[1]);
            return nullptr;
        }
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
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];  // After transpose: OHWI

    // Set explicit padding
    desc.paddingLeft = (NSUInteger)padLeft;
    desc.paddingRight = (NSUInteger)padRight;
    desc.paddingTop = (NSUInteger)padTop;
    desc.paddingBottom = (NSUInteger)padBottom;

    // MPS weightsLayout OIHW means [outputChannels, inputChannels/groups, kH, kW]
    // But we transposed to OHWI format, so we need to transpose again for MPS
    // Actually, let's reconsider: MPS with weightsLayout OIHW expects [O, I, H, W]
    // We have OHWI after our transpose. Need to go OHWI -> OIHW
    // OHWI [O, H, W, I] -> OIHW [O, I, H, W]
    // Permutation: [0, 3, 1, 2]
    MPSGraphTensor* mpsKernel = [g transposeTensor:transposedKernel
                                       permutation:@[@0, @3, @1, @2]
                                              name:nil];

    // Handle NCHW input - MPS conv2d works with both layouts via dataLayout setting
    MPSGraphTensor* convInput = input;
    if (inputIsNCHW) {
        // For NCHW, transpose to NHWC first since MPS is more reliable with NHWC
        // NCHW -> NHWC: [0, 2, 3, 1]
        convInput = [g transposeTensor:input permutation:@[@0, @2, @3, @1] name:nil];
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    }

    // Perform convolution
    MPSGraphTensor* result = [g convolution2DWithSourceTensor:convInput
                                                weightsTensor:mpsKernel
                                                   descriptor:desc
                                                         name:nil];

    // If input was NCHW, transpose output back to NCHW
    if (inputIsNCHW) {
        // Check expected output layout
        bool outputIsNCHW =
            (outputBatchDim == 0 && outputFeatureDim == 1 && outputSpatialDims.size() == 2 &&
             outputSpatialDims[0] == 2 && outputSpatialDims[1] == 3);
        if (outputIsNCHW) {
            // NHWC -> NCHW: [0, 3, 1, 2]
            result = [g transposeTensor:result permutation:@[@0, @3, @1, @2] name:nil];
        }
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.convolution", Handle_convolution);

}  // namespace jax_mps
