// FFT operations: stablehlo.fft

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static MPSGraphTensor* Handle_fft(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
    if (!fftOp) {
        MPS_LOG_ERROR("Expected FftOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger rank = inputShape.count;
    auto fftLength = fftOp.getFftLength();
    NSUInteger nAxes = fftLength.size();

    if (nAxes == 0 || nAxes > rank) {
        MPS_LOG_ERROR("Invalid fft_length rank for stablehlo.fft\n");
        return nullptr;
    }

    NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:nAxes];
    NSUInteger startAxis = rank - nAxes;
    for (NSUInteger i = 0; i < nAxes; ++i) {
        [axes addObject:@(startAxis + i)];
    }

    MPSGraphFFTDescriptor* desc = [MPSGraphFFTDescriptor descriptor];
    desc.roundToOddHermitean = NO;

    auto fftType = fftOp.getFftType();
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            desc.inverse = NO;
            desc.scalingMode = MPSGraphFFTScalingModeNone;
            return [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
        case mlir::stablehlo::FftType::IFFT:
            desc.inverse = YES;
            desc.scalingMode = MPSGraphFFTScalingModeSize;
            return [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
        case mlir::stablehlo::FftType::RFFT:
            desc.inverse = NO;
            desc.scalingMode = MPSGraphFFTScalingModeNone;
            return [g realToHermiteanFFTWithTensor:input axes:axes descriptor:desc name:nil];
        case mlir::stablehlo::FftType::IRFFT:
            desc.inverse = YES;
            desc.scalingMode = MPSGraphFFTScalingModeSize;
            return [g HermiteanToRealFFTWithTensor:input axes:axes descriptor:desc name:nil];
        default:
            MPS_LOG_ERROR("Unsupported fft type\n");
            return nullptr;
    }
}
REGISTER_MPS_OP("stablehlo.fft", Handle_fft);

}  // namespace jax_mps
