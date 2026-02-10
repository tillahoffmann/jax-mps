// FFT operations: stablehlo.fft

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static ProcessResult HandleFft(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
    if (!fftOp) {
        return ProcessResult::Error("fft: expected FftOp");
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return ProcessResult::Error("fft: missing input tensor");

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger rank = inputShape.count;
    auto fftLength = fftOp.getFftLength();
    NSUInteger nAxes = fftLength.size();

    if (nAxes == 0 || nAxes > rank) {
        return ProcessResult::Error("fft: invalid fft_length rank");
    }

    NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:nAxes];
    NSUInteger startAxis = rank - nAxes;
    for (NSUInteger i = 0; i < nAxes; ++i) {
        [axes addObject:@(startAxis + i)];
    }

    MPSGraphFFTDescriptor* desc = [MPSGraphFFTDescriptor descriptor];
    desc.roundToOddHermitean = NO;

    MPSGraphTensor* result = nil;
    auto fftType = fftOp.getFftType();
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            desc.inverse = NO;
            desc.scalingMode = MPSGraphFFTScalingModeNone;
            result = [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
            break;
        case mlir::stablehlo::FftType::IFFT:
            desc.inverse = YES;
            desc.scalingMode = MPSGraphFFTScalingModeSize;
            result = [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
            break;
        case mlir::stablehlo::FftType::RFFT:
            desc.inverse = NO;
            desc.scalingMode = MPSGraphFFTScalingModeNone;
            result = [g realToHermiteanFFTWithTensor:input axes:axes descriptor:desc name:nil];
            break;
        case mlir::stablehlo::FftType::IRFFT:
            desc.inverse = YES;
            desc.scalingMode = MPSGraphFFTScalingModeSize;
            // For IRFFT, fft_length specifies the output size. When it's odd, we must
            // set roundToOddHermitean so MPS produces the correct output length.
            desc.roundToOddHermitean = (fftLength.back() % 2 == 1) ? YES : NO;
            result = [g HermiteanToRealFFTWithTensor:input axes:axes descriptor:desc name:nil];
            break;
        default:
            return ProcessResult::Error("fft: unsupported fft type");
    }

    return Result(values, op, result, "fft");
}
REGISTER_MPS_OP("stablehlo.fft", HandleFft);

}  // namespace jax_mps
