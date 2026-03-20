// Sort, FFT, and complex op handlers.

#include <mlx/fft.h>

#include <complex>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Handler for stablehlo.sort
bool HandleSort(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        MPS_LOG_ERROR("stablehlo.sort: failed to cast\n");
        return false;
    }

    int dimension = static_cast<int>(sortOp.getDimension());
    bool isStable = sortOp.getIsStable();
    (void)isStable;  // MLX sort is always stable

    // Analyze comparator to determine sort direction
    bool ascending = true;
    auto& comparator = sortOp.getComparator();
    if (!comparator.empty()) {
        auto& block = comparator.front();
        for (auto& compOp : block.getOperations()) {
            if (auto cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(compOp)) {
                auto dir = cmpOp.getComparisonDirection();
                if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                    dir == mlir::stablehlo::ComparisonDirection::GE) {
                    ascending = false;
                }
                break;
            }
        }
    }

    size_t numInputs = sortOp.getInputs().size();

    if (numInputs == 1) {
        auto input_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.sort: input not found\n");
            return false;
        }
        auto result = mlx::core::sort(input_opt->get(), dimension);
        if (!ascending) {
            auto shape = result.shape();
            int dimSize = shape[dimension];
            mlx::core::Shape starts(result.ndim(), 0);
            mlx::core::Shape stops(shape.begin(), shape.end());
            mlx::core::Shape steps(result.ndim(), 1);
            starts[dimension] = dimSize - 1;
            stops[dimension] = -dimSize - 1;
            steps[dimension] = -1;
            result = mlx::core::slice(result, starts, stops, steps);
        }
        values.emplace(ToKey(op->getResult(0)), std::move(result));
    } else {
        // Sort-by-key
        auto keys_opt = GetValue(values, sortOp.getInputs()[0]);
        if (!keys_opt) {
            MPS_LOG_ERROR("stablehlo.sort: keys not found\n");
            return false;
        }

        auto indices = mlx::core::argsort(keys_opt->get(), dimension);
        if (!ascending) {
            // Reverse indices along the sort dimension instead of negating keys,
            // which would break for unsigned integer types.
            auto shape = indices.shape();
            int dimSize = shape[dimension];
            mlx::core::Shape starts(indices.ndim(), 0);
            mlx::core::Shape stops(shape.begin(), shape.end());
            mlx::core::Shape steps(indices.ndim(), 1);
            starts[dimension] = dimSize - 1;
            stops[dimension] = -dimSize - 1;
            steps[dimension] = -1;
            indices = mlx::core::slice(indices, starts, stops, steps);
        }

        for (size_t i = 0; i < numInputs; ++i) {
            auto input_opt = GetValue(values, sortOp.getInputs()[i]);
            if (!input_opt) {
                MPS_LOG_ERROR("stablehlo.sort: input %zu not found\n", i);
                return false;
            }
            auto sorted = mlx::core::take_along_axis(input_opt->get(), indices, dimension);
            values.emplace(ToKey(op->getResult(i)), std::move(sorted));
        }
    }

    return true;
}

// Handler for stablehlo.fft
bool HandleFft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(op);
    if (!fftOp) {
        MPS_LOG_ERROR("stablehlo.fft: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.fft: operand not found\n");
        return false;
    }

    auto& input = input_opt->get();
    auto fftType = fftOp.getFftType();
    auto fftLength = fftOp.getFftLength();

    std::vector<int> axes;
    mlx::core::Shape lengths;
    int ndim = static_cast<int>(input.ndim());
    for (size_t i = 0; i < fftLength.size(); ++i) {
        axes.push_back(ndim - static_cast<int>(fftLength.size()) + static_cast<int>(i));
        lengths.push_back(static_cast<int>(fftLength[i]));
    }

    mlx::core::array result = input;
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            result = mlx::core::fft::fftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IFFT:
            result = mlx::core::fft::ifftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::RFFT:
            result = mlx::core::fft::rfftn(input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IRFFT:
            result = mlx::core::fft::irfftn(input, lengths, axes);
            break;
        default:
            MPS_LOG_ERROR("stablehlo.fft: unsupported fft type\n");
            return false;
    }

    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.complex (combine real + imag into complex)
bool HandleComplex(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                   ExecContext& ctx) {
    auto real_opt = GetValue(values, op->getOperand(0));
    auto imag_opt = GetValue(values, op->getOperand(1));
    if (!real_opt || !imag_opt) {
        MPS_LOG_ERROR("stablehlo.complex: operand not found\n");
        return false;
    }
    auto imag_unit = mlx::core::array(std::complex<float>(0.0F, 1.0F));
    auto result = mlx::core::add(
        mlx::core::astype(real_opt->get(), mlx::core::complex64),
        mlx::core::multiply(mlx::core::astype(imag_opt->get(), mlx::core::complex64), imag_unit));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

}  // namespace

void RegisterSortFftComplexHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.sort", HandleSort});
    handlers.insert({"stablehlo.fft", HandleFft});
    handlers.insert({"stablehlo.complex", HandleComplex});
}

}  // namespace jax_mps
