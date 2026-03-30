// Sort, FFT, and complex op handlers.

#include <mlx/fft.h>

#include <complex>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Reverse an array along a given axis using gather with a reversed index array.
mlx::core::array ReverseAxisImpl(const mlx::core::array& a, int axis) {
    int dimSize = a.shape(axis);
    if (dimSize <= 1)
        return a;
    // Build reversed indices: [dimSize-1, dimSize-2, ..., 0]
    auto fwd = mlx::core::arange(dimSize - 1, -1, -1, mlx::core::int32);
    return mlx::core::take(a, fwd, axis);
}

// Build a descending sort key: ascending sort of the key gives descending order
// of the original values, preserving stable tie ordering.
mlx::core::array DescendingKey(const mlx::core::array& input) {
    auto dtype = input.dtype();
    if (mlx::core::issubdtype(dtype, mlx::core::inexact)) {
        // Float/complex: negate (safe, no overflow issues).
        return mlx::core::negative(input);
    }
    // Integer (signed or unsigned): bitwise NOT reverses order without overflow.
    // Unsigned: NOT(0)=MAX, NOT(MAX)=0. Signed: NOT(MIN)=MAX, NOT(MAX)=MIN.
    return mlx::core::bitwise_xor(input, mlx::core::full(input.shape(), -1, dtype));
}

// Compute top-k values and indices along the last axis.
// Uses a descending sort key + ascending argsort + take-first-k to preserve stable
// tie ordering (equal values keep lowest-index-first).
std::pair<mlx::core::array, mlx::core::array> TopKImplFn(const mlx::core::array& input, int k) {
    int axis = static_cast<int>(input.ndim()) - 1;
    auto key = DescendingKey(input);
    auto allIndices = mlx::core::argsort(key, axis);

    mlx::core::Shape starts(allIndices.ndim(), 0);
    mlx::core::Shape stops(allIndices.shape().begin(), allIndices.shape().end());
    stops[axis] = k;
    auto indices = mlx::core::slice(allIndices, starts, stops);

    auto topValues = mlx::core::take_along_axis(input, indices, axis);
    return {topValues, mlx::core::astype(indices, mlx::core::int32)};
}

// Analyze a sort comparator to determine sort direction.
// The comparator block has args (lhs0, rhs0, lhs1, rhs1, ...) where
// lhs_i/rhs_i are the pair for input i. A "normal" ascending comparator
// returns compare LT, %arg0, %arg1 (lhs < rhs). But optimization passes
// may produce compare GT, %arg1, %arg0 (rhs > lhs) which is equivalent.
// We check which block arguments the compare uses to handle this correctly.
bool DetectAscending(mlir::stablehlo::SortOp sortOp) {
    auto& comparator = sortOp.getComparator();
    if (comparator.empty())
        return true;

    auto& block = comparator.front();
    // Find the compare op that feeds the return value
    auto& returnOp = block.back();
    mlir::stablehlo::CompareOp cmpOp = nullptr;
    if (returnOp.getNumOperands() > 0) {
        mlir::Value result = returnOp.getOperand(0);
        if (auto* defOp = result.getDefiningOp()) {
            cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(defOp);
            if (!cmpOp) {
                // Try through select: select(eq, tiebreak, primary)
                if (auto selOp = mlir::dyn_cast<mlir::stablehlo::SelectOp>(defOp)) {
                    if (auto* falseDefOp = selOp.getOnFalse().getDefiningOp()) {
                        cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(falseDefOp);
                    }
                    if (!cmpOp) {
                        if (auto* trueDefOp = selOp.getOnTrue().getDefiningOp()) {
                            cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(trueDefOp);
                        }
                    }
                }
                // Try through or: or(primary_lt, and(eq, tiebreak))
                if (!cmpOp) {
                    if (auto orOp = mlir::dyn_cast<mlir::stablehlo::OrOp>(defOp)) {
                        if (auto* lhsDefOp = orOp.getLhs().getDefiningOp()) {
                            cmpOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(lhsDefOp);
                        }
                    }
                }
            }
        }
    }
    // Fallback: scan for first CompareOp in the block
    if (!cmpOp) {
        for (auto& compOp : block.getOperations()) {
            if (auto found = mlir::dyn_cast<mlir::stablehlo::CompareOp>(compOp)) {
                cmpOp = found;
                break;
            }
        }
    }
    if (!cmpOp)
        return true;

    auto dir = cmpOp.getComparisonDirection();
    bool isGtGe = (dir == mlir::stablehlo::ComparisonDirection::GT ||
                   dir == mlir::stablehlo::ComparisonDirection::GE);
    // Check if operands are swapped (e.g., GT(%arg1, %arg0) = LT(%arg0, %arg1)).
    bool swapped = false;
    if (auto lhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getLhs())) {
        if (auto rhsArg = mlir::dyn_cast<mlir::BlockArgument>(cmpOp.getRhs())) {
            swapped = (lhsArg.getArgNumber() % 2 == 1) && (rhsArg.getArgNumber() % 2 == 0);
        }
    }
    return swapped ? isGtGe : !isGtGe;
}

// Handler for stablehlo.sort
bool HandleSort(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto sortOp = CastOp<mlir::stablehlo::SortOp>(op, "stablehlo.sort");
    if (!sortOp)
        return false;

    int dimension = static_cast<int>(sortOp.getDimension());
    bool ascending = DetectAscending(sortOp);
    size_t numInputs = sortOp.getInputs().size();

    if (numInputs == 1) {
        auto* input = RequireValue(values, sortOp.getInputs()[0], "stablehlo.sort");
        if (!input)
            return false;
        auto result = mlx::core::sort(*input, dimension);
        if (!ascending) {
            result = ReverseAxis(result, dimension);
        }
        values.emplace(ToKey(op->getResult(0)), std::move(result));
    } else {
        // Sort-by-key
        auto* keys = RequireValue(values, sortOp.getInputs()[0], "stablehlo.sort");
        if (!keys)
            return false;

        auto indices = mlx::core::argsort(*keys, dimension);
        if (!ascending) {
            indices = ReverseAxis(indices, dimension);
        }

        for (size_t i = 0; i < numInputs; ++i) {
            auto* input = RequireValue(values, sortOp.getInputs()[i], "stablehlo.sort");
            if (!input)
                return false;
            auto sorted = mlx::core::take_along_axis(*input, indices, dimension);
            values.emplace(ToKey(op->getResult(i)), std::move(sorted));
        }
    }

    return true;
}

// Handler for stablehlo.fft
bool HandleFft(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
               ExecContext& ctx) {
    auto fftOp = CastOp<mlir::stablehlo::FftOp>(op, "stablehlo.fft");
    if (!fftOp)
        return false;

    auto* input = RequireValue(values, op->getOperand(0), "stablehlo.fft");
    if (!input)
        return false;

    auto fftType = fftOp.getFftType();
    auto fftLength = fftOp.getFftLength();

    std::vector<int> axes;
    mlx::core::Shape lengths;
    int ndim = static_cast<int>(input->ndim());
    for (size_t i = 0; i < fftLength.size(); ++i) {
        axes.push_back(ndim - static_cast<int>(fftLength.size()) + static_cast<int>(i));
        lengths.push_back(static_cast<int>(fftLength[i]));
    }

    mlx::core::array result = *input;
    switch (fftType) {
        case mlir::stablehlo::FftType::FFT:
            result = mlx::core::fft::fftn(*input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IFFT:
            result = mlx::core::fft::ifftn(*input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::RFFT:
            result = mlx::core::fft::rfftn(*input, lengths, axes);
            break;
        case mlir::stablehlo::FftType::IRFFT:
            result = mlx::core::fft::irfftn(*input, lengths, axes);
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
    auto* real = RequireValue(values, op->getOperand(0), "stablehlo.complex");
    auto* imag = RequireValue(values, op->getOperand(1), "stablehlo.complex");
    if (!real || !imag)
        return false;
    auto imag_unit = mlx::core::array(std::complex<float>(0.0F, 1.0F));
    auto result = mlx::core::add(
        mlx::core::astype(*real, mlx::core::complex64),
        mlx::core::multiply(mlx::core::astype(*imag, mlx::core::complex64), imag_unit));
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for chlo.top_k
bool HandleChloTopK(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto topKOp = CastOp<mlir::chlo::TopKOp>(op, "chlo.top_k");
    if (!topKOp)
        return false;

    auto* input_ptr = RequireValue(values, topKOp.getOperand(), "chlo.top_k");
    if (!input_ptr)
        return false;

    int k = static_cast<int>(topKOp.getK());
    auto input = mlx::core::contiguous(*input_ptr);
    auto [topValues, indices] = TopKImplFn(input, k);
    values.emplace(ToKey(op->getResult(0)), std::move(topValues));
    values.emplace(ToKey(op->getResult(1)), std::move(indices));
    return true;
}

}  // namespace

// Public API implementations declared in handler_utils.h
mlx::core::array ReverseAxis(const mlx::core::array& a, int axis) {
    return ReverseAxisImpl(a, axis);
}

std::pair<mlx::core::array, mlx::core::array> TopKImpl(const mlx::core::array& input, int k) {
    return TopKImplFn(input, k);
}

void RegisterSortFftComplexHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.sort", HandleSort});
    handlers.insert({"stablehlo.fft", HandleFft});
    handlers.insert({"stablehlo.complex", HandleComplex});
    handlers.insert({"chlo.top_k", HandleChloTopK});
}

}  // namespace jax_mps
