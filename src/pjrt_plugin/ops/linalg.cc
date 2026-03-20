// Linear algebra op handlers (dot_general, convolution, cholesky, triangular_solve).

#include <mlx/einsum.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Helper to check if a permutation is the identity (no transpose needed)
bool IsIdentityPermutation(const std::vector<int>& perm) {
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
            return false;
        }
    }
    return true;
}

// Use einsum for dot_general by default (single fused op vs transpose+reshape+matmul+reshape).
bool UseEinsumForDotGeneral() {
    static bool use_einsum = std::getenv("MPS_NO_EINSUM") == nullptr;
    return use_einsum;
}

// Build einsum subscript string for dot_general
std::string BuildEinsumSubscript(int lhsRank, int rhsRank, llvm::ArrayRef<int64_t> lhsBatchDims,
                                 llvm::ArrayRef<int64_t> rhsBatchDims,
                                 llvm::ArrayRef<int64_t> lhsContractDims,
                                 llvm::ArrayRef<int64_t> rhsContractDims) {
    char nextChar = 'a';

    std::vector<char> lhsChars(lhsRank, 0);
    std::vector<char> rhsChars(rhsRank, 0);

    // Assign shared characters for batch dims
    for (size_t i = 0; i < lhsBatchDims.size(); ++i) {
        char c = nextChar++;
        lhsChars[lhsBatchDims[i]] = c;
        rhsChars[rhsBatchDims[i]] = c;
    }

    // Assign shared characters for contracting dims
    for (size_t i = 0; i < lhsContractDims.size(); ++i) {
        char c = nextChar++;
        lhsChars[lhsContractDims[i]] = c;
        rhsChars[rhsContractDims[i]] = c;
    }

    // Assign unique characters for free dims
    for (int i = 0; i < lhsRank; ++i) {
        if (lhsChars[i] == 0) {
            lhsChars[i] = nextChar++;
        }
    }
    for (int i = 0; i < rhsRank; ++i) {
        if (rhsChars[i] == 0) {
            rhsChars[i] = nextChar++;
        }
    }

    std::string lhsSub(lhsChars.begin(), lhsChars.end());
    std::string rhsSub(rhsChars.begin(), rhsChars.end());

    // Build output: batch dims first, then lhs free dims, then rhs free dims
    std::string outSub;
    for (int64_t d : lhsBatchDims) {
        outSub += lhsChars[d];
    }
    for (int i = 0; i < lhsRank; ++i) {
        bool isBatch = std::find(lhsBatchDims.begin(), lhsBatchDims.end(), i) != lhsBatchDims.end();
        bool isContract =
            std::find(lhsContractDims.begin(), lhsContractDims.end(), i) != lhsContractDims.end();
        if (!isBatch && !isContract) {
            outSub += lhsChars[i];
        }
    }
    for (int i = 0; i < rhsRank; ++i) {
        bool isBatch = std::find(rhsBatchDims.begin(), rhsBatchDims.end(), i) != rhsBatchDims.end();
        bool isContract =
            std::find(rhsContractDims.begin(), rhsContractDims.end(), i) != rhsContractDims.end();
        if (!isBatch && !isContract) {
            outSub += rhsChars[i];
        }
    }

    return lhsSub + "," + rhsSub + "->" + outSub;
}

// Handler for stablehlo.dot_general
bool HandleDotGeneral(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        MPS_LOG_ERROR("stablehlo.dot_general: failed to cast\n");
        return false;
    }

    auto lhs_opt = GetValue(values, op->getOperand(0));
    auto rhs_opt = GetValue(values, op->getOperand(1));
    if (!lhs_opt || !rhs_opt) {
        MPS_LOG_ERROR("stablehlo.dot_general: operand not found in value map\n");
        return false;
    }

    auto& lhs = lhs_opt->get();
    auto& rhs = rhs_opt->get();
    auto dimNumbers = dotOp.getDotDimensionNumbers();

    auto lhsContractDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    auto lhsRank = static_cast<int>(lhs.ndim());
    auto rhsRank = static_cast<int>(rhs.ndim());

    // Try einsum path if enabled
    if (UseEinsumForDotGeneral()) {
        std::string subscript = BuildEinsumSubscript(lhsRank, rhsRank, lhsBatchDims, rhsBatchDims,
                                                     lhsContractDims, rhsContractDims);
        MPS_LOG_DEBUG("dot_general einsum: %s\n", subscript.c_str());
        auto result = mlx::core::einsum(subscript, {lhs, rhs});
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Standard path: transpose -> reshape -> matmul -> reshape
    std::unordered_set<int> lhsContractSet(lhsContractDims.begin(), lhsContractDims.end());
    std::unordered_set<int> rhsContractSet(rhsContractDims.begin(), rhsContractDims.end());
    std::unordered_set<int> lhsBatchSet(lhsBatchDims.begin(), lhsBatchDims.end());
    std::unordered_set<int> rhsBatchSet(rhsBatchDims.begin(), rhsBatchDims.end());

    std::vector<int> lhsFreeDims;
    std::vector<int> rhsFreeDims;
    for (int i = 0; i < lhsRank; ++i) {
        if (lhsBatchSet.count(i) == 0 && lhsContractSet.count(i) == 0) {
            lhsFreeDims.push_back(i);
        }
    }
    for (int i = 0; i < rhsRank; ++i) {
        if (rhsBatchSet.count(i) == 0 && rhsContractSet.count(i) == 0) {
            rhsFreeDims.push_back(i);
        }
    }

    std::vector<int> lhsPerm;
    for (int64_t d : lhsBatchDims)
        lhsPerm.push_back(static_cast<int>(d));
    for (int d : lhsFreeDims)
        lhsPerm.push_back(d);
    for (int64_t d : lhsContractDims)
        lhsPerm.push_back(static_cast<int>(d));

    std::vector<int> rhsPerm;
    for (int64_t d : rhsBatchDims)
        rhsPerm.push_back(static_cast<int>(d));
    for (int64_t d : rhsContractDims)
        rhsPerm.push_back(static_cast<int>(d));
    for (int d : rhsFreeDims)
        rhsPerm.push_back(d);

    auto lhsT = mlx::core::transpose(lhs, lhsPerm);
    auto rhsT = mlx::core::transpose(rhs, rhsPerm);

    int numBatch = static_cast<int>(lhsBatchDims.size());
    int numLhsFree = static_cast<int>(lhsFreeDims.size());
    int numContract = static_cast<int>(lhsContractDims.size());

    int64_t batchSize = 1;
    for (int i = 0; i < numBatch; ++i) {
        batchSize *= lhsT.shape(i);
    }

    int64_t lhsFreeSize = 1;
    for (int i = numBatch; i < numBatch + numLhsFree; ++i) {
        lhsFreeSize *= lhsT.shape(i);
    }

    int64_t contractSize = 1;
    for (int i = numBatch + numLhsFree; i < static_cast<int>(lhsT.ndim()); ++i) {
        contractSize *= lhsT.shape(i);
    }

    int64_t rhsFreeSize = 1;
    for (int i = numBatch + numContract; i < static_cast<int>(rhsT.ndim()); ++i) {
        rhsFreeSize *= rhsT.shape(i);
    }

    std::vector<int> batchShape;
    batchShape.reserve(numBatch);
    for (int i = 0; i < numBatch; ++i) {
        batchShape.push_back(lhsT.shape(i));
    }
    std::vector<int> lhsFreeShape;
    lhsFreeShape.reserve(numLhsFree);
    for (int i = numBatch; i < numBatch + numLhsFree; ++i) {
        lhsFreeShape.push_back(lhsT.shape(i));
    }
    std::vector<int> rhsFreeShape;
    rhsFreeShape.reserve(static_cast<int>(rhsT.ndim()) - numBatch - numContract);
    for (int i = numBatch + numContract; i < static_cast<int>(rhsT.ndim()); ++i) {
        rhsFreeShape.push_back(rhsT.shape(i));
    }

    mlx::core::Shape lhsShape3d = {static_cast<int>(batchSize), static_cast<int>(lhsFreeSize),
                                   static_cast<int>(contractSize)};
    mlx::core::Shape rhsShape3d = {static_cast<int>(batchSize), static_cast<int>(contractSize),
                                   static_cast<int>(rhsFreeSize)};

    auto lhs3d = mlx::core::reshape(lhsT, lhsShape3d);
    auto rhs3d = mlx::core::reshape(rhsT, rhsShape3d);

    auto result3d = mlx::core::matmul(lhs3d, rhs3d);

    mlx::core::Shape finalShape;
    for (int s : batchShape)
        finalShape.push_back(s);
    for (int s : lhsFreeShape)
        finalShape.push_back(s);
    for (int s : rhsFreeShape)
        finalShape.push_back(s);

    auto result = mlx::core::reshape(result3d, finalShape);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.convolution
bool HandleConvolution(mlir::Operation* op, ValueMap& values,
                       std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(op);
    if (!convOp) {
        MPS_LOG_ERROR("stablehlo.convolution: failed to cast\n");
        return false;
    }

    auto input_opt = GetValue(values, op->getOperand(0));
    auto kernel_opt = GetValue(values, op->getOperand(1));
    if (!input_opt || !kernel_opt) {
        MPS_LOG_ERROR("stablehlo.convolution: operand not found in value map\n");
        return false;
    }

    auto& input = input_opt->get();
    auto& kernel = kernel_opt->get();
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

    int numSpatialDims = static_cast<int>(inputSpatialDims.size());

    // Build input permutation to [N, spatial..., C_in] format
    std::vector<int> inputPerm(input.ndim());
    inputPerm[0] = static_cast<int>(inputBatchDim);
    for (int i = 0; i < numSpatialDims; ++i) {
        inputPerm[1 + i] = static_cast<int>(inputSpatialDims[i]);
    }
    inputPerm[1 + numSpatialDims] = static_cast<int>(inputFeatureDim);

    // Build kernel permutation to [C_out, spatial..., C_in] format
    std::vector<int> kernelPerm(kernel.ndim());
    kernelPerm[0] = static_cast<int>(kernelOutputFeatureDim);
    for (int i = 0; i < numSpatialDims; ++i) {
        kernelPerm[1 + i] = static_cast<int>(kernelSpatialDims[i]);
    }
    kernelPerm[1 + numSpatialDims] = static_cast<int>(kernelInputFeatureDim);

    auto inputT = IsIdentityPermutation(inputPerm) ? input : mlx::core::transpose(input, inputPerm);
    auto kernelT =
        IsIdentityPermutation(kernelPerm) ? kernel : mlx::core::transpose(kernel, kernelPerm);

    // Extract strides
    std::vector<int> strides;
    if (auto stridesAttr = convOp.getWindowStrides()) {
        for (int64_t s : *stridesAttr) {
            strides.push_back(static_cast<int>(s));
        }
    } else {
        strides.resize(numSpatialDims, 1);
    }

    // Extract padding
    std::vector<int> paddingLow(numSpatialDims, 0);
    std::vector<int> paddingHigh(numSpatialDims, 0);
    if (auto paddingAttr = convOp.getPadding()) {
        auto paddingValues = paddingAttr->getValues<int64_t>();
        auto it = paddingValues.begin();
        for (int i = 0; i < numSpatialDims; ++i) {
            paddingLow[i] = static_cast<int>(*it++);
            paddingHigh[i] = static_cast<int>(*it++);
        }
    }

    // Extract dilation
    std::vector<int> kernelDilation;
    if (auto dilationAttr = convOp.getRhsDilation()) {
        for (int64_t d : *dilationAttr) {
            kernelDilation.push_back(static_cast<int>(d));
        }
    } else {
        kernelDilation.resize(numSpatialDims, 1);
    }

    std::vector<int> inputDilation;
    if (auto dilationAttr = convOp.getLhsDilation()) {
        for (int64_t d : *dilationAttr) {
            inputDilation.push_back(static_cast<int>(d));
        }
    } else {
        inputDilation.resize(numSpatialDims, 1);
    }

    auto featureGroupCount = static_cast<int>(convOp.getFeatureGroupCount());

    auto batchGroupCount = convOp.getBatchGroupCount();
    if (batchGroupCount != 1) {
        MPS_LOG_ERROR("stablehlo.convolution: batch_group_count != 1 not supported\n");
        return false;
    }

    // Weight gradient VJP optimization
    bool useWeightGradVJP = false;
    mlx::core::array vjpResult = mlx::core::array(0);

    if (numSpatialDims == 2 && inputT.ndim() == 4 && kernelT.ndim() == 4) {
        int H_in = inputT.shape(1);
        int W_in = inputT.shape(2);
        int kH = kernelT.shape(1);
        int kW = kernelT.shape(2);
        int out_H = H_in + paddingLow[0] + paddingHigh[0] - (kH - 1) * kernelDilation[0];
        int out_W = W_in + paddingLow[1] + paddingHigh[1] - (kW - 1) * kernelDilation[1];

        if (out_H > 0 && out_W > 0 && kH >= 2 * out_H && kW >= 2 * out_W && strides[0] == 1 &&
            strides[1] == 1 && inputDilation[0] == 1 && inputDilation[1] == 1 &&
            featureGroupCount == 1) {
            useWeightGradVJP = true;

            auto& origAct = input;
            auto& origGrad = kernel;

            int Ci_true = origAct.shape(static_cast<int>(inputBatchDim));
            int Co_true = origGrad.shape(static_cast<int>(kernelOutputFeatureDim));

            std::vector<int> actPerm = {static_cast<int>(inputFeatureDim)};
            for (int i = 0; i < numSpatialDims; ++i)
                actPerm.push_back(static_cast<int>(inputSpatialDims[i]));
            actPerm.push_back(static_cast<int>(inputBatchDim));

            std::vector<int> gradPerm = {static_cast<int>(kernelInputFeatureDim)};
            for (int i = 0; i < numSpatialDims; ++i)
                gradPerm.push_back(static_cast<int>(kernelSpatialDims[i]));
            gradPerm.push_back(static_cast<int>(kernelOutputFeatureDim));

            auto actStd = mlx::core::transpose(origAct, actPerm);
            auto gradStd = mlx::core::transpose(origGrad, gradPerm);

            int origKH = out_H;
            int origKW = out_W;
            std::vector<int> fwdStride = {kernelDilation[0], kernelDilation[1]};
            std::vector<int> fwdPadLo = {paddingLow[0], paddingLow[1]};
            std::vector<int> fwdPadHi = {paddingHigh[0], paddingHigh[1]};

            auto dummyW = mlx::core::zeros({Co_true, origKH, origKW, Ci_true}, actStd.dtype());

            auto vjpFn =
                [actStd, fwdStride, fwdPadLo, fwdPadHi](
                    const std::vector<mlx::core::array>& primals) -> std::vector<mlx::core::array> {
                return {mlx::core::conv_general(actStd, primals[0], fwdStride, fwdPadLo, fwdPadHi,
                                                std::vector<int>{}, std::vector<int>{}, 1, false)};
            };

            auto [fwdOutputs, vjps] = mlx::core::vjp(vjpFn, {dummyW}, {gradStd});
            auto dW = vjps[0];

            vjpResult = mlx::core::transpose(dW, {3, 1, 2, 0});
        }
    }

    mlx::core::array convResult = mlx::core::array(0);
    if (useWeightGradVJP) {
        convResult = vjpResult;
    } else {
        convResult =
            mlx::core::conv_general(inputT, kernelT, strides, paddingLow, paddingHigh,
                                    kernelDilation, inputDilation, featureGroupCount, false);
    }

    // Build output permutation
    std::vector<int> outputPerm(convResult.ndim());
    outputPerm[outputBatchDim] = 0;
    for (int i = 0; i < numSpatialDims; ++i) {
        outputPerm[outputSpatialDims[i]] = 1 + i;
    }
    outputPerm[outputFeatureDim] = 1 + numSpatialDims;

    auto result = IsIdentityPermutation(outputPerm) ? convResult
                                                    : mlx::core::transpose(convResult, outputPerm);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.cholesky
bool HandleCholesky(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                    ExecContext& ctx) {
    auto cholOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!cholOp) {
        MPS_LOG_ERROR("stablehlo.cholesky: failed to cast\n");
        return false;
    }
    auto input_opt = GetValue(values, cholOp.getA());
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cholesky: operand not found in value map\n");
        return false;
    }
    auto& a = input_opt->get();
    bool lower = cholOp.getLower();
    auto result = mlx::core::linalg::cholesky(a, /*upper=*/!lower);
    values.emplace(ToKey(op->getResult(0)), std::move(result));
    return true;
}

// Handler for stablehlo.triangular_solve
bool HandleTriangularSolve(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: failed to cast\n");
        return false;
    }
    auto a_opt = GetValue(values, triSolveOp.getA());
    auto b_opt = GetValue(values, triSolveOp.getB());
    if (!a_opt || !b_opt) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: operand not found in value map\n");
        return false;
    }
    auto& a = a_opt->get();
    auto& b = b_opt->get();

    bool left_side = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unit_diagonal = triSolveOp.getUnitDiagonal();
    auto transpose = triSolveOp.getTransposeA();

    auto a_solve = a;
    auto b_solve = b;

    if (unit_diagonal) {
        int n = a_solve.shape()[a_solve.ndim() - 1];
        auto eye = mlx::core::eye(n, a_solve.dtype());
        a_solve = a_solve * (1.0F - eye) + eye;
    }

    // Guard against singular triangular matrices
    int n_dim = a_solve.shape()[a_solve.ndim() - 1];
    auto diag = mlx::core::diagonal(a_solve, 0, -2, -1);
    auto zero_mask = mlx::core::equal(diag, mlx::core::zeros_like(diag));
    auto batch_singular = mlx::core::any(zero_mask, std::vector<int>{-1}, false);

    auto eye_mat = mlx::core::eye(n_dim, a_solve.dtype());
    auto eps = mlx::core::array(1e-30F, a_solve.dtype());
    a_solve = a_solve + mlx::core::expand_dims(zero_mask, -1) * eye_mat * eps;

    if (transpose == mlir::stablehlo::Transpose::TRANSPOSE ||
        transpose == mlir::stablehlo::Transpose::ADJOINT) {
        auto ndim = a_solve.ndim();
        std::vector<int> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[ndim - 2], perm[ndim - 1]);
        a_solve = mlx::core::transpose(a_solve, perm);
        lower = !lower;
    }

    auto nan_guard = [&](mlx::core::array result) -> mlx::core::array {
        auto nan_val = mlx::core::full(result.shape(), std::numeric_limits<float>::quiet_NaN(),
                                       result.dtype());
        auto mask = batch_singular;
        for (int i = 0; i < 2; ++i) {
            mask = mlx::core::expand_dims(mask, -1);
        }
        return mlx::core::where(mask, nan_val, result);
    };

    try {
        if (!left_side) {
            auto ndim_a = a_solve.ndim();
            std::vector<int> perm_a(ndim_a);
            std::iota(perm_a.begin(), perm_a.end(), 0);
            std::swap(perm_a[ndim_a - 2], perm_a[ndim_a - 1]);
            a_solve = mlx::core::transpose(a_solve, perm_a);
            lower = !lower;

            auto ndim_b = b_solve.ndim();
            std::vector<int> perm_b(ndim_b);
            std::iota(perm_b.begin(), perm_b.end(), 0);
            std::swap(perm_b[ndim_b - 2], perm_b[ndim_b - 1]);
            b_solve = mlx::core::transpose(b_solve, perm_b);

            auto x_t = mlx::core::linalg::solve_triangular(a_solve, b_solve, /*upper=*/!lower,
                                                           mlx::core::Device::cpu);
            auto result = nan_guard(mlx::core::transpose(x_t, perm_b));
            values.emplace(ToKey(op->getResult(0)), std::move(result));
            return true;
        }

        auto result = nan_guard(mlx::core::linalg::solve_triangular(a_solve, b_solve,
                                                                    /*upper=*/!lower,
                                                                    mlx::core::Device::cpu));
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    } catch (const std::exception& e) {
        MPS_LOG_ERROR("stablehlo.triangular_solve: %s\n", e.what());
        return false;
    }
}

}  // namespace

void RegisterLinalgHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"stablehlo.dot_general", HandleDotGeneral});
    handlers.insert({"stablehlo.convolution", HandleConvolution});
    handlers.insert({"stablehlo.cholesky", HandleCholesky});
    handlers.insert({"stablehlo.triangular_solve", HandleTriangularSolve});
}

}  // namespace jax_mps
