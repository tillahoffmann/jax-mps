// Rewrite pattern: add(dot_general(x, w), broadcast_in_dim(bias))
//                  -> stablehlo.custom_call @mps.addmm(x, w, bias)
//
// Matches the common Linear(bias=True) lowering where a 1-D bias is broadcast
// over the trailing output dimension of a matmul and added. MLX exposes this
// fused kernel as mlx::core::addmm (~30% faster than separate matmul + add on
// large output dims like vocab projections).

#include "pjrt_plugin/passes/fuse_bias_add.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mps {

namespace {

namespace stablehlo = mlir::stablehlo;

using mlir::LogicalResult;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::Value;

// Standard matmul layout: contracting dim = last of lhs / first of rhs after
// optional leading shared batch dims.
bool hasStandardMatmulLayout(stablehlo::DotGeneralOp op) {
    auto dims = op.getDotDimensionNumbers();
    auto lhs = mlir::cast<RankedTensorType>(op.getLhs().getType());
    auto lhsContract = dims.getLhsContractingDimensions();
    auto rhsContract = dims.getRhsContractingDimensions();
    auto lhsBatch = dims.getLhsBatchingDimensions();
    auto rhsBatch = dims.getRhsBatchingDimensions();
    if (lhsContract.size() != 1 || rhsContract.size() != 1)
        return false;
    if (lhsContract[0] != lhs.getRank() - 1)
        return false;
    if (rhsContract[0] != static_cast<int64_t>(lhsBatch.size()))
        return false;
    if (lhsBatch.size() != rhsBatch.size())
        return false;
    for (size_t i = 0; i < lhsBatch.size(); ++i) {
        if (lhsBatch[i] != static_cast<int64_t>(i) || rhsBatch[i] != static_cast<int64_t>(i))
            return false;
    }
    return true;
}

// Walk through reshape ops that only insert/remove size-1 dims to the
// underlying 1-D bias of size `trailingSize`.
Value findUnderlyingBias(Value v, int64_t trailingSize) {
    while (true) {
        auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
        if (!type || type.getRank() == 0)
            return {};  // scalar / non-ranked: not a trailing-dim bias
        if (type.getRank() == 1) {
            return (type.getShape()[0] == trailingSize) ? v : Value{};
        }
        auto shape = type.getShape();
        if (shape.back() != trailingSize)
            return {};
        for (int64_t i = 0; i + 1 < type.getRank(); ++i) {
            if (shape[i] != 1)
                return {};
        }
        auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>();
        if (!reshape)
            return {};
        v = reshape.getOperand();
    }
}

// Match broadcast_in_dim producing a (..., V) trailing-dim broadcast of a 1-D
// bias of size V. Returns the 1-D bias or null.
Value matchTrailingBiasBroadcast(stablehlo::BroadcastInDimOp op,
                                 mlir::ArrayRef<int64_t> resultShape) {
    if (resultShape.empty())
        return {};
    int64_t trailing = resultShape.back();
    auto src = mlir::cast<RankedTensorType>(op.getOperand().getType());
    auto dims = op.getBroadcastDimensions();
    if (dims.size() != static_cast<size_t>(src.getRank()))
        return {};
    if (dims.empty())
        return {};  // scalar broadcast isn't a trailing-dim bias
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t srcDim = src.getShape()[i];
        int64_t targetDim = dims[i];
        if (i + 1 == dims.size()) {
            if (srcDim != trailing || targetDim != static_cast<int64_t>(resultShape.size()) - 1)
                return {};
        } else {
            if (srcDim != 1)
                return {};
        }
    }
    return findUnderlyingBias(op.getOperand(), trailing);
}

class FuseBiasAddPattern : public mlir::OpRewritePattern<stablehlo::AddOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::AddOp addOp,
                                  PatternRewriter& rewriter) const override {
        Value mmVal = addOp.getLhs();
        Value bcastVal = addOp.getRhs();
        auto dotOp = mmVal.getDefiningOp<stablehlo::DotGeneralOp>();
        auto bcastOp = bcastVal.getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!dotOp || !bcastOp) {
            mmVal = addOp.getRhs();
            bcastVal = addOp.getLhs();
            dotOp = mmVal.getDefiningOp<stablehlo::DotGeneralOp>();
            bcastOp = bcastVal.getDefiningOp<stablehlo::BroadcastInDimOp>();
        }
        if (!dotOp || !bcastOp)
            return mlir::failure();

        // Only fuse when the dot result feeds this add alone — otherwise the
        // rewrite forces a recomputation of the matmul elsewhere.
        if (!mmVal.hasOneUse())
            return mlir::failure();

        if (!hasStandardMatmulLayout(dotOp))
            return mlir::failure();

        auto resultType = mlir::cast<RankedTensorType>(addOp.getType());
        if (!mlir::isa<mlir::FloatType>(resultType.getElementType()))
            return mlir::failure();

        Value bias = matchTrailingBiasBroadcast(bcastOp, resultType.getShape());
        if (!bias)
            return mlir::failure();

        llvm::SmallVector<Value, 3> operands{dotOp.getLhs(), dotOp.getRhs(), bias};
        llvm::SmallVector<mlir::Type, 1> resultTypes{resultType};
        auto customCall = stablehlo::CustomCallOp::create(
            rewriter, addOp.getLoc(), resultTypes, operands,
            /*call_target_name=*/llvm::StringRef("mps.addmm"),
            /*has_side_effect=*/false,
            /*backend_config=*/mlir::Attribute(),
            /*api_version=*/stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            /*called_computations=*/rewriter.getArrayAttr({}),
            /*operand_layouts=*/mlir::ArrayAttr(),
            /*result_layouts=*/mlir::ArrayAttr(),
            /*output_operand_aliases=*/rewriter.getArrayAttr({}));

        rewriter.replaceOp(addOp, customCall.getResults());
        return mlir::success();
    }
};

}  // namespace

void populateFuseBiasAddPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FuseBiasAddPattern>(patterns.getContext());
}

}  // namespace mps
