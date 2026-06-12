// Fold broadcasted arithmetic identities that the upstream StableHLO
// aggressive-simplification pass leaves in place. It folds scalar `x + 0`,
// `x - 0`, `x * 1`, but NOT the `broadcast_in_dim(splat)` forms that appear in
// real lowerings (e.g. flax RMSNorm emits `subtract(x, broadcast(0))` as a
// centering no-op). Folding those here, before the mps.* fusion patterns run,
// lets the fusion matchers see clean IR.
//
// SAFETY: every fold requires the op's result type to equal x's type. When the
// splat operand is the broadcast-up target (x smaller than the result), the op
// is a broadcast of x, not an identity — folding to x would change the result
// shape. The shape guard rejects that. `subtract` only folds with the zero on
// the RHS (`0 - x` is negation, not identity).

#include "pjrt_plugin/passes/canonicalize_broadcast_identity.h"

#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mps {

namespace {

namespace stablehlo = mlir::stablehlo;

using mlir::LogicalResult;
using mlir::PatternRewriter;
using mlir::Value;

// True if `v` is a splat float constant equal to `target`, looking through a
// single broadcast_in_dim of a splat constant.
bool isSplatFloat(Value v, double target) {
    mlir::DenseElementsAttr attr;
    if (auto bc = v.getDefiningOp<stablehlo::BroadcastInDimOp>())
        v = bc.getOperand();
    if (!mlir::matchPattern(v, mlir::m_Constant(&attr)) || !attr.isSplat())
        return false;
    if (!mlir::isa<mlir::FloatType>(attr.getElementType()))
        return false;
    return attr.getSplatValue<llvm::APFloat>().convertToDouble() == target;
}

// Fold add(x, 0) / add(0, x) -> x when the result type matches x's type.
class FoldAddZero : public mlir::OpRewritePattern<stablehlo::AddOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(stablehlo::AddOp op, PatternRewriter& rewriter) const override {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value keep;
        if (isSplatFloat(rhs, 0.0))
            keep = lhs;
        else if (isSplatFloat(lhs, 0.0))
            keep = rhs;
        if (!keep || keep.getType() != op.getType())
            return mlir::failure();
        rewriter.replaceOp(op, keep);
        return mlir::success();
    }
};

// Fold subtract(x, 0) -> x (RHS-zero only) when result type matches x's type.
class FoldSubZero : public mlir::OpRewritePattern<stablehlo::SubtractOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(stablehlo::SubtractOp op,
                                  PatternRewriter& rewriter) const override {
        Value lhs = op.getLhs();
        if (!isSplatFloat(op.getRhs(), 0.0) || lhs.getType() != op.getType())
            return mlir::failure();
        rewriter.replaceOp(op, lhs);
        return mlir::success();
    }
};

// Fold multiply(x, 1) / multiply(1, x) -> x when result type matches x's type.
class FoldMulOne : public mlir::OpRewritePattern<stablehlo::MulOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(stablehlo::MulOp op, PatternRewriter& rewriter) const override {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value keep;
        if (isSplatFloat(rhs, 1.0))
            keep = lhs;
        else if (isSplatFloat(lhs, 1.0))
            keep = rhs;
        if (!keep || keep.getType() != op.getType())
            return mlir::failure();
        rewriter.replaceOp(op, keep);
        return mlir::success();
    }
};

}  // namespace

void populateCanonicalizeBroadcastIdentityPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FoldAddZero, FoldSubZero, FoldMulOne>(patterns.getContext());
}

}  // namespace mps
