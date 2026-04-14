// Rewrite pattern: stable softmax chain -> stablehlo.custom_call @mps.softmax
//
// Post-simplification, jax.nn.softmax(x, axis=k) lowers to:
//
//   %max0 = reduce_max(x, dims=[k])                     // x.shape minus k
//   %max  = maximum(broadcast(-inf), %max0)             // simplification
//                                                       //   leaves this in
//   %mkr  = reshape(%max)        -> shape with 1 at k   // keepdims insertion
//   %mbr  = broadcast_in_dim(%mkr) -> x.shape
//   %sub  = subtract(x, %mbr)
//   %exp  = exponential(%sub)
//   %sum0 = reduce_add(%exp, dims=[k])
//   %skr  = reshape(%sum0)       -> shape with 1 at k
//   %sbr  = broadcast_in_dim(%skr) -> x.shape
//   %out  = divide(%exp, %sbr)
//
// We match the DivideOp root, walk up, and emit
//   stablehlo.custom_call @mps.softmax(x) { backend_config = "{\"axis\":k}" }
// Works for any single reduction axis — k is extracted from the reduce ops.

#include "pjrt_plugin/passes/fuse_softmax.h"

#include <string>

#include "mlir/IR/Builders.h"
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
using mlir::RankedTensorType;
using mlir::Value;

// Walk upward through reshape ops that only toggle size-1 dims.
Value stripTrivialReshapes(Value v) {
    while (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>()) {
        v = reshape.getOperand();
    }
    return v;
}

// Matches a broadcast_in_dim that expands a keepdims tensor (size 1 at `axis`,
// matching target elsewhere) to `targetShape`. Returns the pre-broadcast value
// (after reshape-stripping). Otherwise {}.
Value stripAxisKeepdimsBroadcast(Value v, mlir::ArrayRef<int64_t> targetShape, int64_t axis) {
    auto bcast = v.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcast)
        return {};
    auto outType = mlir::cast<RankedTensorType>(bcast.getType());
    if (outType.getShape() != targetShape)
        return {};
    auto srcType = mlir::cast<RankedTensorType>(bcast.getOperand().getType());
    if (srcType.getRank() != outType.getRank())
        return {};
    // Source dims: size-1 at `axis`, matching target at other dims.
    for (int64_t i = 0; i < srcType.getRank(); ++i) {
        int64_t expected = (i == axis) ? 1 : targetShape[i];
        if (srcType.getShape()[i] != expected)
            return {};
    }
    // Broadcast dims must be identity (no permutation).
    auto dims = bcast.getBroadcastDimensions();
    if (static_cast<int64_t>(dims.size()) != srcType.getRank())
        return {};
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != static_cast<int64_t>(i))
            return {};
    }
    return stripTrivialReshapes(bcast.getOperand());
}

// `maximum(broadcast(-inf), x)` (either operand order) -> x. Unchanged if not.
Value stripNegInfMaximum(Value v) {
    auto maxOp = v.getDefiningOp<stablehlo::MaxOp>();
    if (!maxOp)
        return v;

    auto isNegInf = [](Value cand) -> bool {
        if (auto bc = cand.getDefiningOp<stablehlo::BroadcastInDimOp>())
            cand = bc.getOperand();
        mlir::DenseElementsAttr attr;
        if (!mlir::matchPattern(cand, mlir::m_Constant(&attr)))
            return false;
        auto eltType = attr.getElementType();
        if (!eltType.isF32() && !eltType.isF16() && !eltType.isBF16() && !eltType.isF64())
            return false;
        if (!attr.isSplat())
            return false;
        auto f = attr.getSplatValue<llvm::APFloat>();
        return f.isNegative() && f.isInfinity();
    };

    if (isNegInf(maxOp.getLhs()))
        return maxOp.getRhs();
    if (isNegInf(maxOp.getRhs()))
        return maxOp.getLhs();
    return v;
}

// Match a reduce over a single axis. Returns (input, axis) if matched.
template <typename ReducerOp>
std::pair<Value, int64_t> matchSingleAxisReduce(Value v) {
    auto reduce = v.getDefiningOp<stablehlo::ReduceOp>();
    if (!reduce)
        return {Value{}, -1};
    if (reduce.getInputs().size() != 1 || reduce.getNumResults() != 1)
        return {Value{}, -1};
    auto dims = reduce.getDimensions();
    if (dims.size() != 1)
        return {Value{}, -1};
    auto& block = reduce.getBody().front();
    if (block.getOperations().size() != 2)
        return {Value{}, -1};
    auto combiner = mlir::dyn_cast<ReducerOp>(&block.front());
    if (!combiner)
        return {Value{}, -1};
    auto returnOp = mlir::dyn_cast<stablehlo::ReturnOp>(&block.back());
    if (!returnOp || returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != combiner.getResult())
        return {Value{}, -1};
    return {reduce.getInputs()[0], dims[0]};
}

class FuseSoftmaxPattern : public mlir::OpRewritePattern<stablehlo::DivOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::DivOp divOp,
                                  PatternRewriter& rewriter) const override {
        auto resultType = mlir::cast<RankedTensorType>(divOp.getType());
        if (!mlir::isa<mlir::FloatType>(resultType.getElementType()))
            return mlir::failure();
        if (resultType.getRank() < 1)
            return mlir::failure();

        // LHS: exponential(subtract(x, shift)).
        auto expOp = divOp.getLhs().getDefiningOp<stablehlo::ExpOp>();
        if (!expOp)
            return mlir::failure();
        auto subOp = expOp.getOperand().getDefiningOp<stablehlo::SubtractOp>();
        if (!subOp)
            return mlir::failure();

        Value x = subOp.getLhs();
        if (mlir::cast<RankedTensorType>(x.getType()).getShape() != resultType.getShape())
            return mlir::failure();

        // RHS: broadcast(keepdims)(reduce_sum(exp, axis=k_sum)).
        // We find k_sum first by inspecting the reduce, then verify the
        // broadcast pattern uses that axis.
        auto sumBcast = divOp.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!sumBcast)
            return mlir::failure();
        auto [sumReduceInput, sumAxis] =
            matchSingleAxisReduce<stablehlo::AddOp>(stripTrivialReshapes(sumBcast.getOperand()));
        if (!sumReduceInput || sumAxis < 0)
            return mlir::failure();
        if (sumReduceInput != expOp.getResult())
            return mlir::failure();
        Value sumBcastInput =
            stripAxisKeepdimsBroadcast(divOp.getRhs(), resultType.getShape(), sumAxis);
        if (!sumBcastInput)
            return mlir::failure();

        // shift = broadcast(keepdims)(strip-neg-inf(reduce_max(x, axis=k_max))).
        auto maxBcast = subOp.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!maxBcast)
            return mlir::failure();
        Value maxInner = stripNegInfMaximum(stripTrivialReshapes(maxBcast.getOperand()));
        auto [maxReduceInput, maxAxis] = matchSingleAxisReduce<stablehlo::MaxOp>(maxInner);
        if (!maxReduceInput || maxAxis < 0)
            return mlir::failure();
        if (maxReduceInput != x)
            return mlir::failure();
        if (maxAxis != sumAxis)
            return mlir::failure();
        Value shiftBcastInput =
            stripAxisKeepdimsBroadcast(subOp.getRhs(), resultType.getShape(), maxAxis);
        if (!shiftBcastInput)
            return mlir::failure();

        // Encode the softmax axis in a minimal backend_config JSON. Use the
        // positive axis so the runtime handler doesn't have to know the rank.
        std::string backendConfig = "{\"axis\":" + std::to_string(sumAxis) + "}";

        llvm::SmallVector<Value, 1> operands{x};
        llvm::SmallVector<mlir::Type, 1> resultTypes{resultType};
        auto customCall = stablehlo::CustomCallOp::create(
            rewriter, divOp.getLoc(), resultTypes, operands,
            /*call_target_name=*/llvm::StringRef("mps.softmax"),
            /*has_side_effect=*/false,
            /*backend_config=*/rewriter.getStringAttr(backendConfig),
            /*api_version=*/stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            /*called_computations=*/rewriter.getArrayAttr({}),
            /*operand_layouts=*/mlir::ArrayAttr(),
            /*result_layouts=*/mlir::ArrayAttr(),
            /*output_operand_aliases=*/rewriter.getArrayAttr({}));

        rewriter.replaceOp(divOp, customCall.getResults());
        return mlir::success();
    }
};

}  // namespace

void populateFuseSoftmaxPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FuseSoftmaxPattern>(patterns.getContext());
}

}  // namespace mps
