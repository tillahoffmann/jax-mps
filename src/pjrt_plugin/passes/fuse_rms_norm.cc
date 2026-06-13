// Rewrite pattern: RMSNorm decomposition
//   -> stablehlo.custom_call @mps.rms_norm(x, weight) {"eps": e}
//
// Matches the StableHLO that flax.linen.RMSNorm / nnx.RMSNorm emits with
// use_scale = True. Post-simplification that is:
//
//   %sq    = multiply(x, x)
//   %sumsq = reduce_add(%sq, dims=[last])
//   %msq   = divide(%sumsq, N)                 // mean(x^2)
//   %rs    = rsqrt(add(reshape(%msq), eps))
//   %out   = multiply(x, multiply(broadcast(%rs), broadcast(weight)))
//
// We root at the trailing MulOp and walk up. MLX's mlx::core::fast::rms_norm
// computes `x / sqrt(mean(x^2) + eps) * weight` over the LAST axis with a 1-D
// weight of size x.shape[-1], so we only fire under exactly those conditions.
//
// Flax shares its norm code with LayerNorm and emits a `subtract(x,
// broadcast(0))` centering no-op for RMSNorm; that is folded away by
// canonicalize_broadcast_identity (which runs first in MpsFusionPass), so by
// the time this matcher runs the root MulOp sees x directly.

#include "pjrt_plugin/passes/fuse_rms_norm.h"

#include <cstdio>
#include <string>

#include "llvm/ADT/APFloat.h"
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

// Walk through reshape ops that only insert/remove size-1 dims down to the
// 1-D weight of size `trailingSize`.
Value findUnderlying1d(Value v, int64_t trailingSize) {
    while (true) {
        auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
        if (!type || type.getRank() == 0)
            return {};
        if (type.getRank() == 1)
            return (type.getShape()[0] == trailingSize) ? v : Value{};
        auto shape = type.getShape();
        if (shape.back() != trailingSize)
            return {};
        for (int64_t i = 0; i + 1 < type.getRank(); ++i)
            if (shape[i] != 1)
                return {};
        auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>();
        if (!reshape)
            return {};
        v = reshape.getOperand();
    }
}

// Match broadcast_in_dim producing a trailing-dim broadcast of a 1-D param of
// size resultShape.back(). Returns the 1-D param or null.
Value matchTrailingParamBroadcast(Value v, mlir::ArrayRef<int64_t> resultShape) {
    auto op = v.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!op || resultShape.empty())
        return {};
    if (mlir::cast<RankedTensorType>(op.getType()).getShape() != resultShape)
        return {};
    int64_t trailing = resultShape.back();
    auto src = mlir::cast<RankedTensorType>(op.getOperand().getType());
    auto dims = op.getBroadcastDimensions();
    if (dims.empty() || dims.size() != static_cast<size_t>(src.getRank()))
        return {};
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t srcDim = src.getShape()[i];
        int64_t targetDim = dims[i];
        if (i + 1 == dims.size()) {
            if (srcDim != trailing || targetDim != static_cast<int64_t>(resultShape.size()) - 1)
                return {};
        } else if (srcDim != 1) {
            return {};
        }
    }
    return findUnderlying1d(op.getOperand(), trailing);
}

// Strip reshapes that only add/remove size-1 dims (preserve non-1 dim order).
Value stripTrivialReshapes(Value v) {
    while (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>()) {
        auto inType = mlir::cast<RankedTensorType>(reshape.getOperand().getType());
        auto outType = mlir::cast<RankedTensorType>(reshape.getType());
        auto nonOne = [](mlir::ArrayRef<int64_t> shape) {
            llvm::SmallVector<int64_t, 4> out;
            for (int64_t d : shape)
                if (d != 1)
                    out.push_back(d);
            return out;
        };
        if (nonOne(inType.getShape()) != nonOne(outType.getShape()))
            break;
        v = reshape.getOperand();
    }
    return v;
}

bool isSplatZero(Value v) {
    mlir::DenseElementsAttr attr;
    if (auto bc = v.getDefiningOp<stablehlo::BroadcastInDimOp>())
        v = bc.getOperand();
    if (!mlir::matchPattern(v, mlir::m_Constant(&attr)) || !attr.isSplat())
        return false;
    if (!mlir::isa<mlir::FloatType>(attr.getElementType()))
        return false;
    return attr.getSplatValue<llvm::APFloat>().isZero();
}

// Match a single-axis reduce_add with init 0. Returns (input, axis) or {null,-1}.
std::pair<Value, int64_t> matchSumReduce(Value v) {
    auto reduce = v.getDefiningOp<stablehlo::ReduceOp>();
    if (!reduce || reduce.getInputs().size() != 1 || reduce.getNumResults() != 1 ||
        reduce.getInitValues().size() != 1)
        return {Value{}, -1};
    if (!isSplatZero(reduce.getInitValues()[0]))
        return {Value{}, -1};
    auto dims = reduce.getDimensions();
    if (dims.size() != 1)
        return {Value{}, -1};
    auto& block = reduce.getBody().front();
    if (block.getOperations().size() != 2)
        return {Value{}, -1};
    auto add = mlir::dyn_cast<stablehlo::AddOp>(&block.front());
    auto ret = mlir::dyn_cast<stablehlo::ReturnOp>(&block.back());
    if (!add || !ret || ret.getNumOperands() != 1 || ret.getOperand(0) != add.getResult())
        return {Value{}, -1};
    return {reduce.getInputs()[0], dims[0]};
}

// Match `divide(reduce_add(input, [axis]), N)` where N == input.shape[axis].
// Returns (input, axis) for the matched mean, or {null,-1}.
std::pair<Value, int64_t> matchMean(Value v) {
    auto div = v.getDefiningOp<stablehlo::DivOp>();
    if (!div)
        return {Value{}, -1};
    auto [input, axis] = matchSumReduce(stripTrivialReshapes(div.getLhs()));
    if (!input || axis < 0)
        return {Value{}, -1};
    auto inType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inType || axis >= inType.getRank())
        return {Value{}, -1};
    mlir::DenseElementsAttr attr;
    Value denom = div.getRhs();
    if (auto bc = denom.getDefiningOp<stablehlo::BroadcastInDimOp>())
        denom = bc.getOperand();
    if (!mlir::matchPattern(denom, mlir::m_Constant(&attr)) || !attr.isSplat() ||
        !mlir::isa<mlir::FloatType>(attr.getElementType()))
        return {Value{}, -1};
    double n = attr.getSplatValue<llvm::APFloat>().convertToDouble();
    if (n != static_cast<double>(inType.getShape()[axis]))
        return {Value{}, -1};
    return {input, axis};
}

// Extract the eps splat from `add(reshape(meanSq), eps)` (either operand
// order). Returns true and sets `eps` / the meanSq operand on success.
bool matchMeanSqPlusEps(stablehlo::AddOp addOp, float& eps, Value& meanSqOut) {
    auto tryEps = [&](Value epsCand, Value other) -> bool {
        mlir::DenseElementsAttr attr;
        Value c = epsCand;
        if (auto bc = c.getDefiningOp<stablehlo::BroadcastInDimOp>())
            c = bc.getOperand();
        if (!mlir::matchPattern(c, mlir::m_Constant(&attr)) || !attr.isSplat() ||
            !mlir::isa<mlir::FloatType>(attr.getElementType()))
            return false;
        double e = attr.getSplatValue<llvm::APFloat>().convertToDouble();
        if (!(e > 0.0 && e < 1.0))
            return false;
        eps = static_cast<float>(e);
        meanSqOut = other;
        return true;
    };
    return tryEps(addOp.getRhs(), addOp.getLhs()) || tryEps(addOp.getLhs(), addOp.getRhs());
}

class FuseRmsNormPattern : public mlir::OpRewritePattern<stablehlo::MulOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::MulOp mulOp,
                                  PatternRewriter& rewriter) const override {
        auto resultType = mlir::cast<RankedTensorType>(mulOp.getType());
        if (!mlir::isa<mlir::FloatType>(resultType.getElementType()) || resultType.getRank() < 1)
            return mlir::failure();
        auto resultShape = resultType.getShape();
        int64_t lastAxis = resultType.getRank() - 1;

        // Root: multiply(x_centered, scale) where scale = rsqrt * weight.
        // The scale side is itself a multiply of two broadcasts; identify it by
        // trying both operand orderings.
        Value xVal;
        Value scaleVal;
        for (int swap = 0; swap < 2; ++swap) {
            Value cand = swap ? mulOp.getRhs() : mulOp.getLhs();
            Value other = swap ? mulOp.getLhs() : mulOp.getRhs();
            if (other.getDefiningOp<stablehlo::MulOp>()) {
                xVal = cand;
                scaleVal = other;
                break;
            }
        }
        if (!scaleVal)
            return mlir::failure();

        // x = the normalized input (the Flax `subtract(x, broadcast(0))`
        // centering no-op was already folded by canonicalize_broadcast_identity).
        Value x = xVal;
        if (mlir::cast<RankedTensorType>(x.getType()).getShape() != resultShape)
            return mlir::failure();

        // scale = multiply(broadcast(rsqrt), broadcast(weight)) in either order.
        auto scaleMul = scaleVal.getDefiningOp<stablehlo::MulOp>();
        Value weight;
        Value rsqrtBcast;
        for (int swap = 0; swap < 2; ++swap) {
            Value w = matchTrailingParamBroadcast(swap ? scaleMul.getLhs() : scaleMul.getRhs(),
                                                  resultShape);
            if (w) {
                weight = w;
                rsqrtBcast = swap ? scaleMul.getRhs() : scaleMul.getLhs();
                break;
            }
        }
        if (!weight)
            return mlir::failure();

        // rsqrtBcast = broadcast(rsqrt(add(reshape(mean(x^2)), eps))).
        auto rsBcastOp = rsqrtBcast.getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!rsBcastOp ||
            mlir::cast<RankedTensorType>(rsBcastOp.getType()).getShape() != resultShape)
            return mlir::failure();
        auto rsqrtOp =
            stripTrivialReshapes(rsBcastOp.getOperand()).getDefiningOp<stablehlo::RsqrtOp>();
        if (!rsqrtOp)
            return mlir::failure();
        auto msqPlusEps = rsqrtOp.getOperand().getDefiningOp<stablehlo::AddOp>();
        if (!msqPlusEps)
            return mlir::failure();
        float eps = 1e-6F;
        Value meanSqVal;
        if (!matchMeanSqPlusEps(msqPlusEps, eps, meanSqVal))
            return mlir::failure();

        // meanSq = mean(x*x) over the trailing axis of x.
        auto [msqInput, msqAxis] = matchMean(stripTrivialReshapes(meanSqVal));
        if (!msqInput || msqAxis != lastAxis)
            return mlir::failure();
        auto sqMul = msqInput.getDefiningOp<stablehlo::MulOp>();
        if (!sqMul || sqMul.getLhs() != x || sqMul.getRhs() != x)
            return mlir::failure();

        // Weight must be 1-D of size x.shape[-1] (kernel contract).
        auto wType = mlir::cast<RankedTensorType>(weight.getType());
        if (wType.getRank() != 1 || wType.getShape()[0] != resultShape.back())
            return mlir::failure();

        // Serialize eps with full float precision (%.9g >= max_digits10 for
        // float, scientific notation for small values). std::to_string prints
        // only 6 decimals, rounding eps <= 1e-7 to "0.000000" and silently
        // changing the fused op's epsilon (LLVM's JSON parser accepts the
        // scientific form on the runtime side).
        char epsBuf[32];
        std::snprintf(epsBuf, sizeof(epsBuf), "%.9g", static_cast<double>(eps));
        std::string backendConfig = std::string("{\"eps\": ") + epsBuf + "}";
        llvm::SmallVector<Value, 2> operands{x, weight};
        llvm::SmallVector<mlir::Type, 1> resultTypes{resultType};
        auto customCall = stablehlo::CustomCallOp::create(
            rewriter, mulOp.getLoc(), resultTypes, operands,
            /*call_target_name=*/llvm::StringRef("mps.rms_norm"),
            /*has_side_effect=*/false,
            /*backend_config=*/rewriter.getStringAttr(backendConfig),
            /*api_version=*/stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            /*called_computations=*/rewriter.getArrayAttr({}),
            /*operand_layouts=*/mlir::ArrayAttr(),
            /*result_layouts=*/mlir::ArrayAttr(),
            /*output_operand_aliases=*/rewriter.getArrayAttr({}));

        rewriter.replaceOp(mulOp, customCall.getResults());
        return mlir::success();
    }
};

}  // namespace

void populateFuseRmsNormPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FuseRmsNormPattern>(patterns.getContext());
}

}  // namespace mps
