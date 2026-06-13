// Rewrite pattern: affine LayerNorm decomposition
//   -> stablehlo.custom_call @mps.layer_norm(x, weight, bias) {"eps": e}
//
// Matches the StableHLO that flax.linen.LayerNorm / nnx.LayerNorm emits with
// use_scale = use_bias = True (the default). Post-simplification that is:
//
//   %sum   = reduce_add(x, dims=[last])          // -> x.shape minus last
//   %mean  = divide(%sum, N)                     // E[x]
//   %sq    = multiply(x, x)
//   %sumsq = reduce_add(%sq, dims=[last])
//   %msq   = divide(%sumsq, N)                   // E[x^2]
//   %mean2 = multiply(%mean, %mean)              // E[x]^2
//   %var0  = subtract(%msq, %mean2)              // E[x^2] - E[x]^2
//   %var   = maximum(0, %var0)                   // clamp negatives from fp error
//   %rs    = rsqrt(add(reshape(%var), eps))
//   %cen   = subtract(x, broadcast(reshape(%mean)))
//   %out   = add( multiply(%cen, multiply(broadcast(%rs), broadcast(weight))),
//                 broadcast(bias) )
//
// We root at the trailing bias AddOp and walk up. MLX's
// mlx::core::fast::layer_norm normalizes over the LAST axis with 1-D
// weight/bias of size x.shape[-1], so we only fire under exactly those
// conditions. The `maximum(0, var)` clamp is optional in the match (it is
// numerically a no-op for the kernel, which clamps internally), but the
// E[x^2]-E[x]^2 variance shape is required.

#include "pjrt_plugin/passes/fuse_layer_norm.h"

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
// 1-D affine param of size `trailingSize` (shared with fuse_bias_add's logic).
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
    // Divisor must be a splat constant equal to the reduced extent N.
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

// Extract the eps splat from `add(reshape(var), eps)` (either operand order).
// Returns true and sets `eps` / the var operand on success.
bool matchVarPlusEps(stablehlo::AddOp addOp, float& eps, Value& varOut) {
    auto tryEps = [&](Value epsCand, Value varCand) -> bool {
        mlir::DenseElementsAttr attr;
        Value c = epsCand;
        if (auto bc = c.getDefiningOp<stablehlo::BroadcastInDimOp>())
            c = bc.getOperand();
        if (!mlir::matchPattern(c, mlir::m_Constant(&attr)) || !attr.isSplat() ||
            !mlir::isa<mlir::FloatType>(attr.getElementType()))
            return false;
        double e = attr.getSplatValue<llvm::APFloat>().convertToDouble();
        // eps is a small positive constant; reject 0 / large values that would
        // make this an ordinary add rather than the variance-stabilizer.
        if (!(e > 0.0 && e < 1.0))
            return false;
        eps = static_cast<float>(e);
        varOut = varCand;
        return true;
    };
    return tryEps(addOp.getRhs(), addOp.getLhs()) || tryEps(addOp.getLhs(), addOp.getRhs());
}

// Verify `v` is the E[x^2] - E[x]^2 variance over `axis` of `x`. The optional
// maximum(0, .) clamp is stripped first.
bool matchVariance(Value v, Value x, int64_t axis) {
    if (auto maxOp = v.getDefiningOp<stablehlo::MaxOp>()) {
        if (isSplatZero(maxOp.getLhs()))
            v = maxOp.getRhs();
        else if (isSplatZero(maxOp.getRhs()))
            v = maxOp.getLhs();
    }
    auto sub = v.getDefiningOp<stablehlo::SubtractOp>();
    if (!sub)
        return false;
    // lhs = E[x^2] = mean(x*x); rhs = E[x]^2 = mean(x)*mean(x).
    auto [sqInput, sqAxis] = matchMean(sub.getLhs());
    if (!sqInput || sqAxis != axis)
        return false;
    auto sqMul = sqInput.getDefiningOp<stablehlo::MulOp>();
    if (!sqMul || sqMul.getLhs() != x || sqMul.getRhs() != x)
        return false;
    auto meanSq = sub.getRhs().getDefiningOp<stablehlo::MulOp>();
    if (!meanSq)
        return false;
    auto [m1Input, m1Axis] = matchMean(stripTrivialReshapes(meanSq.getLhs()));
    auto [m2Input, m2Axis] = matchMean(stripTrivialReshapes(meanSq.getRhs()));
    if (!m1Input || !m2Input || m1Input != x || m2Input != x || m1Axis != axis || m2Axis != axis)
        return false;
    return true;
}

class FuseLayerNormPattern : public mlir::OpRewritePattern<stablehlo::AddOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::AddOp addOp,
                                  PatternRewriter& rewriter) const override {
        auto resultType = mlir::cast<RankedTensorType>(addOp.getType());
        if (!mlir::isa<mlir::FloatType>(resultType.getElementType()) || resultType.getRank() < 1)
            return mlir::failure();
        auto resultShape = resultType.getShape();
        int64_t lastAxis = resultType.getRank() - 1;

        // Root: add(scaled, broadcast(bias)) in either operand order.
        Value scaledVal = addOp.getLhs();
        Value bias = matchTrailingParamBroadcast(addOp.getRhs(), resultShape);
        if (!bias) {
            scaledVal = addOp.getRhs();
            bias = matchTrailingParamBroadcast(addOp.getLhs(), resultShape);
        }
        if (!bias)
            return mlir::failure();

        // scaled = multiply(centered, scale) in either operand order, where
        // scale = multiply(broadcast(rsqrt), broadcast(weight)).
        auto scaledMul = scaledVal.getDefiningOp<stablehlo::MulOp>();
        if (!scaledMul)
            return mlir::failure();

        // Identify which side is the centered (x - mean) term and which is the
        // rsqrt*weight scale term by trying both orderings.
        Value centeredVal;
        Value scaleVal;
        for (int swap = 0; swap < 2; ++swap) {
            centeredVal = swap ? scaledMul.getRhs() : scaledMul.getLhs();
            scaleVal = swap ? scaledMul.getLhs() : scaledMul.getRhs();
            if (centeredVal.getDefiningOp<stablehlo::SubtractOp>())
                break;
            centeredVal = Value{};
        }
        if (!centeredVal)
            return mlir::failure();

        // scaleVal = multiply(broadcast(rsqrt), broadcast(weight)).
        auto scaleMul = scaleVal.getDefiningOp<stablehlo::MulOp>();
        if (!scaleMul)
            return mlir::failure();
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

        // rsqrtBcast = broadcast(rsqrt(add(reshape(var), eps))).
        auto rsBcastOp = rsqrtBcast.getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!rsBcastOp ||
            mlir::cast<RankedTensorType>(rsBcastOp.getType()).getShape() != resultShape)
            return mlir::failure();
        auto rsqrtOp =
            stripTrivialReshapes(rsBcastOp.getOperand()).getDefiningOp<stablehlo::RsqrtOp>();
        if (!rsqrtOp)
            return mlir::failure();
        auto varPlusEps = rsqrtOp.getOperand().getDefiningOp<stablehlo::AddOp>();
        if (!varPlusEps)
            return mlir::failure();
        float eps = 1e-5F;
        Value varVal;
        if (!matchVarPlusEps(varPlusEps, eps, varVal))
            return mlir::failure();

        // centered = subtract(x, broadcast(reshape(mean(x, last)))).
        auto centeredSub = centeredVal.getDefiningOp<stablehlo::SubtractOp>();
        Value x = centeredSub.getLhs();
        if (mlir::cast<RankedTensorType>(x.getType()).getShape() != resultShape)
            return mlir::failure();
        auto meanBcast = centeredSub.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
        if (!meanBcast ||
            mlir::cast<RankedTensorType>(meanBcast.getType()).getShape() != resultShape)
            return mlir::failure();
        auto [meanInput, meanAxis] = matchMean(stripTrivialReshapes(meanBcast.getOperand()));
        if (!meanInput || meanInput != x || meanAxis != lastAxis)
            return mlir::failure();

        // Variance must be E[x^2]-E[x]^2 over the same trailing axis of x.
        if (!matchVariance(stripTrivialReshapes(varVal), x, lastAxis))
            return mlir::failure();

        // Weight and bias must be 1-D of size x.shape[-1] (kernel contract).
        auto wType = mlir::cast<RankedTensorType>(weight.getType());
        auto bType = mlir::cast<RankedTensorType>(bias.getType());
        if (wType.getRank() != 1 || bType.getRank() != 1 ||
            wType.getShape()[0] != resultShape.back() || bType.getShape()[0] != resultShape.back())
            return mlir::failure();

        // Serialize eps with full float precision (%.9g >= max_digits10 for
        // float, and uses scientific notation for small values). std::to_string
        // prints only 6 decimals, which rounds eps <= 1e-7 to "0.000000" and
        // would silently change the fused op's epsilon (LLVM's JSON parser on
        // the runtime side accepts scientific notation).
        char epsBuf[32];
        std::snprintf(epsBuf, sizeof(epsBuf), "%.9g", static_cast<double>(eps));
        std::string backendConfig = std::string("{\"eps\": ") + epsBuf + "}";
        llvm::SmallVector<Value, 3> operands{x, weight, bias};
        llvm::SmallVector<mlir::Type, 1> resultTypes{resultType};
        auto customCall = stablehlo::CustomCallOp::create(
            rewriter, addOp.getLoc(), resultTypes, operands,
            /*call_target_name=*/llvm::StringRef("mps.layer_norm"),
            /*has_side_effect=*/false,
            /*backend_config=*/rewriter.getStringAttr(backendConfig),
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

void populateFuseLayerNormPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FuseLayerNormPattern>(patterns.getContext());
}

}  // namespace mps
