// Rewrite pattern: hand-rolled attention -> stablehlo.custom_call @mps.sdpa
//
// jnn.softmax(q @ k^T * scale, axis=-1) @ v lowers (after fuse_softmax) to
//
//   %s = dot_general %q, %k, batching = [b0,b1] x [b0',b1'], contracting = [hq] x [hk]
//   %c = broadcast_in_dim %cst                  // optional scale
//   %x = multiply %s, %c                        // or divide %s, %c
//   %p = custom_call @mps.softmax(%x) {axis: 3}
//   %o = dot_general %p, %v, batching = [0,1] x [..], contracting = [3] x [sv]
//        (or dot_general %v, %p, ...: the result is then (B, N, H, T))
//
// We match the second dot_general, walk up, and emit @mps.sdpa on BNTH
// operands. See fuse_sdpa.h.

#include "pjrt_plugin/passes/fuse_sdpa.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
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

using Perm = llvm::SmallVector<int64_t, 4>;

// One operand of a rank-3 or rank-4 dot_general with rank-2 batching dims,
// one contracting dim and one free dim. (Rank 3 arises when JAX squeezes a
// size-1 batch dim, e.g. batch size 1 at inference.)
struct DotOperandDims {
    Perm batch;
    int64_t contracting, free;
};

std::optional<DotOperandDims> classifyOperand(int64_t rank, mlir::ArrayRef<int64_t> batching,
                                              mlir::ArrayRef<int64_t> contracting) {
    if (static_cast<int64_t>(batching.size()) != rank - 2 || contracting.size() != 1)
        return std::nullopt;
    int64_t free = -1;
    for (int64_t d = 0; d < rank; ++d) {
        if (llvm::is_contained(batching, d) || d == contracting[0])
            continue;
        if (free != -1)
            return std::nullopt;
        free = d;
    }
    if (free == -1)
        return std::nullopt;
    return DotOperandDims{Perm(batching.begin(), batching.end()), contracting[0], free};
}

bool isFloatOfRank(Value v, int64_t rank) {
    auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
    return type && type.getRank() == rank && type.hasStaticShape() &&
           mlir::isa<mlir::FloatType>(type.getElementType());
}

// Strips single-use reshapes that only insert or remove size-1 dims and keep
// the trailing dim, so a trailing-axis softmax stays trailing. Returns {} if a
// stripped value has other uses.
Value stripUnitReshapes(Value v) {
    while (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>()) {
        auto in = mlir::cast<RankedTensorType>(reshape.getOperand().getType()).getShape();
        auto out = mlir::cast<RankedTensorType>(reshape.getType()).getShape();
        auto nonUnit = [](mlir::ArrayRef<int64_t> shape) {
            llvm::SmallVector<int64_t, 4> dims;
            for (int64_t d : shape)
                if (d != 1)
                    dims.push_back(d);
            return dims;
        };
        if (in.empty() || out.empty() || in.back() != out.back() || nonUnit(in) != nonUnit(out))
            break;
        if (!v.hasOneUse())
            return {};
        v = reshape.getOperand();
    }
    return v;
}

// A finite, non-zero splat float constant, possibly behind a broadcast_in_dim.
std::optional<double> splatScalar(Value v) {
    if (auto bcast = v.getDefiningOp<stablehlo::BroadcastInDimOp>())
        v = bcast.getOperand();
    mlir::DenseElementsAttr attr;
    if (!mlir::matchPattern(v, mlir::m_Constant(&attr)) || !attr.isSplat())
        return std::nullopt;
    if (!mlir::isa<mlir::FloatType>(attr.getElementType()))
        return std::nullopt;
    auto f = attr.getSplatValue<llvm::APFloat>();
    bool losesInfo = false;
    f.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    double d = f.convertToDouble();
    if (!std::isfinite(d) || d == 0.0)
        return std::nullopt;
    return d;
}

// `v` is a single-use @mps.softmax over its trailing axis. Returns its input.
Value matchTrailingSoftmax(Value v) {
    auto call = v.getDefiningOp<stablehlo::CustomCallOp>();
    if (!call || !v.hasOneUse() || call.getCallTargetName() != "mps.softmax" ||
        call->getNumOperands() != 1)
        return {};
    auto config = mlir::dyn_cast_or_null<mlir::StringAttr>(call.getBackendConfigAttr());
    auto rank = mlir::cast<RankedTensorType>(v.getType()).getRank();
    if (!config || config.getValue() != "{\"axis\":" + std::to_string(rank - 1) + "}")
        return {};
    return call->getOperand(0);
}

// Returns `v` permuted by `perm` (result dim i = v dim perm[i]). Folds a
// defining transpose into the permutation and emits nothing for identity.
Value permute(PatternRewriter& rewriter, mlir::Location loc, Value v, Perm perm) {
    if (auto transpose = v.getDefiningOp<stablehlo::TransposeOp>()) {
        auto inner = transpose.getPermutation();
        for (auto& p : perm)
            p = inner[p];
        v = transpose.getOperand();
    }
    bool identity = true;
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i)
        identity &= perm[i] == i;
    if (identity)
        return v;
    return stablehlo::TransposeOp::create(rewriter, loc, v, rewriter.getDenseI64ArrayAttr(perm));
}

// Permutes a dot operand into (batch..., rows, cols) and, at rank 3, adds a
// leading unit batch dim so it is rank 4 as MLX's SDPA requires.
Value toRank4(PatternRewriter& rewriter, mlir::Location loc, Value v, const DotOperandDims& dims,
              bool contractingLast) {
    Perm perm = dims.batch;
    perm.push_back(contractingLast ? dims.free : dims.contracting);
    perm.push_back(contractingLast ? dims.contracting : dims.free);
    v = permute(rewriter, loc, v, perm);
    auto type = mlir::cast<RankedTensorType>(v.getType());
    if (type.getRank() == 4)
        return v;
    llvm::SmallVector<int64_t, 4> shape{1};
    shape.append(type.getShape().begin(), type.getShape().end());
    return stablehlo::ReshapeOp::create(rewriter, loc,
                                        RankedTensorType::get(shape, type.getElementType()), v);
}

class FuseSdpaPattern : public mlir::OpRewritePattern<stablehlo::DotGeneralOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::DotGeneralOp pvDot,
                                  PatternRewriter& rewriter) const override {
        auto resultType = mlir::cast<RankedTensorType>(pvDot.getType());
        int64_t rank = resultType.getRank();
        if ((rank != 3 && rank != 4) || !isFloatOfRank(pvDot.getResult(), rank))
            return mlir::failure();
        int64_t numBatch = rank - 2;

        // --- Weights @ values. The softmax output may be either operand. ---
        auto pvDims = pvDot.getDotDimensionNumbers();
        Value scaled;
        bool probsOnLhs = false;
        for (bool lhs : {true, false}) {
            Value cand = stripUnitReshapes(lhs ? pvDot.getLhs() : pvDot.getRhs());
            if (!cand)
                continue;
            scaled = matchTrailingSoftmax(cand);
            if (scaled) {
                probsOnLhs = lhs;
                break;
            }
        }
        if (!scaled)
            return mlir::failure();
        Value v = probsOnLhs ? pvDot.getRhs() : pvDot.getLhs();
        auto probsDims = probsOnLhs ? classifyOperand(rank, pvDims.getLhsBatchingDimensions(),
                                                      pvDims.getLhsContractingDimensions())
                                    : classifyOperand(rank, pvDims.getRhsBatchingDimensions(),
                                                      pvDims.getRhsContractingDimensions());
        auto vDims = probsOnLhs ? classifyOperand(rank, pvDims.getRhsBatchingDimensions(),
                                                  pvDims.getRhsContractingDimensions())
                                : classifyOperand(rank, pvDims.getLhsBatchingDimensions(),
                                                  pvDims.getLhsContractingDimensions());
        // Weights are (batch..., T, S) and must be consumed as such: leading
        // batch dims in order, contracting S (the trailing dim).
        if (!probsDims || !vDims || probsDims->contracting != rank - 1)
            return mlir::failure();
        for (int64_t i = 0; i < numBatch; ++i) {
            if (probsDims->batch[i] != i)
                return mlir::failure();
        }
        if (!isFloatOfRank(v, rank))
            return mlir::failure();

        // --- Optional constant scale between scores and softmax. ---
        scaled = stripUnitReshapes(scaled);
        if (!scaled)
            return mlir::failure();
        Value scores = scaled;
        double scale = 1.0;
        if (auto mul = scaled.getDefiningOp<stablehlo::MulOp>()) {
            if (auto c = splatScalar(mul.getRhs())) {
                scores = mul.getLhs();
                scale = *c;
            } else if (auto c = splatScalar(mul.getLhs())) {
                scores = mul.getRhs();
                scale = *c;
            } else {
                return mlir::failure();
            }
        } else if (auto div = scaled.getDefiningOp<stablehlo::DivOp>()) {
            auto c = splatScalar(div.getRhs());
            if (!c)
                return mlir::failure();
            scores = div.getLhs();
            scale = 1.0 / *c;
        }
        if (scores != scaled) {
            if (!scaled.hasOneUse())
                return mlir::failure();
            scores = stripUnitReshapes(scores);
            if (!scores)
                return mlir::failure();
        }

        // --- Queries @ keys -> (batch..., T, S). Q must be the lhs so T
        // comes before S. ---
        auto qkDot = scores.getDefiningOp<stablehlo::DotGeneralOp>();
        if (!qkDot || !scores.hasOneUse() || !isFloatOfRank(scores, rank))
            return mlir::failure();
        auto qkDims = qkDot.getDotDimensionNumbers();
        auto qDims = classifyOperand(rank, qkDims.getLhsBatchingDimensions(),
                                     qkDims.getLhsContractingDimensions());
        auto kDims = classifyOperand(rank, qkDims.getRhsBatchingDimensions(),
                                     qkDims.getRhsContractingDimensions());
        Value q = qkDot.getLhs();
        Value k = qkDot.getRhs();
        if (!qDims || !kDims || !isFloatOfRank(q, rank) || !isFloatOfRank(k, rank))
            return mlir::failure();

        auto elemType = resultType.getElementType();
        for (Value operand : {q, k, v}) {
            if (mlir::cast<RankedTensorType>(operand.getType()).getElementType() != elemType)
                return mlir::failure();
        }
        // The fused kernel needs matching query/value head dims.
        auto qShape = mlir::cast<RankedTensorType>(q.getType()).getShape();
        auto vShape = mlir::cast<RankedTensorType>(v.getType()).getShape();
        int64_t t = qShape[qDims->free];
        int64_t h = vShape[vDims->free];
        if (qShape[qDims->contracting] != h)
            return mlir::failure();
        // dot(p, v) is (batch..., T, H) = the sdpa layout; dot(v, p) is
        // (batch..., H, T). Checked before any op is created: a failing
        // pattern must not mutate the IR.
        llvm::SmallVector<int64_t, 4> pvShape;
        for (int64_t b : qDims->batch)
            pvShape.push_back(qShape[b]);
        llvm::SmallVector<int64_t, 4> sdpaShape(rank == 3 ? 1 : 0, 1);
        sdpaShape.append(pvShape.begin(), pvShape.end());
        sdpaShape.append({t, h});
        pvShape.append({probsOnLhs ? t : h, probsOnLhs ? h : t});
        if (resultType.getShape() != mlir::ArrayRef<int64_t>(pvShape))
            return mlir::failure();

        auto loc = pvDot.getLoc();
        Value q4 = toRank4(rewriter, loc, q, *qDims, /*contractingLast=*/true);
        Value k4 = toRank4(rewriter, loc, k, *kDims, /*contractingLast=*/true);
        Value v4 = toRank4(rewriter, loc, v, *vDims, /*contractingLast=*/false);

        auto maskType = RankedTensorType::get({}, rewriter.getI1Type());
        Value mask = stablehlo::ConstantOp::create(
            rewriter, loc, mlir::DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true)));

        std::array<char, 64> backendConfig{};
        std::snprintf(backendConfig.data(), backendConfig.size(), "{\"scale\": %.17g}", scale);

        llvm::SmallVector<Value, 4> operands{q4, k4, v4, mask};
        llvm::SmallVector<mlir::Type, 1> resultTypes{RankedTensorType::get(sdpaShape, elemType)};
        auto customCall = stablehlo::CustomCallOp::create(
            rewriter, loc, resultTypes, operands,
            /*call_target_name=*/llvm::StringRef("mps.sdpa"),
            /*has_side_effect=*/false,
            /*backend_config=*/rewriter.getStringAttr(backendConfig.data()),
            /*api_version=*/stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            /*called_computations=*/rewriter.getArrayAttr({}),
            /*operand_layouts=*/mlir::ArrayAttr(),
            /*result_layouts=*/mlir::ArrayAttr(),
            /*output_operand_aliases=*/rewriter.getArrayAttr({}),
            /*result_tilings=*/mlir::ArrayAttr());

        Value result = customCall.getResult(0);
        if (!probsOnLhs) {
            result = stablehlo::TransposeOp::create(rewriter, loc, result,
                                                    rewriter.getDenseI64ArrayAttr({0, 1, 3, 2}));
        }
        if (rank == 3)
            result = stablehlo::ReshapeOp::create(rewriter, loc, resultType, result);
        rewriter.replaceOp(pvDot, result);
        return mlir::success();
    }
};

}  // namespace

void populateFuseSdpaPatterns(mlir::RewritePatternSet& patterns) {
    patterns.add<FuseSdpaPattern>(patterns.getContext());
}

}  // namespace mps
