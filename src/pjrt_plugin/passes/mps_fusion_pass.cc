#include "pjrt_plugin/passes/mps_fusion_pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "pjrt_plugin/passes/canonicalize_broadcast_identity.h"
#include "pjrt_plugin/passes/fuse_bias_add.h"
#include "pjrt_plugin/passes/fuse_layer_norm.h"
#include "pjrt_plugin/passes/fuse_rms_norm.h"
#include "pjrt_plugin/passes/fuse_softmax.h"

namespace mps {

namespace {

struct MpsFusionPass
    : public mlir::PassWrapper<MpsFusionPass, mlir::OperationPass<mlir::func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MpsFusionPass)

    llvm::StringRef getArgument() const final {
        return "mps-fusion";
    }
    llvm::StringRef getDescription() const final {
        return "Fuse high-level patterns into mps.* custom_calls";
    }

protected:
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        // Clean up broadcasted arithmetic identities (x - broadcast(0), etc.)
        // first so the fusion matchers below see canonical IR. The greedy
        // driver runs all patterns to a fixpoint, so a folded no-op is gone
        // before a matcher re-examines its consumer.
        populateCanonicalizeBroadcastIdentityPatterns(patterns);
        populateFuseBiasAddPatterns(patterns);
        populateFuseSoftmaxPatterns(patterns);
        populateFuseLayerNormPatterns(patterns);
        populateFuseRmsNormPatterns(patterns);
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createMpsFusionPass() {
    return std::make_unique<MpsFusionPass>();
}

}  // namespace mps
