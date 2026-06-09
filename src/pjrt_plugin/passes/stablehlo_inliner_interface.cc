// Permissive StableHLO/CHLO inliner interface. Compiled -fno-rtti (see this
// directory's CMake target) so it matches MLIR's build and does not reference
// `typeinfo for mlir::DialectInterface`.

#include "pjrt_plugin/passes/stablehlo_inliner_interface.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mps {
namespace {

// All StableHLO/CHLO ops are legal to inline. The func dialect's own inliner
// interface handles the func.call/func.return mechanics (argument and result
// mapping, terminator handling); this interface only tells the inliner that the
// callee body ops may be cloned into the caller.
struct PermissiveInlinerInterface : public mlir::DialectInlinerInterface {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation* /*op*/, mlir::Region* /*dest*/, bool /*wouldBeCloned*/,
                         mlir::IRMapping& /*valueMapping*/) const final {
        return true;
    }
    bool isLegalToInline(mlir::Region* /*dest*/, mlir::Region* /*src*/, bool /*wouldBeCloned*/,
                         mlir::IRMapping& /*valueMapping*/) const final {
        return true;
    }
    bool isLegalToInline(mlir::Operation* /*call*/, mlir::Operation* /*callable*/,
                         bool /*wouldBeCloned*/) const final {
        return true;
    }
};

}  // namespace

void registerStablehloInlinerInterfaces(mlir::MLIRContext& context) {
    if (auto* d = context.getLoadedDialect<mlir::stablehlo::StablehloDialect>()) {
        d->addInterfaces<PermissiveInlinerInterface>();
    }
    if (auto* d = context.getLoadedDialect<mlir::chlo::ChloDialect>()) {
        d->addInterfaces<PermissiveInlinerInterface>();
    }
}

}  // namespace mps
