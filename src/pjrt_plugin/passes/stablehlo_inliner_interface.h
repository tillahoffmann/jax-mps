#ifndef PJRT_PLUGIN_PASSES_STABLEHLO_INLINER_INTERFACE_H
#define PJRT_PLUGIN_PASSES_STABLEHLO_INLINER_INTERFACE_H

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace mps {

// Register a permissive DialectInlinerInterface on the StableHLO and CHLO
// dialects loaded in `context`, so MLIR's inliner can inline func.call ops whose
// callee bodies contain StableHLO/CHLO ops (jnp.std/var/numpyro lower their guard
// `where` to func.call @_where, etc.). Without an inliner interface the stock
// inliner cannot prove those ops are legal to inline and silently no-ops.
//
// This is defined in the -fno-rtti pass translation unit on purpose: MLIR is
// built -fno-rtti and never emits `typeinfo for mlir::DialectInterface`, so a
// subclass compiled with RTTI fails to link (undefined typeinfo). Call after the
// dialects are loaded into the context.
void registerStablehloInlinerInterfaces(mlir::MLIRContext& context);

}  // namespace mps

#endif  // PJRT_PLUGIN_PASSES_STABLEHLO_INLINER_INTERFACE_H
