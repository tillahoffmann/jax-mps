#pragma once

#include <xla/pjrt/c/pjrt_c_api.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace jax_mps {

// Convert MLIR type to PJRT dtype
int MlirTypeToPjrtDtype(mlir::Type type);

}  // namespace jax_mps
