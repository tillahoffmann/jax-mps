#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <xla/pjrt/c/pjrt_c_api.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace jax_mps {

// Convert PJRT dtype to MPSDataType
MPSDataType PjrtDtypeToMps(int dtype);

// Convert MPSDataType to PJRT dtype
int MpsToPjrtDtype(MPSDataType mps_type);

// Convert MLIR type to MPSDataType
MPSDataType MlirTypeToMps(mlir::Type type);

// Convert MLIR type to PJRT dtype
int MlirTypeToPjrtDtype(mlir::Type type);

}  // namespace jax_mps
