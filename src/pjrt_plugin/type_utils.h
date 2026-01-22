#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace jax_mps {

// PJRT dtype constants (from pjrt_c_api.h)
constexpr int kPjrtPred = 1;
constexpr int kPjrtS8 = 2;
constexpr int kPjrtS16 = 3;
constexpr int kPjrtS32 = 4;
constexpr int kPjrtS64 = 5;
constexpr int kPjrtU8 = 6;
constexpr int kPjrtU16 = 7;
constexpr int kPjrtU32 = 8;
constexpr int kPjrtU64 = 9;
constexpr int kPjrtF16 = 10;
constexpr int kPjrtF32 = 11;
constexpr int kPjrtF64 = 12;
constexpr int kPjrtBF16 = 16;

// Convert PJRT dtype to MPSDataType
MPSDataType PjrtDtypeToMps(int dtype);

// Convert MPSDataType to PJRT dtype
int MpsToPjrtDtype(MPSDataType mps_type);

// Convert MLIR type to MPSDataType
MPSDataType MlirTypeToMps(mlir::Type type);

// Convert MLIR type to PJRT dtype
int MlirTypeToPjrtDtype(mlir::Type type);

}  // namespace jax_mps
