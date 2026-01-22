#import "pjrt_plugin/type_utils.h"

namespace jax_mps {

MPSDataType PjrtDtypeToMps(int dtype) {
    switch (dtype) {
        case kPjrtF32:
            return MPSDataTypeFloat32;
        case kPjrtF16:
            return MPSDataTypeFloat16;
        case kPjrtBF16:
            return MPSDataTypeBFloat16;
        case kPjrtS32:
            return MPSDataTypeInt32;
        case kPjrtS64:
            return MPSDataTypeInt64;
        case kPjrtU32:
            return MPSDataTypeUInt32;
        case kPjrtU64:
            return MPSDataTypeUInt64;
        case kPjrtS8:
            return MPSDataTypeInt8;
        case kPjrtU8:
            return MPSDataTypeUInt8;
        case kPjrtS16:
            return MPSDataTypeInt16;
        case kPjrtU16:
            return MPSDataTypeUInt16;
        case kPjrtPred:
            return MPSDataTypeBool;
        default:
            return MPSDataTypeInvalid;
    }
}

int MpsToPjrtDtype(MPSDataType mps_type) {
    switch (mps_type) {
        case MPSDataTypeFloat32:
            return kPjrtF32;
        case MPSDataTypeFloat16:
            return kPjrtF16;
        case MPSDataTypeBFloat16:
            return kPjrtBF16;
        case MPSDataTypeInt32:
            return kPjrtS32;
        case MPSDataTypeInt64:
            return kPjrtS64;
        case MPSDataTypeUInt32:
            return kPjrtU32;
        case MPSDataTypeUInt64:
            return kPjrtU64;
        case MPSDataTypeInt8:
            return kPjrtS8;
        case MPSDataTypeUInt8:
            return kPjrtU8;
        case MPSDataTypeInt16:
            return kPjrtS16;
        case MPSDataTypeUInt16:
            return kPjrtU16;
        case MPSDataTypeBool:
            return kPjrtPred;
        default:
            return -1;
    }
}

MPSDataType MlirTypeToMps(mlir::Type type) {
    if (type.isF32())
        return MPSDataTypeFloat32;
    if (type.isF16())
        return MPSDataTypeFloat16;
    if (type.isBF16())
        return MPSDataTypeBFloat16;

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();

        if (width == 1)
            return MPSDataTypeBool;
        if (width == 8)
            return isUnsigned ? MPSDataTypeUInt8 : MPSDataTypeInt8;
        if (width == 16)
            return isUnsigned ? MPSDataTypeUInt16 : MPSDataTypeInt16;
        if (width == 32)
            return isUnsigned ? MPSDataTypeUInt32 : MPSDataTypeInt32;
        if (width == 64)
            return isUnsigned ? MPSDataTypeUInt64 : MPSDataTypeInt64;
    }

    return MPSDataTypeInvalid;
}

int MlirTypeToPjrtDtype(mlir::Type elemType) {
    if (elemType.isF32())
        return kPjrtF32;
    if (elemType.isF16())
        return kPjrtF16;
    if (elemType.isBF16())
        return kPjrtBF16;
    if (elemType.isF64())
        return kPjrtF64;

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();

        if (width == 1)
            return kPjrtPred;
        if (width == 8)
            return isUnsigned ? kPjrtU8 : kPjrtS8;
        if (width == 16)
            return isUnsigned ? kPjrtU16 : kPjrtS16;
        if (width == 32)
            return isUnsigned ? kPjrtU32 : kPjrtS32;
        if (width == 64)
            return isUnsigned ? kPjrtU64 : kPjrtS64;
    }

    return -1;
}

}  // namespace jax_mps
