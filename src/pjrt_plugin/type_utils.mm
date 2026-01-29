#import "pjrt_plugin/type_utils.h"

namespace jax_mps {

MPSDataType PjrtDtypeToMps(int dtype) {
    switch (dtype) {
        case PJRT_Buffer_Type_F32:
            return MPSDataTypeFloat32;
        case PJRT_Buffer_Type_F16:
            return MPSDataTypeFloat16;
        case PJRT_Buffer_Type_BF16:
            return MPSDataTypeBFloat16;
        case PJRT_Buffer_Type_S32:
            return MPSDataTypeInt32;
        case PJRT_Buffer_Type_S64:
            return MPSDataTypeInt64;
        case PJRT_Buffer_Type_U32:
            return MPSDataTypeUInt32;
        case PJRT_Buffer_Type_U64:
            return MPSDataTypeUInt64;
        case PJRT_Buffer_Type_S8:
            return MPSDataTypeInt8;
        case PJRT_Buffer_Type_U8:
            return MPSDataTypeUInt8;
        case PJRT_Buffer_Type_S16:
            return MPSDataTypeInt16;
        case PJRT_Buffer_Type_U16:
            return MPSDataTypeUInt16;
        case PJRT_Buffer_Type_PRED:
            return MPSDataTypeBool;
        case PJRT_Buffer_Type_C64:
            return MPSDataTypeComplexFloat32;
        case PJRT_Buffer_Type_C128:
            return MPSDataTypeComplexFloat32;  // MPS has no complex float64
        default:
            return MPSDataTypeInvalid;
    }
}

int MpsToPjrtDtype(MPSDataType mps_type) {
    switch (mps_type) {
        case MPSDataTypeFloat32:
            return PJRT_Buffer_Type_F32;
        case MPSDataTypeFloat16:
            return PJRT_Buffer_Type_F16;
        case MPSDataTypeBFloat16:
            return PJRT_Buffer_Type_BF16;
        case MPSDataTypeInt32:
            return PJRT_Buffer_Type_S32;
        case MPSDataTypeInt64:
            return PJRT_Buffer_Type_S64;
        case MPSDataTypeUInt32:
            return PJRT_Buffer_Type_U32;
        case MPSDataTypeUInt64:
            return PJRT_Buffer_Type_U64;
        case MPSDataTypeInt8:
            return PJRT_Buffer_Type_S8;
        case MPSDataTypeUInt8:
            return PJRT_Buffer_Type_U8;
        case MPSDataTypeInt16:
            return PJRT_Buffer_Type_S16;
        case MPSDataTypeUInt16:
            return PJRT_Buffer_Type_U16;
        case MPSDataTypeBool:
            return PJRT_Buffer_Type_PRED;
        case MPSDataTypeComplexFloat32:
            return PJRT_Buffer_Type_C64;
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

    if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(type)) {
        mlir::Type elemType = complexType.getElementType();
        if (elemType.isF32() || elemType.isF64())
            return MPSDataTypeComplexFloat32;
        if (elemType.isF16())
            return MPSDataTypeComplexFloat16;
        return MPSDataTypeInvalid;
    }

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
        return PJRT_Buffer_Type_F32;
    if (elemType.isF16())
        return PJRT_Buffer_Type_F16;
    if (elemType.isBF16())
        return PJRT_Buffer_Type_BF16;
    if (elemType.isF64())
        return PJRT_Buffer_Type_F64;

    if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elemType)) {
        mlir::Type inner = complexType.getElementType();
        if (inner.isF32())
            return PJRT_Buffer_Type_C64;
        if (inner.isF64())
            return PJRT_Buffer_Type_C128;
        return -1;
    }

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();

        if (width == 1)
            return PJRT_Buffer_Type_PRED;
        if (width == 8)
            return isUnsigned ? PJRT_Buffer_Type_U8 : PJRT_Buffer_Type_S8;
        if (width == 16)
            return isUnsigned ? PJRT_Buffer_Type_U16 : PJRT_Buffer_Type_S16;
        if (width == 32)
            return isUnsigned ? PJRT_Buffer_Type_U32 : PJRT_Buffer_Type_S32;
        if (width == 64)
            return isUnsigned ? PJRT_Buffer_Type_U64 : PJRT_Buffer_Type_S64;
    }

    return -1;
}

}  // namespace jax_mps
