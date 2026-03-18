#include "pjrt_plugin/type_utils.h"

namespace jax_mps {

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
