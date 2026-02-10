// Tensor creation operations: constant, iota

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Constant creation - creates a constant tensor from MLIR constant op
static ProcessResult Handle_constant(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
        return ProcessResult::Error("constant: expected ConstantOp");
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("constant: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    auto value = constantOp.getValue();

    // Check for empty tensor (any dimension is 0)
    // MPSGraph doesn't support empty tensors, so create a minimal [1] tensor instead
    // The scatter handler will detect empty indices based on MLIR types and handle appropriately
    bool isEmpty = false;
    for (NSNumber* dim in shape) {
        if ([dim integerValue] == 0) {
            isEmpty = true;
            break;
        }
    }
    if (isEmpty) {
        // Create a minimal tensor with shape [1] and a dummy value
        // This is safe because operations that use this tensor will detect
        // empty dimensions from the MLIR types and not actually use the tensor values
        MPSGraphTensor* result = [g constantWithScalar:0 shape:@[@1] dataType:dtype];
        SetOutputTensor(values, op, result);
        return ProcessResult{};
    }

    MPSGraphTensor* result = nil;
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
        // Check if it's a splat (single value broadcast to all elements)
        if (denseAttr.isSplat()) {
            auto elemType = denseAttr.getElementType();
            double scalarValue = 0.0;

            // Complex splat: extract real and imaginary parts separately.
            if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elemType)) {
                auto complexVal = denseAttr.getSplatValue<std::complex<float>>();
                double realPart = complexVal.real();
                double imagPart = complexVal.imag();
                if (shape.count == 0) {
                    result = [g constantWithRealPart:realPart
                                       imaginaryPart:imagPart
                                            dataType:dtype];
                } else {
                    result = [g constantWithRealPart:realPart
                                       imaginaryPart:imagPart
                                               shape:shape
                                            dataType:dtype];
                }
                if (!result)
                    return ProcessResult::Error("constant: handler returned null");
                SetOutputTensor(values, op, result);
                return ProcessResult{};
            }

            if (elemType.isF32()) {
                scalarValue = denseAttr.getSplatValue<float>();
            } else if (elemType.isF64()) {
                scalarValue = denseAttr.getSplatValue<double>();
            } else if (elemType.isF16()) {
                auto apVal = denseAttr.getSplatValue<llvm::APFloat>();
                scalarValue = apVal.convertToFloat();
            } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                auto apInt = denseAttr.getSplatValue<llvm::APInt>();
                // For signless integers (the default in MLIR/StableHLO), use sign-extend
                // to preserve the two's complement representation.
                // Only use zero-extend for explicitly unsigned types.
                if (intType.isUnsigned()) {
                    scalarValue = static_cast<double>(apInt.getZExtValue());
                } else {
                    // Signless or signed - use sign-extend
                    scalarValue = static_cast<double>(apInt.getSExtValue());
                }
            }

            if (shape.count == 0) {
                // True scalar
                result = [g constantWithScalar:scalarValue dataType:dtype];
            } else {
                // Splat to shape
                result = [g constantWithScalar:scalarValue shape:shape dataType:dtype];
            }
        } else {
            // Non-splat dense constant - use raw data
            auto rawData = denseAttr.getRawData();
            NSData* data = [NSData dataWithBytes:rawData.data() length:rawData.size()];
            result = [g constantWithData:data shape:shape dataType:dtype];
        }
    }

    if (!result)
        return ProcessResult::Error("constant: unsupported value type");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.constant", Handle_constant);

// Iota - create an array of indices
static ProcessResult Handle_iota(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        return ProcessResult::Error("iota: expected IotaOp");
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("iota: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    int64_t iotaDim = static_cast<int64_t>(iotaOp.getIotaDimension());

    // Create a coordinate tensor along the iota dimension
    MPSGraphTensor* result = [g coordinateAlongAxis:(NSInteger)iotaDim withShape:shape name:nil];

    // Cast to the target type if needed
    if (result.dataType != dtype) {
        result = [g castTensor:result toType:dtype name:nil];
    }

    if (!result)
        return ProcessResult::Error("iota: handler returned null");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.iota", Handle_iota);

}  // namespace jax_mps
