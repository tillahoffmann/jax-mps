// Unary operations: tanh, exp, log, negate, abs
// Also includes constant creation, compare, and select

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Helper to get output shape from operation's result type
static NSArray<NSNumber*>* GetOutputShape(mlir::Operation* op, unsigned resultIndex = 0) {
    if (resultIndex >= op->getNumResults()) {
        return nil;
    }
    auto resultType = op->getResult(resultIndex).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType) {
        return nil;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (int64_t dim : tensorType.getShape()) {
        [shape addObject:@(dim)];
    }
    return shape;
}

// Unary ops using macros
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, tanh);
REGISTER_MLIR_UNARY_OP("stablehlo.exponential", exponent, exp);
REGISTER_MLIR_UNARY_OP("stablehlo.log", logarithm, log);
REGISTER_MLIR_UNARY_OP("stablehlo.negate", negative, negate);
REGISTER_MLIR_UNARY_OP("stablehlo.abs", absolute, abs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
static MPSGraphTensor* Handle_log_plus_one(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                           NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [g additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    return [g logarithmWithTensor:onePlusX name:nil];
}
REGISTER_MPS_OP("stablehlo.log_plus_one", Handle_log_plus_one);

// Compare operation
static MPSGraphTensor* Handle_compare(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                      NSArray<NSNumber*>*) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        NSLog(@"ERROR: Expected CompareOp");
        return nullptr;
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return nullptr;

    auto direction = compareOp.getComparisonDirection();
    using Dir = mlir::stablehlo::ComparisonDirection;

    switch (direction) {
        case Dir::LT:
            return [g lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::LE:
            return [g lessThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::GT:
            return [g greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::GE:
            return [g greaterThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::EQ:
            return [g equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        case Dir::NE:
            return [g notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        default:
            NSLog(@"ERROR: Unknown compare direction");
            return nullptr;
    }
}
REGISTER_MPS_OP("stablehlo.compare", Handle_compare);

// Select operation (conditional selection: pred ? true_val : false_val)
static MPSGraphTensor* Handle_select(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                     NSArray<NSNumber*>*) {
    MPSGraphTensor* pred = GetInputTensor(values, op, 0);
    MPSGraphTensor* onTrue = GetInputTensor(values, op, 1);
    MPSGraphTensor* onFalse = GetInputTensor(values, op, 2);
    if (!pred || !onTrue || !onFalse)
        return nullptr;

    return [g selectWithPredicateTensor:pred
                    truePredicateTensor:onTrue
                   falsePredicateTensor:onFalse
                                   name:nil];
}
REGISTER_MPS_OP("stablehlo.select", Handle_select);

// Clamp operation: clamp(min, x, max) = min(max(min, x), max)
static MPSGraphTensor* Handle_clamp(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                    NSArray<NSNumber*>*) {
    MPSGraphTensor* minVal = GetInputTensor(values, op, 0);
    MPSGraphTensor* operand = GetInputTensor(values, op, 1);
    MPSGraphTensor* maxVal = GetInputTensor(values, op, 2);
    if (!minVal || !operand || !maxVal)
        return nullptr;

    // clamp = min(max(minVal, operand), maxVal)
    MPSGraphTensor* clamped = [g maximumWithPrimaryTensor:minVal secondaryTensor:operand name:nil];
    return [g minimumWithPrimaryTensor:clamped secondaryTensor:maxVal name:nil];
}
REGISTER_MPS_OP("stablehlo.clamp", Handle_clamp);

// Constant creation - creates a constant tensor from MLIR constant op
static MPSGraphTensor* Handle_constant(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                       NSArray<NSNumber*>*) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
        NSLog(@"ERROR: Expected ConstantOp");
        return nullptr;
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype for constant operation");
        return nullptr;
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    auto value = constantOp.getValue();

    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
        // Check if it's a splat (single value broadcast to all elements)
        if (denseAttr.isSplat()) {
            auto elemType = denseAttr.getElementType();
            double scalarValue = 0.0;

            if (elemType.isF32()) {
                scalarValue = denseAttr.getSplatValue<float>();
            } else if (elemType.isF64()) {
                scalarValue = denseAttr.getSplatValue<double>();
            } else if (elemType.isF16()) {
                auto apVal = denseAttr.getSplatValue<llvm::APFloat>();
                scalarValue = apVal.convertToFloat();
            } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                auto apInt = denseAttr.getSplatValue<llvm::APInt>();
                scalarValue = static_cast<double>(apInt.getZExtValue());
            }

            if (shape.count == 0) {
                // True scalar
                return [g constantWithScalar:scalarValue dataType:dtype];
            } else {
                // Splat to shape
                return [g constantWithScalar:scalarValue shape:shape dataType:dtype];
            }
        } else {
            // Non-splat dense constant - use raw data
            auto rawData = denseAttr.getRawData();
            NSData* data = [NSData dataWithBytes:rawData.data() length:rawData.size()];
            return [g constantWithData:data shape:shape dataType:dtype];
        }
    }

    NSLog(@"ERROR: Constant operation has unsupported value type");
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.constant", Handle_constant);

}  // namespace jax_mps
