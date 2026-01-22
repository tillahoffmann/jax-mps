// Unary operations: tanh, exp, log, negate, abs
// Also includes constant creation

#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_UNARY_OP(tanh, tanh);
REGISTER_UNARY_OP(exp, exponent);
REGISTER_UNARY_OP(log, logarithm);
REGISTER_UNARY_OP(negate, negative);
REGISTER_UNARY_OP(abs, absolute);
REGISTER_UNARY_OP(sqrt, squareRoot);
REGISTER_UNARY_OP(erf, erf);

// log_plus_one: log(1+x) - matches PyTorch MPS implementation
// Note: No native log1p in MPSGraph, so we compute log(1 + x)
static MPSGraphTensor* Handle_log_plus_one(MPSGraph* g, TensorDict t, const HloOp& op,
                                           NSArray<NSNumber*>* shape) {
    MPSGraphTensor* input = t[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
    if (!input)
        return nullptr;

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* onePlusX = [g additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    return [g logarithmWithTensor:onePlusX name:nil];
}
REGISTER_OP(log_plus_one, Handle_log_plus_one);

// Compare operation
static MPSGraphTensor* Handle_compare(MPSGraph* g, TensorDict t, const HloOp& op,
                                      NSArray<NSNumber*>* shape) {
    MPSGraphTensor* lhs = t[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
    MPSGraphTensor* rhs = t[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
    if (!lhs || !rhs)
        return nullptr;

    const std::string& dir = op.compare_direction;
    if (dir == "LT") {
        return [g lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    } else if (dir == "LE") {
        return [g lessThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    } else if (dir == "GT") {
        return [g greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    } else if (dir == "GE") {
        return [g greaterThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    } else if (dir == "EQ") {
        return [g equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    } else if (dir == "NE") {
        return [g notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    }
    NSLog(@"ERROR: Unknown compare direction: %s", dir.c_str());
    return nullptr;
}
REGISTER_OP(compare, Handle_compare);

// Select operation (conditional selection: pred ? true_val : false_val)
static MPSGraphTensor* Handle_select(MPSGraph* g, TensorDict t, const HloOp& op,
                                     NSArray<NSNumber*>* shape) {
    // select(pred, on_true, on_false)
    MPSGraphTensor* pred = t[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
    MPSGraphTensor* onTrue = t[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
    MPSGraphTensor* onFalse = t[[NSString stringWithUTF8String:op.inputs[2].c_str()]];
    if (!pred || !onTrue || !onFalse)
        return nullptr;

    return [g selectWithPredicateTensor:pred
                    truePredicateTensor:onTrue
                   falsePredicateTensor:onFalse
                                   name:nil];
}
REGISTER_OP(select, Handle_select);

// Constant creation - creates a constant tensor from parsed StableHLO data
static MPSGraphTensor* Handle_constant(MPSGraph* g, TensorDict t, const HloOp& op,
                                       NSArray<NSNumber*>* shape) {
    MPSDataType dtype = PjrtDtypeToMps(op.dtype);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype %d for constant operation", op.dtype);
        return nullptr;
    }

    // Handle scalar/splat constants (most common case, e.g., 0.0 for relu)
    if (op.is_scalar_constant) {
        double scalar_value;
        if (op.uses_raw_data) {
            // Integer scalar - use raw value
            scalar_value = static_cast<double>(op.constant_scalar_raw);
        } else {
            scalar_value = static_cast<double>(op.constant_scalar);
        }

        if (shape.count == 0) {
            // True scalar
            return [g constantWithScalar:scalar_value dataType:dtype];
        } else {
            // Splat to shape
            return [g constantWithScalar:scalar_value shape:shape dataType:dtype];
        }
    }

    // Handle dense array constants with raw data
    if (op.uses_raw_data && !op.constant_raw.empty()) {
        NSData* data = [NSData dataWithBytes:op.constant_raw.data() length:op.constant_raw.size()];
        return [g constantWithData:data shape:shape dataType:dtype];
    }

    // Handle dense array constants (float data)
    if (!op.constant_data.empty()) {
        // Create NSData from the constant values
        NSData* data = [NSData dataWithBytes:op.constant_data.data()
                                      length:op.constant_data.size() * sizeof(float)];
        return [g constantWithData:data shape:shape dataType:dtype];
    }

    // No constant data available - this shouldn't happen if parser worked correctly
    NSLog(@"ERROR: Constant operation has no data (is_scalar=%d, uses_raw=%d, data_size=%zu, "
          @"raw_size=%zu)",
          op.is_scalar_constant, op.uses_raw_data, op.constant_data.size(), op.constant_raw.size());
    return nullptr;
}
REGISTER_OP(constant, Handle_constant);

}  // namespace jax_mps
