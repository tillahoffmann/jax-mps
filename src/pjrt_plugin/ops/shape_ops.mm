// Shape operations: broadcast, reshape, convert, slice, iota, etc.

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static MPSGraphTensor* Handle_broadcast(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                        NSArray<NSNumber*>* shape) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g broadcastTensor:input toShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.broadcast", Handle_broadcast);

// broadcast_in_dim needs special handling for dimension mapping
static MPSGraphTensor* Handle_broadcast_in_dim(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                               NSArray<NSNumber*>*) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
    if (!broadcastOp) {
        NSLog(@"ERROR: Expected BroadcastInDimOp");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        NSLog(@"ERROR: broadcast_in_dim input tensor not found");
        return nullptr;
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;

    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    // If broadcast_dims is empty, just broadcast directly
    if (broadcastDims.empty()) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // If ranks already match, just broadcast
    if (inputRank == outputRank) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // Build intermediate shape: start with all 1s, then fill in from broadcast_dims
    NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
    for (NSUInteger i = 0; i < outputRank; i++) {
        [intermediateShape addObject:@1];
    }

    // Map input dimensions to output dimensions according to broadcast_dims
    for (size_t i = 0; i < broadcastDims.size() && i < inputRank; i++) {
        int64_t outDim = broadcastDims[i];
        if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
            intermediateShape[outDim] = inputShape[i];
        }
    }

    // Reshape input to intermediate shape (same rank as output)
    MPSGraphTensor* reshaped = [g reshapeTensor:input withShape:intermediateShape name:nil];

    // Now broadcast to final output shape
    return [g broadcastTensor:reshaped toShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", Handle_broadcast_in_dim);

static MPSGraphTensor* Handle_reshape(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                      NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g reshapeTensor:input withShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.reshape", Handle_reshape);

static MPSGraphTensor* Handle_transpose(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                        NSArray<NSNumber*>*) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        NSLog(@"ERROR: Expected TransposeOp");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
    }

    return [g transposeTensor:input permutation:perm name:nil];
}
REGISTER_MPS_OP("stablehlo.transpose", Handle_transpose);

static MPSGraphTensor* Handle_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                      NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype for convert operation");
        return nullptr;
    }
    return [g castTensor:input toType:dtype name:nil];
}
REGISTER_MPS_OP("stablehlo.convert", Handle_convert);

// Slice - extract a portion of a tensor (static indices)
static MPSGraphTensor* Handle_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                    NSArray<NSNumber*>*) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
    if (!sliceOp) {
        NSLog(@"ERROR: Expected SliceOp");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceOp.getStartIndices()) {
        [starts addObject:@(s)];
    }
    for (int64_t l : sliceOp.getLimitIndices()) {
        [ends addObject:@(l)];
    }
    for (int64_t s : sliceOp.getStrides()) {
        [strides addObject:@(s)];
    }

    return [g sliceTensor:input starts:starts ends:ends strides:strides name:nil];
}
REGISTER_MPS_OP("stablehlo.slice", Handle_slice);

// Dynamic slice - extract a portion using runtime indices
static MPSGraphTensor* Handle_dynamic_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                            NSArray<NSNumber*>*) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
    if (!dynSliceOp) {
        NSLog(@"ERROR: Expected DynamicSliceOp");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto sliceSizes = dynSliceOp.getSliceSizes();

    // Build the sizes, starts, and strides arrays
    NSMutableArray<NSNumber*>* sizes = [NSMutableArray array];
    NSMutableArray<NSNumber*>* zeroStarts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceSizes) {
        [sizes addObject:@(s)];
        [zeroStarts addObject:@0];
        [strides addObject:@1];
    }

    // Note: This is a TEMPORARY workaround that only works correctly when start indices are 0
    // A proper implementation would need to use gather operations
    return [g sliceTensor:input starts:zeroStarts ends:sizes strides:strides name:nil];
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", Handle_dynamic_slice);

// Iota - create an array of indices
static MPSGraphTensor* Handle_iota(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                   NSArray<NSNumber*>*) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        NSLog(@"ERROR: Expected IotaOp");
        return nullptr;
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype for iota operation");
        return nullptr;
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    int64_t iotaDim = iotaOp.getIotaDimension();

    // Create a coordinate tensor along the iota dimension
    MPSGraphTensor* result = [g coordinateAlongAxis:(NSInteger)iotaDim withShape:shape name:nil];

    // Cast to the target type if needed
    if (result.dataType != dtype) {
        result = [g castTensor:result toType:dtype name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.iota", Handle_iota);

// Bitcast convert - reinterpret bits as a different type
static MPSGraphTensor* Handle_bitcast_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                              NSArray<NSNumber*>*) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        NSLog(@"ERROR: Invalid dtype for bitcast_convert operation");
        return nullptr;
    }

    // MPS reinterpretCastTensor doesn't support rank-0 (scalar) tensors.
    // Work around by reshaping to rank-1, casting, then reshaping back.
    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {
        // Reshape scalar to [1]
        input = [g reshapeTensor:input withShape:@[@1] name:nil];
    }

    // Use reinterpretCast which preserves bit patterns
    MPSGraphTensor* result = [g reinterpretCastTensor:input toType:dtype name:nil];

    if (isScalar) {
        // Reshape back to scalar
        result = [g reshapeTensor:result withShape:@[] name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", Handle_bitcast_convert);

// Custom call - handle specific JAX custom operations
static MPSGraphTensor* Handle_custom_call(MPSGraph* g, mlir::Operation* op, ValueMap& values,
                                          NSArray<NSNumber*>*) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        NSLog(@"ERROR: Expected CustomCallOp");
        return nullptr;
    }

    std::string target = customCallOp.getCallTargetName().str();

    // Sharding is a marker used by JAX for partitioning - just pass through the input
    if (target == "Sharding") {
        return GetInputTensor(values, op, 0);
    }

    // cu_threefry2x32 - Threefry RNG core operation
    if (target == "cu_threefry2x32") {
        NSLog(@"ERROR: Custom call 'cu_threefry2x32' is not yet implemented");
        return nullptr;
    }

    // mhlo.erf - Error function
    if (target == "mhlo.erf") {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input)
            return nullptr;
        return [g erfWithTensor:input name:nil];
    }

    // Unknown custom call
    NSLog(@"ERROR: Unknown custom_call target: %s", target.c_str());
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.custom_call", Handle_custom_call);

}  // namespace jax_mps
