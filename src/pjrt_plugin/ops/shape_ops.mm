// Shape operations: broadcast, reshape, convert, slice, iota, etc.

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static MPSGraphTensor* Handle_broadcast(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g broadcastTensor:input toShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.broadcast", Handle_broadcast);

// broadcast_in_dim needs special handling for dimension mapping
static MPSGraphTensor* Handle_broadcast_in_dim(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
    if (!broadcastOp) {
        MPS_LOG_ERROR("Expected BroadcastInDimOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        MPS_LOG_ERROR("broadcast_in_dim input tensor not found\n");
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

static MPSGraphTensor* Handle_reshape(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g reshapeTensor:input withShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.reshape", Handle_reshape);

static MPSGraphTensor* Handle_transpose(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        MPS_LOG_ERROR("Expected TransposeOp\n");
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

static MPSGraphTensor* Handle_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR("Invalid dtype for convert operation\n");
        return nullptr;
    }
    return [g castTensor:input toType:dtype name:nil];
}
REGISTER_MPS_OP("stablehlo.convert", Handle_convert);

// Slice - extract a portion of a tensor (static indices)
static MPSGraphTensor* Handle_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
    if (!sliceOp) {
        MPS_LOG_ERROR("Expected SliceOp\n");
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
static MPSGraphTensor* Handle_dynamic_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
    if (!dynSliceOp) {
        MPS_LOG_ERROR("Expected DynamicSliceOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto sliceSizes = dynSliceOp.getSliceSizes();
    NSUInteger rank = sliceSizes.size();

    // Build the output shape from slice sizes
    NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
    for (int64_t s : sliceSizes) {
        [outputShape addObject:@(s)];
    }

    // Get start indices as tensors (operands 1 through N)
    // and create coordinate tensors offset by the start indices
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        // Get the start index tensor for this dimension (scalar tensor)
        MPSGraphTensor* startIdx = GetInputTensor(values, op, dim + 1);
        if (!startIdx) {
            MPS_LOG_ERROR("dynamic_slice missing start index for dimension %lu\n",
                          (unsigned long)dim);
            return nullptr;
        }

        // Create coordinate tensor for this dimension (0, 1, 2, ..., slice_size-1)
        MPSGraphTensor* coords = [g coordinateAlongAxis:(NSInteger)dim
                                              withShape:outputShape
                                                   name:nil];

        // Cast coordinates to match start index type for addition
        coords = [g castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates (broadcasts the scalar start index)
        MPSGraphTensor* adjustedCoords = [g additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    // Shape: [slice_size_0, slice_size_1, ..., rank]
    MPSGraphTensor* indices = [g stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Use gatherND to gather the slice from the input tensor
    // batchDimensions: 0 means no batch dimensions
    return [g gatherNDWithUpdatesTensor:input indicesTensor:indices batchDimensions:0 name:nil];
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", Handle_dynamic_slice);

// Iota - create an array of indices
static MPSGraphTensor* Handle_iota(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        MPS_LOG_ERROR("Expected IotaOp\n");
        return nullptr;
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR("Invalid dtype for iota operation\n");
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
static MPSGraphTensor* Handle_bitcast_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR("Invalid dtype for bitcast_convert operation\n");
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
static MPSGraphTensor* Handle_custom_call(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        MPS_LOG_ERROR("Expected CustomCallOp\n");
        return nullptr;
    }

    std::string target = customCallOp.getCallTargetName().str();

    // Sharding is a marker used by JAX for partitioning - just pass through the input
    if (target == "Sharding") {
        return GetInputTensor(values, op, 0);
    }

    // cu_threefry2x32 - Threefry RNG core operation
    if (target == "cu_threefry2x32") {
        MPS_LOG_ERROR("Custom call 'cu_threefry2x32' is not yet implemented\n");
        return nullptr;
    }

    // mhlo.erf - Error function
    if (target == "mhlo.erf") {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input)
            return nullptr;
        return [g erfWithTensor:input name:nil];
    }

    // mhlo.asin - Arcsine
    if (target == "mhlo.asin") {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input)
            return nullptr;
        return [g asinWithTensor:input name:nil];
    }

    // mhlo.acos - Arccosine
    if (target == "mhlo.acos") {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input)
            return nullptr;
        return [g acosWithTensor:input name:nil];
    }

    // Unknown custom call
    MPS_LOG_ERROR("Unknown custom_call target: %s\n", target.c_str());
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.custom_call", Handle_custom_call);

// Pad - add padding around tensor
static MPSGraphTensor* Handle_pad(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
    if (!padOp) {
        MPS_LOG_ERROR("Expected PadOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(values, op, 1);
    if (!input || !paddingValue)
        return nullptr;

    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();

    // Check if interior padding is all zeros (simple edge padding case)
    bool hasInteriorPadding = false;
    for (int64_t p : interiorPadding) {
        if (p != 0) {
            hasInteriorPadding = true;
            break;
        }
    }

    if (hasInteriorPadding) {
        MPS_LOG_ERROR("Interior padding not yet supported\n");
        return nullptr;
    }

    // Get output shape and create a tensor filled with padding value
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    MPSGraphTensor* padded = [g broadcastTensor:paddingValue toShape:outputShape name:nil];

    // Calculate starts and ends for sliceUpdate (where to place the input)
    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    NSArray<NSNumber*>* inputShape = input.shape;
    for (NSUInteger i = 0; i < edgePaddingLow.size(); i++) {
        int64_t start = edgePaddingLow[i];
        int64_t inputDim = [inputShape[i] longLongValue];
        [starts addObject:@(start)];
        [ends addObject:@(start + inputDim)];
        [strides addObject:@1];
    }

    // Use sliceUpdateDataTensor to insert input into the padded tensor
    return [g sliceUpdateDataTensor:padded
                       updateTensor:input
                             starts:starts
                               ends:ends
                            strides:strides
                               name:nil];
}
REGISTER_MPS_OP("stablehlo.pad", Handle_pad);

// Dynamic update slice - update a portion of a tensor with new values
static MPSGraphTensor* Handle_dynamic_update_slice(MPSGraph* g, mlir::Operation* op,
                                                   ValueMap& values) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(op);
    if (!updateSliceOp) {
        MPS_LOG_ERROR("Expected DynamicUpdateSliceOp\n");
        return nullptr;
    }

    MPSGraphTensor* operand = GetInputTensor(values, op, 0);
    MPSGraphTensor* update = GetInputTensor(values, op, 1);
    if (!operand || !update)
        return nullptr;

    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;

    // Get start indices (operands 2 through N)
    NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        MPSGraphTensor* startIdx = GetInputTensor(values, op, i + 2);
        if (!startIdx) {
            MPS_LOG_ERROR("dynamic_update_slice missing start index for dimension %lu\n",
                          (unsigned long)i);
            return nullptr;
        }
        [startIndices addObject:startIdx];
    }

    // Build starts array by reading the scalar start indices
    // For sliceUpdateDataTensor, we need static starts/ends/strides
    // But the start indices are dynamic tensors, so we need to use scatter instead

    // Create coordinate tensors for the update region
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        MPSGraphTensor* startIdx = startIndices[dim];

        // Create coordinate tensor for this dimension (0, 1, 2, ..., update_size-1)
        MPSGraphTensor* coords = [g coordinateAlongAxis:(NSInteger)dim
                                              withShape:updateShape
                                                   name:nil];

        // Cast coordinates to match start index type
        coords = [g castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates
        MPSGraphTensor* adjustedCoords = [g additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    MPSGraphTensor* indices = [g stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Cast indices to int32 if needed
    indices = EnsureInt32(g, indices);

    // Use scatterND to update the operand at the specified indices
    return [g scatterNDWithDataTensor:operand
                        updatesTensor:update
                        indicesTensor:indices
                      batchDimensions:0
                                 mode:MPSGraphScatterModeSet
                                 name:nil];
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", Handle_dynamic_update_slice);

// Gather - generalized indexing operation
// Handles embedding lookups and other gather patterns
static MPSGraphTensor* Handle_gather(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op);
    if (!gatherOp) {
        MPS_LOG_ERROR("Expected GatherOp\n");
        return nullptr;
    }

    MPSGraphTensor* operand = GetInputTensor(values, op, 0);
    MPSGraphTensor* startIndices = GetInputTensor(values, op, 1);
    if (!operand || !startIndices)
        return nullptr;

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto sliceSizes = gatherOp.getSliceSizes();

    // Handle common embedding lookup pattern:
    // operand: [num_embeddings, embedding_dim]
    // indices: [batch..., 1] where the last dim is the index vector
    // offset_dims: [last_dim] - the embedding dimension
    // collapsed_slice_dims: [0] - the looked-up dimension
    // start_index_map: [0] - indices point into dim 0

    NSArray<NSNumber*>* indicesShape = startIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Check if index_vector_dim is the last dimension and has size 1
    // This is the common embedding pattern
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
        collapsedSliceDims.size() == 1 && collapsedSliceDims[0] == startIndexMap[0]) {
        int64_t gatherAxis = startIndexMap[0];

        // Squeeze the index vector dimension from indices
        // [batch..., 1] -> [batch...]
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        MPSGraphTensor* squeezedIndices = [g reshapeTensor:startIndices
                                                 withShape:squeezedShape
                                                      name:nil];

        // Cast indices to int32 if needed (MPS gather requires int32)
        squeezedIndices = EnsureInt32(g, squeezedIndices);

        // Use gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:
        // This gathers slices from operand along the specified axis using indices
        // Result shape: indices.shape + operand.shape[axis+1:]
        // For embedding [100, 5] with indices [3]: result is [3, 5]
        MPSGraphTensor* result = [g gatherWithUpdatesTensor:operand
                                              indicesTensor:squeezedIndices
                                                       axis:(NSUInteger)gatherAxis
                                            batchDimensions:0
                                                       name:nil];

        return result;
    }

    // For now, log unsupported patterns
    MPS_LOG_ERROR("Unsupported gather pattern - offset_dims size: %lu, collapsed_slice_dims "
                  "size: %lu, start_index_map size: %lu, index_vector_dim: %lld\n",
                  (unsigned long)offsetDims.size(), (unsigned long)collapsedSliceDims.size(),
                  (unsigned long)startIndexMap.size(), indexVectorDim);
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.gather", Handle_gather);

// Scatter - update tensor at specified indices
// This handles the common pattern used by gather gradients
static MPSGraphTensor* Handle_scatter(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
    if (!scatterOp) {
        MPS_LOG_ERROR("Expected ScatterOp\n");
        return nullptr;
    }

    // Get inputs (may be variadic, but we handle single input case)
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(values, op, 1);
    MPSGraphTensor* updates = GetInputTensor(values, op, 2);
    if (!input || !scatterIndices || !updates)
        return nullptr;

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Handle common embedding gradient pattern (reverse of gather):
    // input: [num_embeddings, embedding_dim] - zeros initially
    // indices: [batch..., 1] where last dim is index vector
    // updates: [batch..., embedding_dim] - gradients to scatter
    // Result: accumulate updates into input at specified indices

    // Check for the common pattern where:
    // - index_vector_dim is the last dimension of indices
    // - indices has size 1 in that dimension
    // - we're scattering along a single dimension
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        insertedWindowDims.size() == 1 && insertedWindowDims[0] == scatterDimsToOperandDims[0]) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        // Squeeze the index vector dimension from indices
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        // If squeezing produces a scalar, keep as [1] so MPS has a valid rank for the axis
        if (squeezedShape.count == 0)
            [squeezedShape addObject:@1];

        MPSGraphTensor* squeezedIndices = [g reshapeTensor:scatterIndices
                                                 withShape:squeezedShape
                                                      name:nil];

        // Cast indices to int32 if needed
        squeezedIndices = EnsureInt32(g, squeezedIndices);

        // Determine the scatter mode based on the update computation.
        // Default to Set (plain assignment); arithmetic ops override below.
        MPSGraphScatterMode mode = MPSGraphScatterModeSet;

        auto& updateRegion = scatterOp.getUpdateComputation();
        if (!updateRegion.empty()) {
            auto& block = updateRegion.front();
            for (auto& innerOp : block) {
                if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                    mode = MPSGraphScatterModeAdd;
                    break;
                } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                    mode = MPSGraphScatterModeSub;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                    mode = MPSGraphScatterModeMul;
                    break;
                } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                    mode = MPSGraphScatterModeDiv;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                    mode = MPSGraphScatterModeMax;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                    mode = MPSGraphScatterModeMin;
                    break;
                }
            }
        }

        // Ensure updates is at least rank 1 (MPS doesn't support scalar updates)
        if (updates.shape.count == 0)
            updates = [g reshapeTensor:updates withShape:@[@1] name:nil];

        // Use scatterWithDataTensor to scatter updates into input
        return [g scatterWithDataTensor:input
                          updatesTensor:updates
                          indicesTensor:squeezedIndices
                                   axis:(NSUInteger)scatterAxis
                                   mode:mode
                                   name:nil];
    }

    MPS_LOG_ERROR("Unsupported scatter pattern - update_window_dims size: %lu, "
                  "inserted_window_dims size: %lu, scatter_dims_to_operand_dims size: %lu, "
                  "index_vector_dim: %lld\n",
                  (unsigned long)updateWindowDims.size(), (unsigned long)insertedWindowDims.size(),
                  (unsigned long)scatterDimsToOperandDims.size(), indexVectorDim);
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.scatter", Handle_scatter);

// Reverse - reverse elements along specified dimensions
static MPSGraphTensor* Handle_reverse(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(op);
    if (!reverseOp) {
        MPS_LOG_ERROR("Expected ReverseOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    return [g reverseTensor:input axes:axes name:nil];
}
REGISTER_MPS_OP("stablehlo.reverse", Handle_reverse);

}  // namespace jax_mps
