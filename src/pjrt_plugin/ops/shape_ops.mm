// Shape operations: broadcast, reshape, convert, slice, concatenate,
// custom_call, etc.

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static ProcessResult HandleBroadcast(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("broadcast: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    return Result(ctx, result, "broadcast");
}
REGISTER_MPS_OP("stablehlo.broadcast", HandleBroadcast);

// broadcast_in_dim needs special handling for dimension mapping
static ProcessResult HandleBroadcastInDim(HandlerContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(ctx.op);
    if (!broadcastOp) {
        return ProcessResult::Error("broadcast_in_dim: expected BroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("broadcast_in_dim: input tensor not found");
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    MPSGraphTensor* result = nil;

    // If broadcast_dims is empty or ranks match, just broadcast directly
    if (broadcastDims.empty() || inputRank == outputRank) {
        result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    } else {
        // Build intermediate shape: start with all 1s, then fill in from broadcast_dims
        NSMutableArray<NSNumber*>* intermediateShape =
            [NSMutableArray arrayWithCapacity:outputRank];
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
        MPSGraphTensor* reshaped = [ctx.graph reshapeTensor:input
                                                  withShape:intermediateShape
                                                       name:nil];

        // Now broadcast to final output shape
        result = [ctx.graph broadcastTensor:reshaped toShape:outputShape name:nil];
    }

    return Result(ctx, result, "broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", HandleBroadcastInDim);

static ProcessResult HandleReshape(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reshape: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "reshape");
}
REGISTER_MPS_OP("stablehlo.reshape", HandleReshape);

static ProcessResult HandleTranspose(HandlerContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(ctx.op);
    if (!transposeOp) {
        return ProcessResult::Error("transpose: expected TransposeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("transpose: missing input tensor");

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
    }

    MPSGraphTensor* result = [ctx.graph transposeTensor:input permutation:perm name:nil];
    return Result(ctx, result, "transpose");
}
REGISTER_MPS_OP("stablehlo.transpose", HandleTranspose);

static ProcessResult HandleConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("convert: invalid dtype for convert operation");
    }
    MPSGraphTensor* result = [ctx.graph castTensor:input toType:dtype name:nil];
    return Result(ctx, result, "convert");
}
REGISTER_MPS_OP("stablehlo.convert", HandleConvert);

// Slice - extract a portion of a tensor (static indices)
static ProcessResult HandleSlice(HandlerContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(ctx.op);
    if (!sliceOp) {
        return ProcessResult::Error("slice: expected SliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("slice: missing input tensor");

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

    MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                             starts:starts
                                               ends:ends
                                            strides:strides
                                               name:nil];
    return Result(ctx, result, "slice");
}
REGISTER_MPS_OP("stablehlo.slice", HandleSlice);

// Dynamic slice - extract a portion using runtime indices
static ProcessResult HandleDynamicSlice(HandlerContext& ctx) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(ctx.op);
    if (!dynSliceOp) {
        return ProcessResult::Error("dynamic_slice: expected DynamicSliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("dynamic_slice: missing input tensor");

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
        MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_slice: missing start index for dimension");
        }

        // Create coordinate tensor for this dimension (0, 1, 2, ..., slice_size-1)
        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:outputShape
                                                           name:nil];

        // Cast coordinates to match start index type for addition
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates (broadcasts the scalar start index)
        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    // Shape: [slice_size_0, slice_size_1, ..., rank]
    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Use gatherND to gather the slice from the input tensor
    // batchDimensions: 0 means no batch dimensions
    MPSGraphTensor* result = [ctx.graph gatherNDWithUpdatesTensor:input
                                                    indicesTensor:indices
                                                  batchDimensions:0
                                                             name:nil];
    return Result(ctx, result, "dynamic_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", HandleDynamicSlice);

// Bitcast convert - reinterpret bits as a different type
static ProcessResult HandleBitcastConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("bitcast_convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("bitcast_convert: invalid dtype");
    }

    // MPS reinterpretCastTensor doesn't support rank-0 (scalar) tensors.
    // Work around by reshaping to rank-1, casting, then reshaping back.
    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {
        // Reshape scalar to [1]
        input = [ctx.graph reshapeTensor:input withShape:@[@1] name:nil];
    }

    // Use reinterpretCast which preserves bit patterns
    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:input toType:dtype name:nil];

    if (isScalar) {
        // Reshape back to scalar
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(ctx, result, "bitcast_convert");
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", HandleBitcastConvert);

// Concatenate - joins tensors along a dimension
static ProcessResult HandleConcatenate(HandlerContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(ctx.op);
    if (!concatOp) {
        return ProcessResult::Error("concatenate: expected ConcatenateOp");
    }

    // Gather all input tensors
    NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray array];
    for (mlir::Value operand : ctx.op->getOperands()) {
        MPSGraphTensor* tensor = GetTensor(ctx.values, operand);
        if (tensor) {
            [input_tensors addObject:tensor];
        }
    }

    if (input_tensors.count == 0) {
        return ProcessResult::Error("concatenate: no valid inputs");
    }

    // Get the concatenate dimension from the op
    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    MPSGraphTensor* result = [ctx.graph concatTensors:input_tensors dimension:dimension name:nil];
    return Result(ctx, result, "concatenate");
}
REGISTER_MPS_OP("stablehlo.concatenate", HandleConcatenate);

// Sharding is a marker used by JAX for partitioning - just pass through the input
static ProcessResult HandleSharding(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("sharding: missing input tensor");
    SetOutputTensor(ctx.values, ctx.op, input);
    return ProcessResult{};
}
REGISTER_CUSTOM_CALL("Sharding", HandleSharding, sharding);

// Pad - add padding around tensor
static ProcessResult HandlePad(HandlerContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(ctx.op);
    if (!padOp) {
        return ProcessResult::Error("pad: expected PadOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(ctx, 1);
    if (!input || !paddingValue)
        return ProcessResult::Error("pad: missing input tensor");

    auto edgePaddingLow = padOp.getEdgePaddingLow();
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
        return ProcessResult::Error("pad: interior padding not yet supported");
    }

    // Get output shape and create a tensor filled with padding value
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* padded = [ctx.graph broadcastTensor:paddingValue toShape:outputShape name:nil];

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
    MPSGraphTensor* result = [ctx.graph sliceUpdateDataTensor:padded
                                                 updateTensor:input
                                                       starts:starts
                                                         ends:ends
                                                      strides:strides
                                                         name:nil];
    return Result(ctx, result, "pad");
}
REGISTER_MPS_OP("stablehlo.pad", HandlePad);

// Dynamic update slice - update a portion of a tensor with new values
static ProcessResult HandleDynamicUpdateSlice(HandlerContext& ctx) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(ctx.op);
    if (!updateSliceOp) {
        return ProcessResult::Error("dynamic_update_slice: expected DynamicUpdateSliceOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* update = GetInputTensor(ctx, 1);
    if (!operand || !update)
        return ProcessResult::Error("dynamic_update_slice: missing input tensor");

    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;

    // Get start indices (operands 2 through N)
    NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        MPSGraphTensor* startIdx = GetInputTensor(ctx, i + 2);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_update_slice: missing start index");
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
        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:updateShape
                                                           name:nil];

        // Cast coordinates to match start index type
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates
        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Cast indices to int32 if needed
    indices = EnsureInt32(ctx.graph, indices);

    // Use scatterND to update the operand at the specified indices
    MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:operand
                                                  updatesTensor:update
                                                  indicesTensor:indices
                                                batchDimensions:0
                                                           mode:MPSGraphScatterModeSet
                                                           name:nil];
    return Result(ctx, result, "dynamic_update_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice);

// Gather - generalized indexing operation
// Handles embedding lookups and other gather patterns
static ProcessResult HandleGather(HandlerContext& ctx) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(ctx.op);
    if (!gatherOp) {
        return ProcessResult::Error("gather: expected GatherOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* startIndices = GetInputTensor(ctx, 1);
    if (!operand || !startIndices)
        return ProcessResult::Error("gather: missing input tensor");

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

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

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:startIndices
                                                         withShape:squeezedShape
                                                              name:nil];

        // Cast indices to int32 if needed (MPS gather requires int32)
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        // Use gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:
        // This gathers slices from operand along the specified axis using indices
        // Result shape: indices.shape + operand.shape[axis+1:]
        // For embedding [100, 5] with indices [3]: result is [3, 5]
        MPSGraphTensor* result = [ctx.graph gatherWithUpdatesTensor:operand
                                                      indicesTensor:squeezedIndices
                                                               axis:(NSUInteger)gatherAxis
                                                    batchDimensions:0
                                                               name:nil];

        return Result(ctx, result, "gather");
    }

    // Handle full-index gather pattern (e.g., x[0, 0, 0] on a rank-3 tensor):
    // - indices is a 1D vector of length input_rank
    // - index_vector_dim is 0, meaning the entire indices tensor is one index vector
    // - start_index_map covers all operand dimensions in order [0, 1, ..., rank-1]
    // - offset_dims is empty (no trailing slice dimensions)
    // - collapsed_slice_dims is either empty (slice sizes are all 1) or fully collapsed
    //   ([0, 1, ..., rank-1]); both represent point gathers yielding a scalar
    if (indexVectorDim == 0 && indicesRank == 1 && startIndexMap.size() == operand.shape.count &&
        offsetDims.empty() && [indicesShape[0] integerValue] == (NSInteger)operand.shape.count) {
        bool hasFullCollapsedSliceDims = collapsedSliceDims.size() == operand.shape.count;
        if (hasFullCollapsedSliceDims) {
            for (NSUInteger dim = 0; dim < operand.shape.count; ++dim) {
                if (collapsedSliceDims[dim] != (int64_t)dim) {
                    hasFullCollapsedSliceDims = false;
                    break;
                }
            }
        }

        bool fullRange = true;
        for (NSUInteger dim = 0; dim < operand.shape.count; ++dim) {
            if (startIndexMap[dim] != (int64_t)dim) {
                fullRange = false;
                break;
            }
        }

        if (!fullRange || !(collapsedSliceDims.empty() || hasFullCollapsedSliceDims)) {
            return ProcessResult::Error("gather: unsupported full-index gather pattern");
        }

        // MPS gatherND expects indices as [N, rank]. Reshape [rank] -> [1, rank].
        MPSGraphTensor* ndIndices = [ctx.graph reshapeTensor:startIndices
                                                   withShape:@[@1, @(operand.shape.count)]
                                                        name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        MPSGraphTensor* gathered = [ctx.graph gatherNDWithUpdatesTensor:operand
                                                          indicesTensor:ndIndices
                                                        batchDimensions:0
                                                                   name:nil];

        // Result is [1] for a scalar point gather; reshape to scalar.
        if (gathered.shape.count == 1 && [gathered.shape[0] integerValue] == 1) {
            gathered = [ctx.graph reshapeTensor:gathered withShape:@[] name:nil];
        }
        return Result(ctx, gathered, "gather");
    }

    // For now, log unsupported patterns
    return ProcessResult::Error("gather: unsupported gather pattern");
}
REGISTER_MPS_OP("stablehlo.gather", HandleGather);

// Helper to determine scatter mode from the update computation region
static MPSGraphScatterMode GetScatterMode(mlir::stablehlo::ScatterOp scatterOp) {
    MPSGraphScatterMode mode = MPSGraphScatterModeSet;
    auto& updateRegion = scatterOp.getUpdateComputation();
    if (!updateRegion.empty()) {
        auto& block = updateRegion.front();
        for (auto& innerOp : block) {
            if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                return MPSGraphScatterModeAdd;
            } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                return MPSGraphScatterModeSub;
            } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                return MPSGraphScatterModeMul;
            } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                return MPSGraphScatterModeDiv;
            } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                return MPSGraphScatterModeMax;
            } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                return MPSGraphScatterModeMin;
            }
        }
    }
    return mode;
}

// Scatter - update tensor at specified indices
// This handles the common pattern used by gather gradients
static ProcessResult HandleScatter(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp) {
        return ProcessResult::Error("scatter: expected ScatterOp");
    }

    // Get inputs (may be variadic, but we handle single input case)
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(ctx, 1);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    if (!input || !scatterIndices || !updates)
        return ProcessResult::Error("scatter: missing input tensor");

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Handle batched scatter pattern used by sort gradients:
    // Pattern: scatter with batching dimensions where each batch element scatters independently
    // Example: input [5,7], indices [5,7,1], updates [5,7]
    //   - input_batching_dims = [0], scatter_indices_batching_dims = [0]
    //   - For each batch i, scatter updates[i,:] into input[i,:] at indices[i,:,0]
    if (!inputBatchingDims.empty() && !scatterIndicesBatchingDims.empty() &&
        inputBatchingDims.size() == scatterIndicesBatchingDims.size() &&
        scatterDimsToOperandDims.size() == 1 && insertedWindowDims.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        // Squeeze the index vector dimension from indices: [batch..., N, 1] -> [batch..., N]
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        // Use scatterAlongAxis which handles batched scatter correctly:
        // For each position in updates, it scatters to the position given by indices at that
        // position
        MPSGraphTensor* result = [ctx.graph scatterAlongAxis:static_cast<NSInteger>(scatterAxis)
                                              withDataTensor:input
                                               updatesTensor:updates
                                               indicesTensor:squeezedIndices
                                                        mode:mode
                                                        name:nil];
        return Result(ctx, result, "scatter");
    }

    // Handle common embedding gradient pattern (reverse of gather):
    // input: [num_embeddings, embedding_dim] - zeros initially
    // indices: [batch..., 1] where last dim is index vector
    // updates: [batch..., embedding_dim] - gradients to scatter
    // Result: accumulate updates into input at specified indices

    // Check for the common pattern where:
    // - index_vector_dim is the last dimension of indices
    // - indices has size 1 in that dimension
    // - we're scattering along a single dimension
    // - no batching dimensions
    if (inputBatchingDims.empty() && indexVectorDim == (int64_t)indicesRank - 1 &&
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

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        // Ensure updates is at least rank 1 (MPS doesn't support scalar updates)
        if (updates.shape.count == 0)
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];

        // MPS scatter requires updates rank to match operand rank. When updates has higher
        // rank (e.g. 2D batch of indices into 2D embedding table), MPS cannot handle it.
        if (updates.shape.count > input.shape.count) {
            return ProcessResult::Error(
                "scatter: MPS does not support scatter where updates rank (" +
                std::to_string(updates.shape.count) + ") > operand rank (" +
                std::to_string(input.shape.count) +
                "). This can occur with multi-dimensional indices into embeddings.");
        }

        // Scalar index updates in StableHLO can drop the scattered axis from the update
        // shape (e.g. input [10,1,4], updates [1,4] for axis 0). MPS scatter expects the
        // update tensor rank to match the operand rank, so reinsert singleton axes as needed.
        if (updates.shape.count < input.shape.count) {
            NSArray<NSNumber*>* updatesShape = updates.shape;
            NSMutableArray<NSNumber*>* alignedUpdatesShape =
                [NSMutableArray arrayWithCapacity:input.shape.count];
            NSUInteger updateDimIndex = 0;

            for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                if ((int64_t)dim == scatterAxis) {
                    [alignedUpdatesShape addObject:@1];
                    continue;
                }

                if (updateDimIndex >= updatesShape.count) {
                    [alignedUpdatesShape addObject:@1];
                    continue;
                }

                [alignedUpdatesShape addObject:updatesShape[updateDimIndex++]];
            }

            updates = [ctx.graph reshapeTensor:updates withShape:alignedUpdatesShape name:nil];
        }

        // Use scatterWithDataTensor to scatter updates into input
        MPSGraphTensor* result =
            [ctx.graph scatterWithDataTensor:input
                               updatesTensor:updates
                               indicesTensor:squeezedIndices
                                        axis:static_cast<NSInteger>(scatterAxis)
                                        mode:mode
                                        name:nil];
        return Result(ctx, result, "scatter");
    }

    // Handle full-rank point updates (e.g. x.at[0,0,...].set(value) on MPS):
    // the update index tensor stores a full index vector and update_window_dims is empty.
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    if (updateWindowDims.empty() && inputBatchingDims.empty() &&
        scatterDimsToOperandDims.size() == input.shape.count &&
        insertedWindowDims.size() == input.shape.count && indicesRank == 1 &&
        [indicesShape[0] integerValue] == (NSInteger)input.shape.count && indexVectorDim == 0) {
        bool fullRange = true;
        NSUInteger inputRank = input.shape.count;
        for (NSUInteger dim = 0; dim < inputRank; ++dim) {
            if (scatterDimsToOperandDims[dim] != (int64_t)dim ||
                insertedWindowDims[dim] != (int64_t)dim) {
                fullRange = false;
                break;
            }
        }
        if (!fullRange) {
            return ProcessResult::Error("scatter: unsupported full-rank scatter pattern");
        }

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        // MPS scatterND expects indices as [N, rank]. For full point updates with a scalar
        // index vector (e.g. [0,0,0]), reshape to one update row.
        MPSGraphTensor* ndIndices = [ctx.graph reshapeTensor:scatterIndices
                                                   withShape:@[@1, indicesShape[0]]
                                                        name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        if (updates.shape.count == 0) {
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];
        }

        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:ndIndices
                                                    batchDimensions:0
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    return ProcessResult::Error("scatter: unsupported scatter pattern");
}
REGISTER_MPS_OP("stablehlo.scatter", HandleScatter);

// Reverse - reverse elements along specified dimensions
static ProcessResult HandleReverse(HandlerContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(ctx.op);
    if (!reverseOp) {
        return ProcessResult::Error("reverse: expected ReverseOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reverse: missing input tensor");

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    MPSGraphTensor* result = [ctx.graph reverseTensor:input axes:axes name:nil];
    return Result(ctx, result, "reverse");
}
REGISTER_MPS_OP("stablehlo.reverse", HandleReverse);

}  // namespace jax_mps
