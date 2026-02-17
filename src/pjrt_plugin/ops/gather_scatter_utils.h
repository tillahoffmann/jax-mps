// Centralized safe wrappers for MPS gather/scatter operations.
//
// MPSGraph's gather operations internally convert integer data to float32,
// causing precision loss for 32-bit and 64-bit integers with values > 2^24.
// This affects random key operations like jax.random.split().
//
// These wrappers apply the bitcast workaround:
// 1. Bitcast 32-bit integers to float32 (preserving bit patterns, not values)
// 2. Perform the gather/scatter on the "float32" data
// 3. Bitcast back to the original integer type
// For 64-bit integers, we reshape to pairs of 32-bit values first.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace jax_mps {

// Helper to check if a type needs the bitcast workaround
inline bool NeedsBitcastWorkaround(MPSDataType type) {
    return type == MPSDataTypeInt32 || type == MPSDataTypeUInt32 || type == MPSDataTypeInt64 ||
           type == MPSDataTypeUInt64;
}

inline bool Is64BitInteger(MPSDataType type) {
    return type == MPSDataTypeInt64 || type == MPSDataTypeUInt64;
}

// Prepare a tensor for gather/scatter by bitcasting integers to float32.
// Returns the prepared tensor, and sets the output parameters for reversal.
inline MPSGraphTensor* PrepareIntegerTensor(MPSGraph* graph, MPSGraphTensor* input,
                                            MPSDataType& originalType, bool& needsReverse,
                                            bool& is64Bit) {
    originalType = input.dataType;
    needsReverse = (originalType == MPSDataTypeInt32 || originalType == MPSDataTypeUInt32);
    is64Bit = Is64BitInteger(originalType);

    if (is64Bit) {
        // Reshape input: [shape...] -> [shape..., 2] treating each 64-bit as two 32-bits
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithArray:input.shape];
        [expandedShape addObject:@2];

        // Reinterpret as uint32 pairs
        input = [graph reinterpretCastTensor:input toType:MPSDataTypeUInt32 name:nil];
        input = [graph reshapeTensor:input withShape:expandedShape name:nil];
        needsReverse = true;
    }

    if (needsReverse) {
        // Bitcast to float32 (reinterpret bits, no conversion)
        input = [graph reinterpretCastTensor:input toType:MPSDataTypeFloat32 name:nil];
    }

    return input;
}

// Reverse the PrepareIntegerTensor operation - bitcasts back to original type.
inline MPSGraphTensor* FinalizeIntegerTensor(MPSGraph* graph, MPSGraphTensor* result,
                                             MPSDataType originalType, bool needsReverse,
                                             bool is64Bit) {
    if (needsReverse) {
        // MPS reinterpret_cast doesn't work on scalar tensors (rank 0).
        // If the result is a scalar, reshape to [1], cast, then reshape back.
        bool isScalar = result.shape.count == 0;
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[@1] name:nil];
        }

        // Bitcast back to original type (or uint32 for 64-bit case)
        MPSDataType targetType = is64Bit ? MPSDataTypeUInt32 : originalType;
        result = [graph reinterpretCastTensor:result toType:targetType name:nil];

        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[] name:nil];
        }
    }

    if (is64Bit) {
        // Reinterpret as original 64-bit type
        bool isScalar = result.shape.count == 0;
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[@1] name:nil];
        }
        result = [graph reinterpretCastTensor:result toType:originalType name:nil];
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[] name:nil];
        }
    }

    return result;
}

// Safe wrapper for gatherNDWithUpdatesTensor
// Handles int32/uint32/int64/uint64 precision fix internally
inline MPSGraphTensor* SafeGatherND(MPSGraph* graph, MPSGraphTensor* updatesTensor,
                                    MPSGraphTensor* indicesTensor, NSUInteger batchDimensions) {
    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherNDWithUpdatesTensor:updatesTensor
                                                indicesTensor:indicesTensor
                                              batchDimensions:batchDimensions
                                                         name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Safe wrapper for gatherWithUpdatesTensor (axis-based gather)
inline MPSGraphTensor* SafeGather(MPSGraph* graph, MPSGraphTensor* updatesTensor,
                                  MPSGraphTensor* indicesTensor, NSUInteger axis,
                                  NSUInteger batchDimensions) {
    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherWithUpdatesTensor:updatesTensor
                                              indicesTensor:indicesTensor
                                                       axis:axis
                                            batchDimensions:batchDimensions
                                                       name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Safe wrapper for gatherAlongAxis
inline MPSGraphTensor* SafeGatherAlongAxis(MPSGraph* graph, NSInteger axis,
                                           MPSGraphTensor* updatesTensor,
                                           MPSGraphTensor* indicesTensor) {
    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherAlongAxis:axis
                                  withUpdatesTensor:updatesTensor
                                      indicesTensor:indicesTensor
                                               name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Safe wrapper for scatterNDWithDataTensor
// Scatter operations have the same float32 conversion bug as gather operations.
// Testing confirmed that scatter corrupts uint32 values > 2^24 identically to gather.
inline MPSGraphTensor* SafeScatterND(MPSGraph* graph, MPSGraphTensor* dataTensor,
                                     MPSGraphTensor* updatesTensor, MPSGraphTensor* indicesTensor,
                                     NSUInteger batchDimensions, MPSGraphScatterMode mode) {
    // For scatter, both data and updates need the workaround if they're integers
    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterNDWithDataTensor:dataTensor
                                              updatesTensor:updatesTensor
                                              indicesTensor:indicesTensor
                                            batchDimensions:batchDimensions
                                                       mode:mode
                                                       name:nil];

    // The result should match the data tensor type
    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

// Safe wrapper for scatterWithDataTensor (axis-based scatter)
inline MPSGraphTensor* SafeScatter(MPSGraph* graph, MPSGraphTensor* dataTensor,
                                   MPSGraphTensor* updatesTensor, MPSGraphTensor* indicesTensor,
                                   NSInteger axis, MPSGraphScatterMode mode) {
    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterWithDataTensor:dataTensor
                                            updatesTensor:updatesTensor
                                            indicesTensor:indicesTensor
                                                     axis:axis
                                                     mode:mode
                                                     name:nil];

    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

// Safe wrapper for scatterAlongAxis
inline MPSGraphTensor* SafeScatterAlongAxis(MPSGraph* graph, NSInteger axis,
                                            MPSGraphTensor* dataTensor,
                                            MPSGraphTensor* updatesTensor,
                                            MPSGraphTensor* indicesTensor,
                                            MPSGraphScatterMode mode) {
    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterAlongAxis:axis
                                      withDataTensor:dataTensor
                                       updatesTensor:updatesTensor
                                       indicesTensor:indicesTensor
                                                mode:mode
                                                name:nil];

    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

}  // namespace jax_mps
