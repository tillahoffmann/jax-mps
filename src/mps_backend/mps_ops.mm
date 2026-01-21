// Metal Performance Shaders Graph operations
// This file contains helper functions for MPSGraph operations

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

namespace jax_mps {
namespace ops {

// Helper to create an MPSGraph tensor from shape and dtype
MPSGraphTensor* CreatePlaceholder(MPSGraph* graph,
                                   NSArray<NSNumber*>* shape,
                                   MPSDataType dtype,
                                   NSString* name) {
    return [graph placeholderWithShape:shape dataType:dtype name:name];
}

// Element-wise addition
MPSGraphTensor* Add(MPSGraph* graph,
                    MPSGraphTensor* lhs,
                    MPSGraphTensor* rhs) {
    return [graph additionWithPrimaryTensor:lhs
                            secondaryTensor:rhs
                                       name:nil];
}

// Element-wise subtraction
MPSGraphTensor* Subtract(MPSGraph* graph,
                         MPSGraphTensor* lhs,
                         MPSGraphTensor* rhs) {
    return [graph subtractionWithPrimaryTensor:lhs
                               secondaryTensor:rhs
                                          name:nil];
}

// Element-wise multiplication
MPSGraphTensor* Multiply(MPSGraph* graph,
                         MPSGraphTensor* lhs,
                         MPSGraphTensor* rhs) {
    return [graph multiplicationWithPrimaryTensor:lhs
                                  secondaryTensor:rhs
                                             name:nil];
}

// Element-wise division
MPSGraphTensor* Divide(MPSGraph* graph,
                       MPSGraphTensor* lhs,
                       MPSGraphTensor* rhs) {
    return [graph divisionWithPrimaryTensor:lhs
                            secondaryTensor:rhs
                                       name:nil];
}

// Matrix multiplication
MPSGraphTensor* MatMul(MPSGraph* graph,
                       MPSGraphTensor* lhs,
                       MPSGraphTensor* rhs) {
    return [graph matrixMultiplicationWithPrimaryTensor:lhs
                                        secondaryTensor:rhs
                                                   name:nil];
}

// Tanh activation
MPSGraphTensor* Tanh(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph tanhWithTensor:input name:nil];
}

// ReLU activation
MPSGraphTensor* Relu(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph reLUWithTensor:input name:nil];
}

// Sigmoid activation
MPSGraphTensor* Sigmoid(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph sigmoidWithTensor:input name:nil];
}

// Exponential
MPSGraphTensor* Exp(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph exponentWithTensor:input name:nil];
}

// Natural logarithm
MPSGraphTensor* Log(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph logarithmWithTensor:input name:nil];
}

// Square root
MPSGraphTensor* Sqrt(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph squareRootWithTensor:input name:nil];
}

// Negation
MPSGraphTensor* Negate(MPSGraph* graph, MPSGraphTensor* input) {
    return [graph negativeWithTensor:input name:nil];
}

// Reduce sum
MPSGraphTensor* ReduceSum(MPSGraph* graph,
                          MPSGraphTensor* input,
                          NSArray<NSNumber*>* axes) {
    return [graph reductionSumWithTensor:input
                                    axes:axes
                                    name:nil];
}

// Transpose
MPSGraphTensor* Transpose(MPSGraph* graph,
                          MPSGraphTensor* input,
                          NSUInteger dim0,
                          NSUInteger dim1) {
    return [graph transposeTensor:input
                        dimension:dim0
                    withDimension:dim1
                             name:nil];
}

// Reshape
MPSGraphTensor* Reshape(MPSGraph* graph,
                        MPSGraphTensor* input,
                        NSArray<NSNumber*>* shape) {
    return [graph reshapeTensor:input
                      withShape:shape
                           name:nil];
}

// Broadcast to shape
MPSGraphTensor* BroadcastTo(MPSGraph* graph,
                            MPSGraphTensor* input,
                            NSArray<NSNumber*>* shape) {
    return [graph broadcastTensor:input
                          toShape:shape
                             name:nil];
}

}  // namespace ops
}  // namespace jax_mps
