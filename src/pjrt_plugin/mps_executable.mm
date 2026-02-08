#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cctype>
#include <unordered_map>

#import "pjrt_plugin/issue_url.h"
#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/ops/control_flow_ops.h"
#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

MpsExecutable::MpsExecutable(MpsClient* client, mps::ParsedModule module)
    : client_(client), name_("main") {
    if (!module.ok()) {
        error_ = "Invalid ParsedModule";
        valid_ = false;
        return;
    }

    // Take ownership of the MLIR context and module
    context_ = std::move(module.context);
    module_ = std::move(module.module);
    entry_func_ = module.entry_func;

    // Get the function name
    name_ = entry_func_.getName().str();

    // Set the number of outputs based on result types
    num_outputs_ = static_cast<int>(entry_func_.getNumResults());

    valid_ = true;
}

MpsExecutable::~MpsExecutable() {
    // MLIR context and module are automatically cleaned up via unique_ptr/OwningOpRef
    // Release the cached graph (graph owns all its tensors)
    if (cached_graph_) {
        MPSGraph* graph = (__bridge_transfer MPSGraph*)cached_graph_;
        (void)graph;  // ARC will release
        cached_graph_ = nullptr;
    }
    // Note: cached_placeholders_ and cached_targets_ point to tensors owned by the graph,
    // so we don't release them separately - they'll be freed when the graph is freed.
}

// Helper to get output shape from MLIR type
static NSArray<NSNumber*>* GetShapeFromType(mlir::Type type) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType) {
        return nil;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (int64_t dim : tensorType.getShape()) {
        [shape addObject:@(dim)];
    }
    return shape;
}

// Forward declaration for recursive processing
static ProcessResult processOperations(MPSGraph* graph, mlir::Block& block, ValueMap& values,
                                       mlir::ModuleOp module, int depth);

enum class ArgReduceKind { kUnknown, kMax, kMin };

static bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

static ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        // If the compare operands are swapped, the max/min interpretation flips.
        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

static ProcessResult processMultiResultReduceOp(MPSGraph* graph, mlir::Operation* op,
                                                ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return ProcessResult::Error("Expected stablehlo.reduce");
    }
    if (op->getNumResults() != 2 || op->getNumOperands() < 2) {
        return ProcessResult::Error("Unsupported multi-result reduce shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(values, op, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("Only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("Unsupported multi-result reduce body");
    }

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (kind == ArgReduceKind::kMax) {
        valueOut = [graph reductionMaximumWithTensor:valueInput axis:axis name:nil];
        indexOut = [graph reductionArgMaximumWithTensor:valueInput axis:axis name:nil];
    } else {
        valueOut = [graph reductionMinimumWithTensor:valueInput axis:axis name:nil];
        indexOut = [graph reductionArgMinimumWithTensor:valueInput axis:axis name:nil];
    }
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("Failed to lower multi-result reduce");
    }

    MPSDataType valueType = GetResultMpsType(op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [graph castTensor:indexOut toType:indexType name:nil];
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(op, 0);
    if (valueShape && valueOut) {
        valueOut = [graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(op, 1);
    if (indexShape && indexOut) {
        indexOut = [graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    values[op->getResult(0).getAsOpaquePointer()] = valueOut;
    values[op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

static ProcessResult processSortOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        return ProcessResult::Error("Expected stablehlo.sort");
    }

    auto dimAttr = op->getAttrOfType<mlir::IntegerAttr>("dimension");
    if (!dimAttr) {
        return ProcessResult::Error("stablehlo.sort missing dimension attribute");
    }
    NSInteger axis = (NSInteger)dimAttr.getInt();

    bool descending = false;
    if (!sortOp.getComparator().empty()) {
        for (mlir::Operation& nestedOp : sortOp.getComparator().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            auto direction = compareOp.getComparisonDirection();
            descending = direction == mlir::stablehlo::ComparisonDirection::GT ||
                         direction == mlir::stablehlo::ComparisonDirection::GE;
            break;
        }
    }

    if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
        MPSGraphTensor* input = GetInputTensor(values, op, 0);
        if (!input) {
            return ProcessResult::Error("stablehlo.sort input tensor not found");
        }
        MPSGraphTensor* sorted =
            [graph sortWithTensor:input axis:axis descending:descending name:nil];
        if (!sorted) {
            return ProcessResult::Error("stablehlo.sort lowering failed");
        }
        values[op->getResult(0).getAsOpaquePointer()] = sorted;
        return ProcessResult{};
    }

    if (op->getNumOperands() >= 2 && op->getNumOperands() == op->getNumResults()) {
        // Tuple sort lowering used by lexsort-like patterns:
        // sort first N-1 tensors as lexicographic keys, apply permutation to all tensors.
        // We build permutation by stable-sorting from least-significant key to most-significant.
        MPSGraphTensor* base = GetInputTensor(values, op, 0);
        if (!base) {
            return ProcessResult::Error("stablehlo.sort key tensor not found");
        }

        MPSGraphTensor* perm = [graph coordinateAlongAxis:axis withShape:base.shape name:nil];
        perm = EnsureInt32(graph, perm);

        for (NSInteger keyIdx = (NSInteger)op->getNumOperands() - 2; keyIdx >= 0; --keyIdx) {
            MPSGraphTensor* keyTensor = GetInputTensor(values, op, (unsigned)keyIdx);
            if (!keyTensor) {
                return ProcessResult::Error("stablehlo.sort key tensor missing");
            }
            MPSGraphTensor* keyAtPerm = [graph gatherAlongAxis:axis
                                              withUpdatesTensor:keyTensor
                                                  indicesTensor:perm
                                                           name:nil];
            MPSGraphTensor* localOrder =
                [graph argSortWithTensor:keyAtPerm axis:axis descending:descending name:nil];
            if (!localOrder) {
                return ProcessResult::Error("stablehlo.sort argSort lowering failed");
            }
            localOrder = EnsureInt32(graph, localOrder);
            perm = [graph gatherAlongAxis:axis
                        withUpdatesTensor:perm
                            indicesTensor:localOrder
                                     name:nil];
        }

        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            MPSGraphTensor* operandTensor = GetInputTensor(values, op, i);
            if (!operandTensor) {
                return ProcessResult::Error("stablehlo.sort operand tensor missing");
            }
            MPSGraphTensor* sorted = [graph gatherAlongAxis:axis
                                           withUpdatesTensor:operandTensor
                                               indicesTensor:perm
                                                        name:nil];
            if (!sorted) {
                return ProcessResult::Error("stablehlo.sort gather lowering failed");
            }
            values[op->getResult(i).getAsOpaquePointer()] = sorted;
        }
        return ProcessResult{};
    }

    return ProcessResult::Error("Unsupported stablehlo.sort operand/result shape");
}

static int64_t inferTopKFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count ||
        outputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return outDim;
        }
    }
    return [outputShape.lastObject longLongValue];
}

static NSInteger inferTopKAxisFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count ||
        inputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return (NSInteger)i;
        }
    }
    return (NSInteger)inputShape.count - 1;
}

static bool isTopKCustomCallTarget(const std::string& target) {
    std::string lower = target;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lower == "topk" || lower == "top_k";
}

static ProcessResult processTopKOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values) {
    if (op->getNumOperands() < 1 || op->getNumResults() != 2) {
        return ProcessResult::Error("Unsupported top_k operand/result shape");
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        return ProcessResult::Error("top_k input tensor not found");
    }

    int64_t k = -1;
    if (auto kAttr = op->getAttrOfType<mlir::IntegerAttr>("k")) {
        k = kAttr.getInt();
    }

    NSInteger axis = -1;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
        axis = (NSInteger)axisAttr.getInt();
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(op, 0);
    if (k < 0) {
        k = inferTopKFromShapes(input.shape, valueShape);
    }
    if (axis < 0) {
        axis = inferTopKAxisFromShapes(input.shape, valueShape);
    }

    if (k <= 0) {
        return ProcessResult::Error("Failed to infer top_k parameter k");
    }
    if (axis < 0 || !input.shape || axis >= (NSInteger)input.shape.count) {
        return ProcessResult::Error("Failed to infer top_k axis");
    }

    NSArray<MPSGraphTensor*>* topk = nil;
    if (axis == (NSInteger)input.shape.count - 1) {
        topk = [graph topKWithSourceTensor:input k:(NSUInteger)k name:nil];
    } else {
        topk = [graph topKWithSourceTensor:input axis:axis k:(NSUInteger)k name:nil];
    }
    if (!topk || topk.count != 2) {
        return ProcessResult::Error("top_k lowering failed");
    }

    MPSGraphTensor* valueOut = topk[0];
    MPSGraphTensor* indexOut = topk[1];

    MPSDataType valueType = GetResultMpsType(op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [graph castTensor:indexOut toType:indexType name:nil];
    }

    if (valueShape) {
        valueOut = [graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(op, 1);
    if (indexShape) {
        indexOut = [graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    values[op->getResult(0).getAsOpaquePointer()] = valueOut;
    values[op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

// Process a func.call operation by looking up the callee and processing its body
static ProcessResult processCallOp(MPSGraph* graph, mlir::func::CallOp callOp, ValueMap& values,
                                   mlir::ModuleOp module, int depth) {
    if (depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive function");
    }

    // Look up the callee function
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
    if (!callee) {
        return ProcessResult::Error("Could not find callee function: " + callOp.getCallee().str());
    }

    // Check that callee has a body
    if (callee.empty()) {
        return ProcessResult::Error("Callee function has no body: " + callOp.getCallee().str());
    }

    // Map call arguments to function parameters
    auto& calleeBlock = callee.front();
    for (size_t i = 0; i < callOp.getNumOperands() && i < calleeBlock.getNumArguments(); i++) {
        mlir::Value callArg = callOp.getOperand(i);
        mlir::BlockArgument funcArg = calleeBlock.getArgument(i);

        // Get the tensor for the call argument
        MPSGraphTensor* argTensor = GetTensor(values, callArg);
        if (!argTensor) {
            return ProcessResult::Error("Call argument " + std::to_string(i) + " not found");
        }

        // Map the function argument to the same tensor
        values[funcArg.getAsOpaquePointer()] = argTensor;
    }

    // Process the callee function's body
    ProcessResult calleeResult = processOperations(graph, calleeBlock, values, module, depth + 1);
    if (!calleeResult.ok()) {
        return calleeResult;
    }

    // Map return values back to call results
    for (size_t i = 0; i < callOp.getNumResults() && i < calleeResult.return_values.size(); i++) {
        mlir::Value returnValue = calleeResult.return_values[i];
        MPSGraphTensor* returnTensor = GetTensor(values, returnValue);
        if (!returnTensor) {
            return ProcessResult::Error("Return value " + std::to_string(i) +
                                        " not found in callee");
        }
        values[callOp.getResult(i).getAsOpaquePointer()] = returnTensor;
    }

    return ProcessResult{};
}

// Process operations in a block, handling func.call recursively
static ProcessResult processOperations(MPSGraph* graph, mlir::Block& block, ValueMap& values,
                                       mlir::ModuleOp module, int depth) {
    ProcessResult result;

    for (mlir::Operation& operation : block) {
        mlir::Operation* op = &operation;
        std::string op_name = op->getName().getStringRef().str();

        // Handle function and StableHLO region returns.
        if (mlir::isa<mlir::func::ReturnOp>(op) || mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
            for (mlir::Value operand : op->getOperands()) {
                result.return_values.push_back(operand);
            }
            continue;
        }

        // Handle func.call - process the callee function recursively
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            ProcessResult callResult = processCallOp(graph, callOp, values, module, depth);
            if (!callResult.ok()) {
                return callResult;
            }
            continue;
        }

        // Check for control flow ops (stablehlo.while, stablehlo.case)
        if (IsControlFlowOp(op_name)) {
            ProcessResult cfResult;
            if (op_name == "stablehlo.while") {
                cfResult = HandleWhileOp(graph, op, values, module, depth, processOperations);
            } else {
                cfResult = HandleCaseOp(graph, op, values, module, depth, processOperations);
            }
            if (!cfResult.ok())
                return cfResult;
            continue;
        }

        if (op_name == "stablehlo.reduce" && op->getNumResults() > 1) {
            ProcessResult multiReduceResult = processMultiResultReduceOp(graph, op, values);
            if (!multiReduceResult.ok())
                return multiReduceResult;
            continue;
        }
        if (op_name == "stablehlo.sort") {
            ProcessResult sortResult = processSortOp(graph, op, values);
            if (!sortResult.ok())
                return sortResult;
            continue;
        }
        if (op_name == "chlo.top_k") {
            ProcessResult topKResult = processTopKOp(graph, op, values);
            if (!topKResult.ok())
                return topKResult;
            continue;
        }
        if (op_name == "stablehlo.custom_call" && op->getNumResults() == 2) {
            if (auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
                std::string target = customCallOp.getCallTargetName().str();
                if (isTopKCustomCallTarget(target)) {
                    ProcessResult topKResult = processTopKOp(graph, op, values);
                    if (!topKResult.ok())
                        return topKResult;
                    continue;
                }
            }
        }

        // Look up handler in registry
        OpHandler handler = OpRegistry::Find(op_name);
        if (!handler) {
            return ProcessResult::Error(UnsupportedOpsMessage({op_name}) +
                                        "\n\n"
                                        "Supported operations: " +
                                        OpRegistry::ListRegistered());
        }

        // Check for multi-result operations (not yet supported)
        if (op->getNumResults() > 1) {
            return ProcessResult::Error("Operation '" + op_name +
                                        "' has multiple results which is not yet supported");
        }

        // Execute the handler
        MPSGraphTensor* out = handler(graph, op, values);
        if (!out) {
            return ProcessResult::Error("Operation '" + op_name + "' handler returned null");
        }

        // Map the result to the output tensor
        if (op->getNumResults() > 0) {
            values[op->getResult(0).getAsOpaquePointer()] = out;
        }
    }

    return result;
}

bool MpsExecutable::BuildGraph() {
    if (graph_built_)
        return true;

    @autoreleasepool {
        if (!client_) {
            error_ = "No MPS client available";
            return false;
        }

        // Create and cache MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) {
            error_ = "Failed to create MPSGraph";
            return false;
        }

        // Value map for building the graph
        ValueMap values;

        // Create placeholders for each function argument
        size_t num_args = entry_func_.getNumArguments();
        cached_placeholders_.resize(num_args);

        for (size_t i = 0; i < num_args; i++) {
            mlir::BlockArgument arg = entry_func_.getArgument(i);
            NSArray<NSNumber*>* shape = GetShapeFromType(arg.getType());
            if (!shape) {
                error_ = "Invalid argument type at index " + std::to_string(i);
                return false;
            }

            // Get dtype from MLIR type
            auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
            MPSDataType mps_dtype =
                MlirTypeToPjrtDtype(tensorType.getElementType()) >= 0
                    ? PjrtDtypeToMps(MlirTypeToPjrtDtype(tensorType.getElementType()))
                    : MPSDataTypeFloat32;

            NSString* argName = [NSString stringWithFormat:@"arg%zu", i];
            MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                             dataType:mps_dtype
                                                                 name:argName];

            values[arg.getAsOpaquePointer()] = placeholder;
            cached_placeholders_[i] = (__bridge void*)placeholder;  // Graph owns the tensor
        }

        // Process operations to build the graph
        ProcessResult processResult;
        @try {
            mlir::Block& entryBlock = entry_func_.front();
            processResult = processOperations(graph, entryBlock, values, *module_, 0);
        } @catch (NSException* exception) {
            error_ = "MPS graph build failed: " + std::string([[exception reason] UTF8String]);
            return false;
        }

        if (!processResult.ok()) {
            error_ = processResult.error;
            return false;
        }

        // Cache target tensors and return types
        for (const auto& ret_value : processResult.return_values) {
            MPSGraphTensor* ret_tensor = values[ret_value.getAsOpaquePointer()];
            if (!ret_tensor) {
                error_ = "Return value not found in tensors";
                return false;
            }

            cached_targets_.push_back((__bridge void*)ret_tensor);  // Graph owns the tensor
            cached_return_types_.push_back(ret_value.getType());
        }

        // Cache the graph
        cached_graph_ = (__bridge_retained void*)graph;
        graph_built_ = true;
    }
    return true;
}

ExecutionResult MpsExecutable::Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device) {
    ExecutionResult result;

    // Check for compilation errors
    if (!valid_) {
        return ExecutionResult::Error("Cannot execute: compilation failed - " + error_);
    }

    // Build graph on first execution (lazy initialization)
    if (!graph_built_) {
        if (!BuildGraph()) {
            return ExecutionResult::Error("Graph build failed: " + error_);
        }
    }

    @autoreleasepool {
        // Verify Metal device is available
        if (!client_) {
            return ExecutionResult::Error("No MPS client available");
        }
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)client_->metal_device();
        if (!mtl_device) {
            return ExecutionResult::Error(
                "No Metal GPU device available. MPS backend requires Apple Silicon or AMD GPU.");
        }

        // Use cached graph
        MPSGraph* graph = (__bridge MPSGraph*)cached_graph_;

        // Create feeds dictionary from input buffers
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
            [NSMutableDictionary dictionary];

        size_t num_args = cached_placeholders_.size();
        if (inputs.size() < num_args) {
            return ExecutionResult::Error("Input count mismatch: expected " +
                                          std::to_string(num_args) + " inputs, got " +
                                          std::to_string(inputs.size()));
        }

        for (size_t i = 0; i < num_args; i++) {
            MpsBuffer* input = inputs[i];
            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }

            MPSGraphTensor* placeholder = (__bridge MPSGraphTensor*)cached_placeholders_[i];
            id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)input->metal_buffer();
            if (!mtl_buffer) {
                return ExecutionResult::Error("Input buffer at index " + std::to_string(i) +
                                              " has no Metal buffer");
            }

            // Get shape and dtype from placeholder
            MPSGraphTensorData* tensor_data =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buffer
                                                        shape:placeholder.shape
                                                     dataType:placeholder.dataType];
            feeds[placeholder] = tensor_data;
        }

        // Build target tensors array from cache
        NSMutableArray<MPSGraphTensor*>* target_tensors = [NSMutableArray array];
        for (void* p : cached_targets_) {
            [target_tensors addObject:(__bridge MPSGraphTensor*)p];
        }

        // Handle identity functions (no operations)
        if (target_tensors.count == 0 && !inputs.empty()) {
            MpsBuffer* input = inputs[0];
            if (!input) {
                return ExecutionResult::Error("Identity function with null input");
            }
            const auto& dims = input->dimensions();
            size_t byte_size = input->byte_size();
            id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input->metal_buffer();
            if (!input_buffer) {
                return ExecutionResult::Error("Identity function input has no Metal buffer");
            }

            id<MTLBuffer> output_buffer =
                [mtl_device newBufferWithBytes:input_buffer.contents
                                        length:byte_size
                                       options:MTLResourceStorageModeShared];
            if (!output_buffer) {
                return ExecutionResult::Error("Failed to allocate buffer for identity function");
            }

            auto buffer = std::make_unique<MpsBuffer>(device, (__bridge void*)output_buffer,
                                                      input->dtype(), dims);
            result.buffers.push_back(std::move(buffer));
            return result;
        }

        // Function with no outputs (e.g., side-effect-only or token functions)
        if (target_tensors.count == 0) {
            return result;
        }

        // Get cached command queue from client
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)client_->command_queue();
        if (!commandQueue) {
            return ExecutionResult::Error("No Metal command queue available");
        }

        // Execute the cached graph
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* result_dict = nil;
        @try {
            result_dict = [graph runWithMTLCommandQueue:commandQueue
                                                  feeds:feeds
                                          targetTensors:target_tensors
                                       targetOperations:nil];
        } @catch (NSException* exception) {
            return ExecutionResult::Error("MPS graph execution failed: " +
                                          std::string([[exception reason] UTF8String]));
        }

        // Process outputs using cached return types
        for (size_t i = 0; i < target_tensors.count; i++) {
            MPSGraphTensor* target = target_tensors[i];
            MPSGraphTensorData* result_data = result_dict[target];
            if (!result_data) {
                return ExecutionResult::Error("MPS graph execution produced no result for output " +
                                              std::to_string(i));
            }

            std::vector<int64_t> result_shape;
            for (NSNumber* dim in result_data.shape) {
                result_shape.push_back([dim longLongValue]);
            }

            mlir::Type elemType =
                mlir::cast<mlir::RankedTensorType>(cached_return_types_[i]).getElementType();
            int output_dtype = MlirTypeToPjrtDtype(elemType);
            if (output_dtype < 0) {
                return ExecutionResult::Error("Unsupported output type for result " +
                                              std::to_string(i));
            }

            size_t byte_size = 1;
            for (int64_t dim : result_shape) {
                byte_size *= dim;
            }
            byte_size *= DtypeByteSize(output_dtype);

            id<MTLBuffer> output_buffer =
                [mtl_device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
            if (!output_buffer) {
                return ExecutionResult::Error("Failed to allocate output buffer of size " +
                                              std::to_string(byte_size) + " bytes");
            }

            MPSNDArray* ndarray = [result_data mpsndarray];
            if (!ndarray) {
                return ExecutionResult::Error("Failed to get MPSNDArray from result data");
            }
            [ndarray readBytes:output_buffer.contents strideBytes:nil];

            auto buffer = std::make_unique<MpsBuffer>(device, (__bridge void*)output_buffer,
                                                      output_dtype, result_shape);
            result.buffers.push_back(std::move(buffer));
        }
    }

    return result;
}

}  // namespace jax_mps
