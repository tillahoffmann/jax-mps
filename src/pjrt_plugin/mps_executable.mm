#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

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
    num_outputs_ = entry_func_.getNumResults();

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
