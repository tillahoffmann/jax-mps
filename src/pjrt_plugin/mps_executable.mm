#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <unordered_map>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
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
    if (num_outputs_ == 0)
        num_outputs_ = 1;

    valid_ = true;
}

MpsExecutable::~MpsExecutable() {
    // MLIR context and module are automatically cleaned up via unique_ptr/OwningOpRef
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

// Helper to get PJRT dtype from MLIR element type
static int MlirTypeToPjrtDtype(mlir::Type elemType) {
    if (elemType.isF32())
        return 11;  // PJRT_F32
    if (elemType.isF16())
        return 10;  // PJRT_F16
    if (elemType.isBF16())
        return 16;  // PJRT_BF16
    if (elemType.isF64())
        return 12;  // PJRT_F64

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();

        if (width == 1)
            return 1;  // PJRT_PRED
        if (width == 8)
            return isUnsigned ? 6 : 2;  // PJRT_U8 or PJRT_S8
        if (width == 16)
            return isUnsigned ? 7 : 3;  // PJRT_U16 or PJRT_S16
        if (width == 32)
            return isUnsigned ? 8 : 4;  // PJRT_U32 or PJRT_S32
        if (width == 64)
            return isUnsigned ? 9 : 5;  // PJRT_U64 or PJRT_S64
    }

    return -1;
}

// Result type for processOperations - can be an error or return values
struct ProcessResult {
    std::string error;
    std::vector<mlir::Value> return_values;

    bool ok() const {
        return error.empty();
    }
    static ProcessResult Error(const std::string& msg) {
        ProcessResult r;
        r.error = msg;
        return r;
    }
};

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

        // Handle func.return - collect return values
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
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

        // Look up handler in registry
        OpHandler handler = OpRegistry::Find(op_name);
        if (!handler) {
            std::string supported = OpRegistry::ListRegistered();
            return ProcessResult::Error(
                "Unsupported operation: '" + op_name +
                "'. The MPS backend does not have a handler for this operation. "
                "Supported operations: " +
                supported);
        }

        // Get output shape for this operation
        NSArray<NSNumber*>* output_shape = nil;
        if (op->getNumResults() > 0) {
            output_shape = GetShapeFromType(op->getResult(0).getType());
        }

        // Check for multi-result operations (not yet supported)
        if (op->getNumResults() > 1) {
            return ProcessResult::Error("Operation '" + op_name +
                                        "' has multiple results which is not yet supported");
        }

        // Execute the handler
        MPSGraphTensor* out = handler(graph, op, values, output_shape);
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

ExecutionResult MpsExecutable::Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device) {
    ExecutionResult result;

    // Check for compilation errors
    if (!valid_) {
        return ExecutionResult::Error("Cannot execute: compilation failed - " + error_);
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

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) {
            return ExecutionResult::Error("Failed to create MPSGraph. Metal may not be available.");
        }

        // Value map: mlir::Value (opaque pointer) -> MPSGraphTensor*
        ValueMap values;

        // Create placeholder tensors for function arguments
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
            [NSMutableDictionary dictionary];

        // Validate input count
        size_t num_args = entry_func_.getNumArguments();
        if (inputs.size() < num_args) {
            return ExecutionResult::Error("Input count mismatch: expected " +
                                          std::to_string(num_args) + " inputs, got " +
                                          std::to_string(inputs.size()));
        }

        // Create placeholders for each function argument
        for (size_t i = 0; i < num_args && i < inputs.size(); i++) {
            mlir::BlockArgument arg = entry_func_.getArgument(i);
            MpsBuffer* input = inputs[i];

            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }

            // Get shape from argument type
            NSArray<NSNumber*>* shape = GetShapeFromType(arg.getType());
            if (!shape) {
                return ExecutionResult::Error("Invalid argument type at index " +
                                              std::to_string(i));
            }

            MPSDataType mps_dtype = PjrtDtypeToMps(input->dtype());
            if (mps_dtype == MPSDataTypeInvalid) {
                return ExecutionResult::Error("Unsupported data type (PJRT dtype " +
                                              std::to_string(input->dtype()) +
                                              ") for input at index " + std::to_string(i));
            }

            // Create placeholder
            NSString* argName = [NSString stringWithFormat:@"arg%zu", i];
            MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                             dataType:mps_dtype
                                                                 name:argName];

            // Map the MLIR argument to the placeholder tensor
            values[arg.getAsOpaquePointer()] = placeholder;

            // Create tensor data from input buffer
            id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)input->metal_buffer();
            if (!mtl_buffer) {
                return ExecutionResult::Error("Input buffer at index " + std::to_string(i) +
                                              " has no Metal buffer");
            }
            MPSGraphTensorData* tensor_data =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buffer
                                                        shape:shape
                                                     dataType:mps_dtype];
            feeds[placeholder] = tensor_data;
        }

        // Process operations, handling func.call recursively
        ProcessResult processResult;
        @try {
            mlir::Block& entryBlock = entry_func_.front();
            processResult = processOperations(graph, entryBlock, values, *module_, 0);
        } @catch (NSException* exception) {
            return ExecutionResult::Error("MPS operation failed with Objective-C exception: " +
                                          std::string([[exception name] UTF8String]) + " - " +
                                          std::string([[exception reason] UTF8String]));
        }

        if (!processResult.ok()) {
            return ExecutionResult::Error(processResult.error);
        }

        std::vector<mlir::Value>& return_values = processResult.return_values;

        // Handle identity functions - computation with no ops that just passes through input
        if (return_values.empty() && !inputs.empty()) {
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

            // Create a new buffer with copied data
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

        // Collect target tensors from return values
        NSMutableArray<MPSGraphTensor*>* target_tensors = [NSMutableArray array];
        std::vector<mlir::Type> return_types;

        for (const auto& ret_value : return_values) {
            MPSGraphTensor* ret_tensor = values[ret_value.getAsOpaquePointer()];
            if (!ret_tensor) {
                // Print the MLIR value for debugging
                std::string valStr;
                llvm::raw_string_ostream os(valStr);
                ret_value.print(os);
                return ExecutionResult::Error("Return value '" + valStr + "' not found in tensors");
            }
            [target_tensors addObject:ret_tensor];
            return_types.push_back(ret_value.getType());
        }

        if (target_tensors.count == 0) {
            return ExecutionResult::Error(
                "No result tensor produced. "
                "This indicates an internal error in the MPS graph construction.");
        }

        // Execute graph
        id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];
        if (!commandQueue) {
            return ExecutionResult::Error("Failed to create Metal command queue");
        }

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* result_dict = nil;
        @try {
            result_dict = [graph runWithMTLCommandQueue:commandQueue
                                                  feeds:feeds
                                          targetTensors:target_tensors
                                       targetOperations:nil];
        } @catch (NSException* exception) {
            return ExecutionResult::Error(
                "MPS graph execution failed with Objective-C exception: " +
                std::string([[exception name] UTF8String]) + " - " +
                std::string([[exception reason] UTF8String]));
        }

        // Process each output
        for (size_t i = 0; i < target_tensors.count; i++) {
            MPSGraphTensor* target = target_tensors[i];
            MPSGraphTensorData* result_data = result_dict[target];
            if (!result_data) {
                return ExecutionResult::Error("MPS graph execution produced no result for output " +
                                              std::to_string(i));
            }

            // Get result shape
            std::vector<int64_t> result_shape;
            for (NSNumber* dim in result_data.shape) {
                result_shape.push_back([dim longLongValue]);
            }

            // Get dtype from MLIR type
            mlir::Type elemType =
                mlir::cast<mlir::RankedTensorType>(return_types[i]).getElementType();
            int output_dtype = MlirTypeToPjrtDtype(elemType);
            if (output_dtype < 0) {
                return ExecutionResult::Error("Unsupported output type for result " +
                                              std::to_string(i));
            }

            // Calculate byte size
            size_t byte_size = 1;
            for (int64_t dim : result_shape) {
                byte_size *= dim;
            }
            byte_size *= DtypeByteSize(output_dtype);

            // Create output buffer with shared storage
            id<MTLBuffer> output_buffer =
                [mtl_device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
            if (!output_buffer) {
                return ExecutionResult::Error("Failed to allocate output buffer of size " +
                                              std::to_string(byte_size) + " bytes");
            }

            // Copy result data using MPSNDArray
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
