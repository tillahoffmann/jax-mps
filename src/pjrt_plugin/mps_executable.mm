#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <functional>
#include <unordered_map>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

// Map StableHLO element type string to PJRT dtype
// Returns -1 for unknown types
static int StablehloTypeToDtype(const std::string& type) {
    if (type == "f32")
        return 11;  // PJRT_F32
    if (type == "f16")
        return 10;  // PJRT_F16
    if (type == "bf16")
        return 16;  // PJRT_BF16
    if (type == "f64")
        return 12;  // PJRT_F64
    if (type == "i32" || type == "si32")
        return 4;  // PJRT_S32
    if (type == "i64" || type == "si64")
        return 5;  // PJRT_S64
    if (type == "ui32")
        return 8;  // PJRT_U32
    if (type == "ui64")
        return 9;  // PJRT_U64
    if (type == "i8" || type == "si8")
        return 2;  // PJRT_S8
    if (type == "ui8")
        return 6;  // PJRT_U8
    if (type == "i16" || type == "si16")
        return 3;  // PJRT_S16
    if (type == "ui16")
        return 7;  // PJRT_U16
    if (type == "i1")
        return 1;  // PJRT_PRED
    return -1;     // Unknown type - caller must handle
}

MpsExecutable::MpsExecutable(MpsClient* client, const mps::StableHLOModule& module)
    : client_(client),
      name_(module.entry_function.empty() ? "main" : module.entry_function),
      mps_graph_(nullptr),
      mps_executable_(nullptr) {
    CompileFromStableHLO(module);
}

void MpsExecutable::CompileFromStableHLO(const mps::StableHLOModule& module) {
    // Find the entry function (usually "main")
    const mps::StableHLOFunction* entry_func = nullptr;
    for (const auto& func : module.functions) {
        if (func.name == "main" || func.name == module.entry_function) {
            entry_func = &func;
            break;
        }
    }

    if (!entry_func) {
        valid_ = false;
        return;
    }

    // Convert StableHLO function to HloComputation for now
    computation_.name = entry_func->name;

    // Convert argument types to parameters
    for (size_t i = 0; i < entry_func->arg_types.size(); i++) {
        std::string param_name = "%arg" + std::to_string(i);
        const auto& arg_type = entry_func->arg_types[i];
        computation_.parameters.push_back({param_name, arg_type.shape});
    }

    // Process operations with inlining support
    // Track name mappings for call inlining (caller name -> callee name)
    std::unordered_map<std::string, std::string> name_mapping;
    int op_counter = 0;

    // Lambda to process ops from a function (supports recursive inlining)
    // Returns a vector of return value names (for multi-result functions)
    std::function<std::vector<std::string>(const mps::StableHLOFunction*,
                                           const std::vector<std::string>&)>
        processFunction;
    processFunction = [&](const mps::StableHLOFunction* func,
                          const std::vector<std::string>& arg_names) -> std::vector<std::string> {
        // Map function arguments to provided names
        for (size_t i = 0; i < func->arg_types.size() && i < arg_names.size(); i++) {
            name_mapping["%arg" + std::to_string(i)] = arg_names[i];
        }

        // Track return values for multi-result functions
        std::vector<std::string> return_values;

        for (const auto& shlo_op : func->ops) {
            // Handle return ops - track all return values
            if (shlo_op.kind == mps::OpKind::Return) {
                for (const auto& operand : shlo_op.operands) {
                    std::string operand_name = operand.name;
                    if (name_mapping.count(operand_name)) {
                        operand_name = name_mapping[operand_name];
                    }
                    return_values.push_back(operand_name);
                }
                // Update root_name to the last return value for single-result functions
                if (!return_values.empty()) {
                    computation_.root_name = return_values.back();
                }
                continue;
            }

            // Handle call ops by inlining the called function
            if (shlo_op.kind == mps::OpKind::Call) {
                // Look up the called function by name
                const mps::StableHLOFunction* callee = nullptr;
                for (const auto& f : module.functions) {
                    if (f.name == shlo_op.call_target) {
                        callee = &f;
                        break;
                    }
                }
                if (!callee) {
                    error_ = "Call to unknown function: " + shlo_op.call_target;
                    valid_ = false;
                    return {};
                }

                // Save the current name_mapping state before inlining
                // The callee's operations use names like %0, %1, etc. which may conflict
                // with the caller's names. We need to restore after inlining.
                std::unordered_map<std::string, std::string> saved_mapping;
                for (const auto& op : callee->ops) {
                    if (!op.name.empty() && name_mapping.count(op.name)) {
                        saved_mapping[op.name] = name_mapping[op.name];
                    }
                    // Also save multi-result names like %N.0, %N.1
                    for (int i = 0; i < 10; i++) {  // Reasonable limit for multi-result
                        std::string multiName = op.name + "." + std::to_string(i);
                        if (name_mapping.count(multiName)) {
                            saved_mapping[multiName] = name_mapping[multiName];
                        }
                    }
                }
                // Also save %argN mappings that will be overwritten
                for (size_t i = 0; i < callee->arg_types.size(); i++) {
                    std::string arg_name = "%arg" + std::to_string(i);
                    if (name_mapping.count(arg_name)) {
                        saved_mapping[arg_name] = name_mapping[arg_name];
                    }
                }

                // Map caller's operands to callee's arguments
                std::vector<std::string> callee_args;
                for (const auto& operand : shlo_op.operands) {
                    std::string operand_name = operand.name;
                    if (name_mapping.count(operand_name)) {
                        operand_name = name_mapping[operand_name];
                    }
                    callee_args.push_back(operand_name);
                }

                // Process the called function and get its return values
                std::vector<std::string> callee_returns = processFunction(callee, callee_args);

                // Restore saved mappings (remove callee's mappings that weren't there before)
                for (const auto& op : callee->ops) {
                    if (!op.name.empty()) {
                        if (saved_mapping.count(op.name)) {
                            name_mapping[op.name] = saved_mapping[op.name];
                        } else {
                            name_mapping.erase(op.name);
                        }
                        // Also restore multi-result names
                        for (int i = 0; i < 10; i++) {
                            std::string multiName = op.name + "." + std::to_string(i);
                            if (saved_mapping.count(multiName)) {
                                name_mapping[multiName] = saved_mapping[multiName];
                            } else {
                                name_mapping.erase(multiName);
                            }
                        }
                    }
                }
                for (size_t i = 0; i < callee->arg_types.size(); i++) {
                    std::string arg_name = "%arg" + std::to_string(i);
                    if (saved_mapping.count(arg_name)) {
                        name_mapping[arg_name] = saved_mapping[arg_name];
                    } else {
                        name_mapping.erase(arg_name);
                    }
                }

                // Map call results to the callee's return values
                // For single-result calls: %N -> first return value
                // For multi-result calls: %N.0, %N.1, etc. -> corresponding return values
                if (callee_returns.size() == 1) {
                    name_mapping[shlo_op.name] = callee_returns[0];
                } else {
                    for (size_t i = 0; i < callee_returns.size(); i++) {
                        std::string resultName = shlo_op.name + "." + std::to_string(i);
                        name_mapping[resultName] = callee_returns[i];
                    }
                }
                continue;
            }

            HloOp op;
            // Generate unique output name for this op
            op.output = "%op" + std::to_string(op_counter++);
            op.dtype = StablehloTypeToDtype(shlo_op.result_type.element_type);
            if (op.dtype < 0) {
                error_ = "Unsupported element type: " + shlo_op.result_type.element_type;
                valid_ = false;
                return {};
            }
            op.shape = shlo_op.result_type.shape;

            // Map original name to new name
            name_mapping[shlo_op.name] = op.output;

            // Map StableHLO op kind to HLO op name
            switch (shlo_op.kind) {
                case mps::OpKind::Add:
                    op.name = "add";
                    break;
                case mps::OpKind::Multiply:
                    op.name = "multiply";
                    break;
                case mps::OpKind::Subtract:
                    op.name = "subtract";
                    break;
                case mps::OpKind::Divide:
                    op.name = "divide";
                    break;
                case mps::OpKind::Maximum:
                    op.name = "maximum";
                    break;
                case mps::OpKind::Minimum:
                    op.name = "minimum";
                    break;
                case mps::OpKind::Tanh:
                    op.name = "tanh";
                    break;
                case mps::OpKind::Exp:
                    op.name = "exp";
                    break;
                case mps::OpKind::Log:
                    op.name = "log";
                    break;
                case mps::OpKind::Negate:
                    op.name = "negate";
                    break;
                case mps::OpKind::Dot:
                case mps::OpKind::DotGeneral:
                    op.name = "dot";
                    break;
                case mps::OpKind::Reshape:
                    op.name = "reshape";
                    break;
                case mps::OpKind::Transpose:
                    op.name = "transpose";
                    break;
                case mps::OpKind::Convert:
                    op.name = "convert";
                    break;
                case mps::OpKind::BroadcastInDim:
                    op.name = "broadcast_in_dim";
                    op.broadcast_dims = shlo_op.broadcast_dimensions;
                    break;
                case mps::OpKind::Broadcast:
                    op.name = "broadcast";
                    break;
                case mps::OpKind::Abs:
                    op.name = "abs";
                    break;
                case mps::OpKind::Sqrt:
                    op.name = "sqrt";
                    break;
                case mps::OpKind::LogPlusOne:
                    op.name = "log_plus_one";
                    break;
                case mps::OpKind::Compare:
                    op.name = "compare";
                    op.compare_direction = shlo_op.compare_direction;
                    break;
                case mps::OpKind::Select:
                    op.name = "select";
                    break;
                case mps::OpKind::Constant:
                    op.name = "constant";
                    op.constant_data = shlo_op.constant_data;
                    op.constant_raw = shlo_op.constant_raw;
                    op.constant_scalar = shlo_op.constant_scalar;
                    op.constant_scalar_raw = shlo_op.constant_scalar_raw;
                    op.is_scalar_constant = shlo_op.is_scalar_constant;
                    op.uses_raw_data = shlo_op.uses_raw_data;
                    break;
                // Bitwise operations (needed for RNG)
                case mps::OpKind::And:
                    op.name = "and";
                    break;
                case mps::OpKind::Or:
                    op.name = "or";
                    break;
                case mps::OpKind::Xor:
                    op.name = "xor";
                    break;
                case mps::OpKind::ShiftRightLogical:
                    op.name = "shift_right_logical";
                    break;
                case mps::OpKind::ShiftLeft:
                    op.name = "shift_left";
                    break;
                // Other operations
                case mps::OpKind::Concatenate:
                    op.name = "concatenate";
                    op.concatenate_dim = shlo_op.concatenate_dimension;
                    break;
                case mps::OpKind::Slice:
                    op.name = "slice";
                    op.slice_starts = shlo_op.slice_starts;
                    op.slice_limits = shlo_op.slice_limits;
                    op.slice_strides = shlo_op.slice_strides;
                    break;
                case mps::OpKind::DynamicSlice:
                    op.name = "dynamic_slice";
                    op.slice_sizes = shlo_op.slice_sizes;
                    break;
                case mps::OpKind::Iota:
                    op.name = "iota";
                    op.iota_dim = shlo_op.iota_dimension;
                    break;
                case mps::OpKind::BitcastConvert:
                    op.name = "bitcast_convert";
                    break;
                case mps::OpKind::CustomCall:
                    op.name = "custom_call";
                    op.custom_call_target = shlo_op.custom_call_target;
                    break;
                case mps::OpKind::Unknown:
                    // Unknown ops are not allowed - fail at compile time
                    error_ = "Unsupported StableHLO operation encountered during compilation. "
                             "The MPS backend does not yet support this operation. "
                             "Check the JAX MPS supported operations list.";
                    valid_ = false;
                    return {};
                default:
                    // This should not happen if OpKind enum is kept in sync
                    error_ = "Internal error: unhandled OpKind in compilation";
                    valid_ = false;
                    return {};
            }

            // Copy operands, applying name mapping
            for (const auto& operand : shlo_op.operands) {
                std::string operand_name = operand.name;
                if (name_mapping.count(operand_name)) {
                    operand_name = name_mapping[operand_name];
                }
                op.inputs.push_back(operand_name);
            }

            computation_.ops.push_back(op);
            computation_.root_name = op.output;
        }
        return return_values;
    };

    // Process the entry function with its arguments
    std::vector<std::string> entry_args;
    for (size_t i = 0; i < entry_func->arg_types.size(); i++) {
        entry_args.push_back("%arg" + std::to_string(i));
    }
    std::vector<std::string> entry_returns = processFunction(entry_func, entry_args);

    // Store return values for multi-output functions
    computation_.return_values = entry_returns;

    // Set the number of outputs based on result types
    num_outputs_ = entry_func->result_types.size();
    if (num_outputs_ == 0)
        num_outputs_ = 1;

    valid_ = true;
}

MpsExecutable::~MpsExecutable() {
    if (mps_executable_) {
        CFRelease((__bridge CFTypeRef)mps_executable_);
    }
    if (mps_graph_) {
        CFRelease((__bridge CFTypeRef)mps_graph_);
    }
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

        // Map from HLO names to MPSGraphTensor
        NSMutableDictionary<NSString*, MPSGraphTensor*>* tensors = [NSMutableDictionary dictionary];

        // Create placeholder tensors for parameters
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
            [NSMutableDictionary dictionary];

        // Validate input count
        if (inputs.size() < computation_.parameters.size()) {
            return ExecutionResult::Error("Input count mismatch: expected " +
                                          std::to_string(computation_.parameters.size()) +
                                          " inputs, got " + std::to_string(inputs.size()));
        }

        for (size_t i = 0; i < computation_.parameters.size() && i < inputs.size(); i++) {
            const auto& param = computation_.parameters[i];
            MpsBuffer* input = inputs[i];

            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }

            // Create shape array
            NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
            for (int64_t dim : param.second) {
                [shape addObject:@(dim)];
            }

            MPSDataType mps_dtype = PjrtDtypeToMps(input->dtype());
            if (mps_dtype == MPSDataTypeInvalid) {
                return ExecutionResult::Error("Unsupported data type (PJRT dtype " +
                                              std::to_string(input->dtype()) +
                                              ") for input at index " + std::to_string(i));
            }

            // Create placeholder
            MPSGraphTensor* placeholder =
                [graph placeholderWithShape:shape
                                   dataType:mps_dtype
                                       name:[NSString stringWithUTF8String:param.first.c_str()]];

            tensors[[NSString stringWithUTF8String:param.first.c_str()]] = placeholder;

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

        // Build operations with Objective-C exception handling
        MPSGraphTensor* result_tensor = nil;
        std::string op_error;

        std::string current_op_name;
        @try {
            for (const auto& op : computation_.ops) {
                current_op_name = op.name;
                NSMutableArray<NSNumber*>* output_shape = [NSMutableArray array];
                for (int64_t dim : op.shape) {
                    [output_shape addObject:@(dim)];
                }

                // Look up handler in registry
                OpHandler handler = OpRegistry::Find(op.name);
                if (!handler) {
                    // Get list of supported ops for error message
                    std::string supported = OpRegistry::ListRegistered();
                    op_error = "Unsupported operation: '" + op.name +
                               "'. The MPS backend does not have a handler for this operation. "
                               "Supported operations: " +
                               supported;
                    break;
                }

                MPSGraphTensor* out = handler(graph, tensors, op, output_shape);
                if (!out) {
                    // List what tensors we have for debugging
                    std::string available_tensors;
                    for (NSString* key in tensors) {
                        if (!available_tensors.empty())
                            available_tensors += ", ";
                        available_tensors += [key UTF8String];
                    }
                    std::string inputs_str;
                    for (const auto& inp : op.inputs) {
                        if (!inputs_str.empty())
                            inputs_str += ", ";
                        inputs_str += inp;
                    }
                    op_error = "Operation '" + op.name + "' handler returned null. Inputs: [" +
                               inputs_str + "]. Available tensors: [" + available_tensors + "]";
                    break;
                }
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
        } @catch (NSException* exception) {
            return ExecutionResult::Error("MPS operation '" + current_op_name +
                                          "' failed with Objective-C exception: " +
                                          std::string([[exception name] UTF8String]) + " - " +
                                          std::string([[exception reason] UTF8String]));
        }

        if (!op_error.empty()) {
            return ExecutionResult::Error(op_error);
        }

        // Handle identity functions - computation with no ops that just passes through input
        // This is a valid JAX pattern, not an error
        if (!result_tensor && computation_.ops.empty() && !inputs.empty()) {
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
        std::vector<std::string> return_names;

        if (!computation_.return_values.empty()) {
            // Multi-output function: use all return values
            for (const auto& ret_name : computation_.return_values) {
                MPSGraphTensor* ret_tensor =
                    tensors[[NSString stringWithUTF8String:ret_name.c_str()]];
                if (!ret_tensor) {
                    return ExecutionResult::Error("Return value '" + ret_name +
                                                  "' not found in tensors");
                }
                [target_tensors addObject:ret_tensor];
                return_names.push_back(ret_name);
            }
        } else if (result_tensor) {
            // Single-output: use the last tensor
            [target_tensors addObject:result_tensor];
            return_names.push_back(computation_.root_name);
        } else {
            return ExecutionResult::Error(
                "No result tensor produced after executing " +
                std::to_string(computation_.ops.size()) +
                " operations. "
                "This indicates an internal error in the MPS graph construction.");
        }

        // Execute graph with Objective-C exception handling
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

            // Find the dtype for this output by looking up the op that produced it
            int output_dtype = -1;
            const std::string& output_name = return_names[i];
            for (const auto& op : computation_.ops) {
                if (op.output == output_name) {
                    output_dtype = op.dtype;
                    break;
                }
            }
            if (output_dtype < 0) {
                // Fallback to last op's dtype
                output_dtype = computation_.ops.back().dtype;
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
