#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/stablehlo_parser.h"
#import "pjrt_plugin/ops/registry.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

namespace jax_mps {

// Map StableHLO element type string to PJRT dtype
static int StablehloTypeToDtype(const std::string& type) {
    if (type == "f32") return 11;  // PJRT_F32
    if (type == "f16") return 10;  // PJRT_F16
    if (type == "bf16") return 16; // PJRT_BF16
    if (type == "f64") return 12;  // PJRT_F64
    if (type == "i32" || type == "si32") return 4;  // PJRT_S32
    if (type == "i64" || type == "si64") return 5;  // PJRT_S64
    if (type == "ui32") return 8;  // PJRT_U32
    if (type == "i1") return 1;    // PJRT_PRED
    return 11;  // Default to f32
}

MpsExecutable::MpsExecutable(MpsClient* client, const mps::StableHLOModule& module)
    : client_(client)
    , name_(module.entry_function.empty() ? "main" : module.entry_function)
    , mps_graph_(nullptr)
    , mps_executable_(nullptr) {
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
        // NSLog(@"No entry function found in StableHLO module");
        valid_ = false;
        return;
    }

    // Convert StableHLO function to HloComputation for now
    // (This allows reusing existing execution code)
    computation_.name = entry_func->name;

    // NSLog(@"Entry function: %s, %zu args, %zu results, %zu ops",
    //       entry_func->name.c_str(),
    //       entry_func->arg_types.size(),
    //       entry_func->result_types.size(),
    //       entry_func->ops.size());

    // Convert argument types to parameters
    for (size_t i = 0; i < entry_func->arg_types.size(); i++) {
        std::string param_name = "%arg" + std::to_string(i);
        const auto& arg_type = entry_func->arg_types[i];
        // NSLog(@"  Arg %zu: shape size=%zu, element_type=%s",
        //       i, arg_type.shape.size(), arg_type.element_type.c_str());
        computation_.parameters.push_back({param_name, arg_type.shape});
    }

    // Convert operations
    for (const auto& shlo_op : entry_func->ops) {
        // Skip return ops
        if (shlo_op.kind == mps::OpKind::Return) {
            continue;
        }

        // Skip call ops for now (inline the called function)
        if (shlo_op.kind == mps::OpKind::Call) {
            // For now, pass through the operands
            continue;
        }

        HloOp op;
        op.output = shlo_op.name;
        op.dtype = StablehloTypeToDtype(shlo_op.result_type.element_type);
        op.shape = shlo_op.result_type.shape;

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
                // Store broadcast dimensions
                op.broadcast_dims = shlo_op.broadcast_dimensions;
                break;
            case mps::OpKind::Broadcast:
                op.name = "broadcast";
                break;
            case mps::OpKind::Abs:
                op.name = "abs";
                break;
            case mps::OpKind::Constant:
                op.name = "constant";
                break;
            default:
                // NSLog(@"Unsupported StableHLO op kind: %d", (int)shlo_op.kind);
                op.name = "unknown";
                break;
        }

        // Copy operands
        for (const auto& operand : shlo_op.operands) {
            op.inputs.push_back(operand.name);
        }

        computation_.ops.push_back(op);

        // Track the last non-return op as root
        computation_.root_name = op.output;
    }

    // Set the number of outputs based on result types
    num_outputs_ = entry_func->result_types.size();
    if (num_outputs_ == 0) num_outputs_ = 1;

    // Identity functions (just returning inputs) are valid even with 0 ops
    valid_ = true;
    // NSLog(@"Compiled StableHLO to %zu ops, %d outputs", computation_.ops.size(), num_outputs_);
}

MpsExecutable::~MpsExecutable() {
    if (mps_executable_) {
        CFRelease((__bridge CFTypeRef)mps_executable_);
    }
    if (mps_graph_) {
        CFRelease((__bridge CFTypeRef)mps_graph_);
    }
}

std::vector<std::unique_ptr<MpsBuffer>> MpsExecutable::Execute(
    const std::vector<MpsBuffer*>& inputs,
    MpsDevice* device) {

    // NSLog(@"Execute: %zu inputs, %zu parameters, %zu ops",
    //       inputs.size(), computation_.parameters.size(), computation_.ops.size());

    std::vector<std::unique_ptr<MpsBuffer>> results;

    @autoreleasepool {
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Map from HLO names to MPSGraphTensor
        NSMutableDictionary<NSString*, MPSGraphTensor*>* tensors = [NSMutableDictionary dictionary];

        // Create placeholder tensors for parameters
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)client_->metal_device();

        for (size_t i = 0; i < computation_.parameters.size() && i < inputs.size(); i++) {
            const auto& param = computation_.parameters[i];
            MpsBuffer* input = inputs[i];

            // Create shape array
            NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
            for (int64_t dim : param.second) {
                [shape addObject:@(dim)];
            }

            MPSDataType mps_dtype = PjrtDtypeToMps(input->dtype());

            // Create placeholder
            MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                             dataType:mps_dtype
                                                                 name:[NSString stringWithUTF8String:param.first.c_str()]];

            tensors[[NSString stringWithUTF8String:param.first.c_str()]] = placeholder;

            // Create tensor data from input buffer
            id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)input->metal_buffer();
            MPSGraphTensorData* tensor_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buffer
                                                                                      shape:shape
                                                                                   dataType:mps_dtype];
            feeds[placeholder] = tensor_data;
        }

        // Build operations
        MPSGraphTensor* result_tensor = nil;

        for (const auto& op : computation_.ops) {
            NSMutableArray<NSNumber*>* output_shape = [NSMutableArray array];
            for (int64_t dim : op.shape) {
                [output_shape addObject:@(dim)];
            }

            // Look up handler in registry
            OpHandler handler = OpRegistry::Find(op.name);
            if (handler) {
                MPSGraphTensor* out = handler(graph, tensors, op, output_shape);
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            } else {
                // Unsupported op: pass through first input as fallback
                if (!op.inputs.empty()) {
                    MPSGraphTensor* input = GetTensor(tensors, op.inputs[0]);
                    if (input) {
                        tensors[[NSString stringWithUTF8String:op.output.c_str()]] = input;
                        result_tensor = input;
                    }
                }
            }
        }

        // Handle identity functions (no ops, or failed to produce result)
        if (!result_tensor && !inputs.empty()) {
            // NSLog(@"Identity function - returning input as output");
            // For identity, just copy the input buffer
            MpsBuffer* input = inputs[0];
            const auto& dims = input->dimensions();

            size_t byte_size = input->byte_size();
            id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input->metal_buffer();

            // Create a new buffer with copied data
            id<MTLBuffer> output_buffer = [mtl_device newBufferWithBytes:input_buffer.contents
                                                                  length:byte_size
                                                                 options:MTLResourceStorageModeShared];

            auto buffer = std::make_unique<MpsBuffer>(
                device,
                (__bridge void*)output_buffer,
                input->dtype(),
                dims
            );
            results.push_back(std::move(buffer));
            return results;
        }

        if (!result_tensor) {
            // NSLog(@"No result tensor produced and no inputs available");
            return results;
        }

        // Execute graph
        id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* result_dict =
            [graph runWithMTLCommandQueue:commandQueue
                                    feeds:feeds
                            targetTensors:@[result_tensor]
                         targetOperations:nil];

        MPSGraphTensorData* result_data = result_dict[result_tensor];
        if (result_data) {
            // Get result shape
            std::vector<int64_t> result_shape;
            for (NSNumber* dim in result_data.shape) {
                result_shape.push_back([dim longLongValue]);
            }

            // Calculate byte size
            size_t byte_size = 1;
            for (int64_t dim : result_shape) {
                byte_size *= dim;
            }
            byte_size *= DtypeByteSize(computation_.ops.back().dtype);

            // Create output buffer with shared storage
            id<MTLBuffer> output_buffer = [mtl_device newBufferWithLength:byte_size
                                                                  options:MTLResourceStorageModeShared];

            // Copy result data using MPSNDArray
            // Get the underlying MPSNDArray and read its data
            MPSNDArray* ndarray = [result_data mpsndarray];
            if (ndarray) {
                // Read data from the ndarray into our buffer
                [ndarray readBytes:output_buffer.contents strideBytes:nil];
            }

            auto buffer = std::make_unique<MpsBuffer>(
                device,
                (__bridge void*)output_buffer,
                computation_.ops.back().dtype,
                result_shape
            );
            results.push_back(std::move(buffer));
        }
    }

    return results;
}

}  // namespace jax_mps
