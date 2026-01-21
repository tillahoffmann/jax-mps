#import "pjrt_plugin/mps_executable.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/stablehlo_parser.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#import <regex>
#import <sstream>

namespace jax_mps {

// Map PJRT dtype to MPSDataType
static MPSDataType PjrtDtypeToMps(int dtype) {
    // Using same enum values as in mps_buffer.mm
    switch (dtype) {
        case 11: return MPSDataTypeFloat32;  // PJRT_F32
        case 10: return MPSDataTypeFloat16;  // PJRT_F16
        case 16: return MPSDataTypeBFloat16; // PJRT_BF16
        case 4:  return MPSDataTypeInt32;    // PJRT_S32
        case 5:  return MPSDataTypeInt64;    // PJRT_S64
        case 8:  return MPSDataTypeUInt32;   // PJRT_U32
        case 1:  return MPSDataTypeBool;     // PJRT_PRED
        default: return MPSDataTypeFloat32;
    }
}

// Parse shape string like "f32[2,3]" into dtype and dimensions
static bool ParseShapeString(const std::string& shape_str, int& dtype, std::vector<int64_t>& dims) {
    dims.clear();

    // Extract type prefix
    std::string type_str;
    size_t bracket_pos = shape_str.find('[');
    if (bracket_pos != std::string::npos) {
        type_str = shape_str.substr(0, bracket_pos);
    } else {
        type_str = shape_str;
    }

    // Map type string to PJRT dtype
    if (type_str == "f32") dtype = 11;
    else if (type_str == "f16") dtype = 10;
    else if (type_str == "bf16") dtype = 16;
    else if (type_str == "s32" || type_str == "i32") dtype = 4;
    else if (type_str == "s64" || type_str == "i64") dtype = 5;
    else if (type_str == "u32") dtype = 8;
    else if (type_str == "pred" || type_str == "bool") dtype = 1;
    else dtype = 11;  // Default to f32

    // Extract dimensions
    if (bracket_pos != std::string::npos) {
        size_t end_bracket = shape_str.find(']', bracket_pos);
        if (end_bracket != std::string::npos) {
            std::string dims_str = shape_str.substr(bracket_pos + 1, end_bracket - bracket_pos - 1);
            if (!dims_str.empty()) {
                std::stringstream ss(dims_str);
                std::string dim;
                while (std::getline(ss, dim, ',')) {
                    dims.push_back(std::stoll(dim));
                }
            }
        }
    }

    // Scalar if no dimensions
    if (dims.empty()) {
        dims.push_back(1);
    }

    return true;
}

// Simple HLO text parser
// Example HLO:
//   HloModule test
//   ENTRY main {
//     %p0 = f32[2,3] parameter(0)
//     %p1 = f32[2,3] parameter(1)
//     ROOT %add = f32[2,3] add(%p0, %p1)
//   }
HloComputation ParseHloText(const std::string& hlo_text) {
    HloComputation comp;
    comp.name = "main";

    std::istringstream stream(hlo_text);
    std::string line;

    std::regex param_regex(R"(%(\w+)\s*=\s*(\S+)\s+parameter\((\d+)\))");
    std::regex op_regex(R"(%(\w+)\s*=\s*(\S+)\s+(\w+)\(([^)]*)\))");
    std::regex root_regex(R"(ROOT\s+%(\w+)\s*=\s*(\S+)\s+(\w+)\(([^)]*)\))");

    while (std::getline(stream, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        std::smatch match;

        // Check for parameter
        if (std::regex_search(line, match, param_regex)) {
            std::string name = "%" + match[1].str();
            std::string shape_str = match[2].str();
            int dtype;
            std::vector<int64_t> dims;
            ParseShapeString(shape_str, dtype, dims);
            comp.parameters.push_back({name, dims});
            continue;
        }

        // Check for ROOT operation
        if (std::regex_search(line, match, root_regex)) {
            HloOp op;
            op.output = "%" + match[1].str();
            std::string shape_str = match[2].str();
            op.name = match[3].str();
            std::string args_str = match[4].str();

            ParseShapeString(shape_str, op.dtype, op.shape);

            // Parse arguments
            std::stringstream args_stream(args_str);
            std::string arg;
            while (std::getline(args_stream, arg, ',')) {
                // Trim whitespace
                size_t s = arg.find_first_not_of(" \t");
                size_t e = arg.find_last_not_of(" \t");
                if (s != std::string::npos) {
                    op.inputs.push_back(arg.substr(s, e - s + 1));
                }
            }

            comp.ops.push_back(op);
            comp.root_name = op.output;
            continue;
        }

        // Check for regular operation
        if (std::regex_search(line, match, op_regex)) {
            HloOp op;
            op.output = "%" + match[1].str();
            std::string shape_str = match[2].str();
            op.name = match[3].str();
            std::string args_str = match[4].str();

            ParseShapeString(shape_str, op.dtype, op.shape);

            // Parse arguments
            std::stringstream args_stream(args_str);
            std::string arg;
            while (std::getline(args_stream, arg, ',')) {
                size_t s = arg.find_first_not_of(" \t");
                size_t e = arg.find_last_not_of(" \t");
                if (s != std::string::npos) {
                    op.inputs.push_back(arg.substr(s, e - s + 1));
                }
            }

            comp.ops.push_back(op);
        }
    }

    return comp;
}

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

MpsExecutable::MpsExecutable(MpsClient* client, const HloComputation& computation)
    : client_(client)
    , name_(computation.name)
    , mps_graph_(nullptr)
    , mps_executable_(nullptr) {
    CompileFromHLO(computation);
}

MpsExecutable::MpsExecutable(MpsClient* client, const mps::StableHLOModule& module)
    : client_(client)
    , name_(module.entry_function.empty() ? "main" : module.entry_function)
    , mps_graph_(nullptr)
    , mps_executable_(nullptr) {
    CompileFromStableHLO(module);
}

void MpsExecutable::CompileFromHLO(const HloComputation& computation) {
    computation_ = computation;
    valid_ = !computation_.ops.empty();
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
        NSLog(@"No entry function found in StableHLO module");
        valid_ = false;
        return;
    }

    // Convert StableHLO function to HloComputation for now
    // (This allows reusing existing execution code)
    computation_.name = entry_func->name;

    NSLog(@"Entry function: %s, %zu args, %zu results, %zu ops",
          entry_func->name.c_str(),
          entry_func->arg_types.size(),
          entry_func->result_types.size(),
          entry_func->ops.size());

    // Convert argument types to parameters
    for (size_t i = 0; i < entry_func->arg_types.size(); i++) {
        std::string param_name = "%arg" + std::to_string(i);
        const auto& arg_type = entry_func->arg_types[i];
        NSLog(@"  Arg %zu: shape size=%zu, element_type=%s",
              i, arg_type.shape.size(), arg_type.element_type.c_str());
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
            default:
                NSLog(@"Unsupported StableHLO op kind: %d", (int)shlo_op.kind);
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
    NSLog(@"Compiled StableHLO to %zu ops, %d outputs", computation_.ops.size(), num_outputs_);
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

    NSLog(@"Execute: %zu inputs, %zu parameters, %zu ops",
          inputs.size(), computation_.parameters.size(), computation_.ops.size());

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

            if (op.name == "add") {
                MPSGraphTensor* lhs = tensors[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
                MPSGraphTensor* rhs = tensors[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
                MPSGraphTensor* out = [graph additionWithPrimaryTensor:lhs
                                                       secondaryTensor:rhs
                                                                  name:nil];
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
            else if (op.name == "dot" || op.name == "dot_general") {
                MPSGraphTensor* lhs = tensors[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
                MPSGraphTensor* rhs = tensors[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
                MPSGraphTensor* out = [graph matrixMultiplicationWithPrimaryTensor:lhs
                                                                   secondaryTensor:rhs
                                                                              name:nil];
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
            else if (op.name == "tanh") {
                MPSGraphTensor* input = tensors[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
                MPSGraphTensor* out = [graph tanhWithTensor:input name:nil];
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
            else if (op.name == "multiply") {
                MPSGraphTensor* lhs = tensors[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
                MPSGraphTensor* rhs = tensors[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
                MPSGraphTensor* out = [graph multiplicationWithPrimaryTensor:lhs
                                                             secondaryTensor:rhs
                                                                        name:nil];
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
            else if (op.name == "subtract") {
                MPSGraphTensor* lhs = tensors[[NSString stringWithUTF8String:op.inputs[0].c_str()]];
                MPSGraphTensor* rhs = tensors[[NSString stringWithUTF8String:op.inputs[1].c_str()]];
                MPSGraphTensor* out = [graph subtractionWithPrimaryTensor:lhs
                                                          secondaryTensor:rhs
                                                                     name:nil];
                tensors[[NSString stringWithUTF8String:op.output.c_str()]] = out;
                result_tensor = out;
            }
            else {
                NSLog(@"Unsupported operation: %s", op.name.c_str());
            }
        }

        // Handle identity functions (no ops, just return input)
        if (!result_tensor && computation_.ops.empty() && !inputs.empty()) {
            NSLog(@"Identity function - returning input as output");
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
            NSLog(@"No result tensor produced");
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
