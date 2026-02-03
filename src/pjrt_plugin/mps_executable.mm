#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <unordered_map>
#include <unordered_set>

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#import "pjrt_plugin/issue_url.h"
#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/mps_client.h"
#import "pjrt_plugin/mps_device.h"
#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

// ---------------------------------------------------------------------------
// ExecutionPlan types (full definition, opaque in header)
// ---------------------------------------------------------------------------

using SlotId = int;

struct SlotInfo {
    NSArray<NSNumber*>* shape;
    MPSDataType dtype;
    size_t byte_size;
};

struct GraphStep {
    MPSGraph* graph;
    std::vector<std::pair<SlotId, MPSGraphTensor*>> feeds;    // slot -> placeholder
    std::vector<std::pair<SlotId, MPSGraphTensor*>> targets;  // slot -> target tensor
};

struct NativeStep {
    NativeOpHandler handler;
    mlir::Operation* op;
    std::vector<SlotId> input_slots;
    SlotId output_slot;
};

struct Step {
    enum Kind { GRAPH, NATIVE } kind;
    size_t index;  // Index into graph_steps or native_steps
};

struct ExecutionPlan {
    std::vector<Step> steps;
    std::vector<GraphStep> graph_steps;
    std::vector<NativeStep> native_steps;
    std::vector<SlotId> input_slots;   // function arg slots
    std::vector<SlotId> output_slots;  // return value slots
    std::vector<mlir::Type> return_types;
    std::vector<SlotInfo> slots;
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

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
    // plan_ is destroyed by unique_ptr; ARC releases the ObjC objects inside.
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// Compute byte size from a ranked tensor type.
static size_t ByteSizeFromType(mlir::Type type) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType)
        return 0;
    int pjrt_dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    size_t total = elem_size;
    for (int64_t dim : tensorType.getShape()) {
        total *= dim;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Graph-segment helpers (unchanged from original processCallOp / processOperations)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BuildExecutionPlan
// ---------------------------------------------------------------------------

bool MpsExecutable::BuildExecutionPlan() {
    if (plan_built_)
        return true;

    @autoreleasepool {
        if (!client_) {
            error_ = "No MPS client available";
            return false;
        }

        // Manually inline func.call ops whose callees contain native ops.
        // JAX wraps ops like cholesky in private helper functions called via
        // func.call – the segmentation below only walks the entry block, so
        // native ops inside callees would be missed.  We splice the callee's
        // operations directly into the entry block, avoiding the MLIR inliner
        // pass (which requires DialectInlinerInterface on every dialect).
        {
            bool changed = true;
            while (changed) {
                changed = false;
                mlir::Block& block = entry_func_.front();
                for (auto it = block.begin(), end = block.end(); it != end; ++it) {
                    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(&*it);
                    if (!callOp)
                        continue;

                    auto callee = module_->lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                    if (!callee || callee.empty())
                        continue;

                    // Only inline callees that contain native ops.
                    bool has_native = false;
                    callee.walk([&](mlir::Operation* inner) {
                        if (NativeOpRegistry::Find(inner->getName().getStringRef().str()))
                            has_native = true;
                    });
                    if (!has_native)
                        continue;

                    // Clone the callee's body into the caller.
                    mlir::Block& calleeBlock = callee.front();
                    mlir::IRMapping mapping;
                    for (unsigned i = 0; i < calleeBlock.getNumArguments(); i++) {
                        mapping.map(calleeBlock.getArgument(i), callOp.getOperand(i));
                    }

                    mlir::OpBuilder builder(callOp);
                    for (auto& op : calleeBlock) {
                        if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(&op)) {
                            // Wire callee return values to call results.
                            for (unsigned i = 0; i < callOp.getNumResults(); i++) {
                                callOp.getResult(i).replaceAllUsesWith(
                                    mapping.lookup(retOp.getOperand(i)));
                            }
                            continue;
                        }
                        builder.clone(op, mapping);
                    }
                    callOp.erase();
                    changed = true;
                    break;  // Restart – block iterators invalidated.
                }
            }
        }

        auto plan = std::make_unique<ExecutionPlan>();

        mlir::Block& block = entry_func_.front();

        // ------------------------------------------------------------------
        // Pass 1 – segment the ops
        // ------------------------------------------------------------------
        struct SegmentInfo {
            bool is_native;
            std::vector<mlir::Operation*> ops;
        };

        std::vector<SegmentInfo> segments;
        SegmentInfo current_segment{false, {}};
        mlir::func::ReturnOp returnOp = nullptr;

        for (mlir::Operation& operation : block) {
            mlir::Operation* op = &operation;

            if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
                returnOp = ret;
                continue;
            }

            std::string op_name = op->getName().getStringRef().str();
            NativeOpHandler native_handler = NativeOpRegistry::Find(op_name);

            if (native_handler) {
                // Close current graph segment
                if (!current_segment.ops.empty()) {
                    segments.push_back(std::move(current_segment));
                    current_segment = {false, {}};
                }
                // Native op gets its own segment
                segments.push_back({true, {op}});
            } else {
                current_segment.ops.push_back(op);
            }
        }
        if (!current_segment.ops.empty()) {
            segments.push_back(std::move(current_segment));
        }

        // ------------------------------------------------------------------
        // Pass 1b – determine which values need materialised slots
        // ------------------------------------------------------------------
        // Build op → segment map
        std::unordered_map<mlir::Operation*, int> op_to_seg;
        for (int i = 0; i < (int)segments.size(); i++) {
            for (auto* op : segments[i].ops) {
                op_to_seg[op] = i;
            }
        }

        // Slot allocation helper
        std::unordered_map<void*, SlotId> value_to_slot;
        auto getOrCreateSlot = [&](mlir::Value value) -> SlotId {
            void* key = value.getAsOpaquePointer();
            auto it = value_to_slot.find(key);
            if (it != value_to_slot.end())
                return it->second;

            SlotId slot = (SlotId)plan->slots.size();
            value_to_slot[key] = slot;

            SlotInfo info;
            info.shape = GetShapeFromType(value.getType());
            auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
            info.dtype =
                tensorType ? MlirTypeToMps(tensorType.getElementType()) : MPSDataTypeFloat32;
            info.byte_size = ByteSizeFromType(value.getType());
            plan->slots.push_back(info);
            return slot;
        };

        // Function arguments always get slots
        for (unsigned i = 0; i < entry_func_.getNumArguments(); i++) {
            auto arg = entry_func_.getArgument(i);
            SlotId slot = getOrCreateSlot(arg);
            plan->input_slots.push_back(slot);
        }

        // Return values always get slots
        if (returnOp) {
            for (mlir::Value operand : returnOp->getOperands()) {
                SlotId slot = getOrCreateSlot(operand);
                plan->output_slots.push_back(slot);
                plan->return_types.push_back(operand.getType());
            }
        }

        // Values that cross segment boundaries need slots
        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            for (auto* op : segments[seg_idx].ops) {
                // Check operands – if defined in a different segment, need a slot
                for (mlir::Value operand : op->getOperands()) {
                    auto* defOp = operand.getDefiningOp();
                    if (!defOp)
                        continue;  // block argument – already has a slot
                    auto it = op_to_seg.find(defOp);
                    if (it == op_to_seg.end() || it->second != seg_idx) {
                        getOrCreateSlot(operand);
                    }
                }
                // Check results – if used in a different segment (or by return)
                for (mlir::Value result : op->getResults()) {
                    for (auto* user : result.getUsers()) {
                        if (mlir::isa<mlir::func::ReturnOp>(user)) {
                            getOrCreateSlot(result);
                            break;
                        }
                        auto it = op_to_seg.find(user);
                        if (it == op_to_seg.end() || it->second != seg_idx) {
                            getOrCreateSlot(result);
                            break;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pass 2 – build graph / native steps
        // ------------------------------------------------------------------
        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            auto& seg = segments[seg_idx];

            if (seg.is_native) {
                // ----- Native step -----
                mlir::Operation* op = seg.ops[0];
                std::string op_name = op->getName().getStringRef().str();
                NativeOpHandler handler = NativeOpRegistry::Find(op_name);

                if (op->getNumResults() != 1) {
                    error_ = "Native op '" + op_name + "' must have exactly one result, got " +
                             std::to_string(op->getNumResults());
                    return false;
                }

                NativeStep ns;
                ns.handler = handler;
                ns.op = op;
                for (mlir::Value operand : op->getOperands()) {
                    ns.input_slots.push_back(value_to_slot.at(operand.getAsOpaquePointer()));
                }
                ns.output_slot = value_to_slot.at(op->getResult(0).getAsOpaquePointer());

                plan->native_steps.push_back(ns);
                plan->steps.push_back({Step::NATIVE, plan->native_steps.size() - 1});
            } else {
                // ----- Graph step -----
                MPSGraph* graph = [[MPSGraph alloc] init];
                if (!graph) {
                    error_ = "Failed to create MPSGraph";
                    return false;
                }
                ValueMap values;
                GraphStep gs;
                gs.graph = graph;

                // Create placeholders for values entering this segment from outside.
                // Values defined within this segment should NOT get placeholders.
                std::unordered_set<mlir::Operation*> seg_ops_set(seg.ops.begin(), seg.ops.end());
                std::unordered_set<void*> created_ph;
                for (auto* op : seg.ops) {
                    for (mlir::Value operand : op->getOperands()) {
                        // Skip values produced within this segment.
                        mlir::Operation* defOp = operand.getDefiningOp();
                        if (defOp && seg_ops_set.count(defOp)) {
                            continue;
                        }

                        void* key = operand.getAsOpaquePointer();
                        auto slot_it = value_to_slot.find(key);
                        if (slot_it != value_to_slot.end() && !created_ph.count(key)) {
                            SlotId slot = slot_it->second;
                            NSArray<NSNumber*>* shape = plan->slots[slot].shape;
                            MPSDataType mps_dtype = plan->slots[slot].dtype;

                            MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                                             dataType:mps_dtype
                                                                                 name:nil];
                            values[key] = placeholder;
                            gs.feeds.push_back({slot, placeholder});
                            created_ph.insert(key);
                        }
                    }
                }

                // Process ops within this segment
                @try {
                    for (auto* op : seg.ops) {
                        std::string op_name = op->getName().getStringRef().str();

                        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
                            ProcessResult callResult =
                                processCallOp(graph, callOp, values, *module_, 0);
                            if (!callResult.ok()) {
                                error_ = callResult.error;
                                return false;
                            }
                            continue;
                        }

                        OpHandler handler = OpRegistry::Find(op_name);
                        if (!handler) {
                            error_ = UnsupportedOpsMessage({op_name}) +
                                     "\n\nSupported operations: " + OpRegistry::ListRegistered();
                            return false;
                        }

                        if (op->getNumResults() > 1) {
                            error_ = "Operation '" + op_name +
                                     "' has multiple results which is not yet supported";
                            return false;
                        }

                        MPSGraphTensor* out = handler(graph, op, values);
                        if (!out) {
                            error_ = "Operation '" + op_name + "' handler returned null";
                            return false;
                        }

                        if (op->getNumResults() > 0) {
                            values[op->getResult(0).getAsOpaquePointer()] = out;
                        }
                    }
                } @catch (NSException* exception) {
                    error_ =
                        "MPS graph build failed: " + std::string([[exception reason] UTF8String]);
                    return false;
                }

                // Identify target values (values leaving this segment)
                for (auto* op : seg.ops) {
                    for (mlir::Value result : op->getResults()) {
                        void* key = result.getAsOpaquePointer();
                        if (value_to_slot.count(key)) {
                            SlotId slot = value_to_slot.at(key);
                            MPSGraphTensor* tensor = values[key];
                            if (tensor) {
                                gs.targets.push_back({slot, tensor});
                            }
                        }
                    }
                }

                plan->graph_steps.push_back(gs);
                plan->steps.push_back({Step::GRAPH, plan->graph_steps.size() - 1});
            }
        }

        plan_ = std::move(plan);
        plan_built_ = true;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

ExecutionResult MpsExecutable::Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device) {
    // Check for compilation errors
    if (!valid_) {
        return ExecutionResult::Error("Cannot execute: compilation failed - " + error_);
    }

    // Build plan on first execution (lazy initialization)
    if (!plan_built_) {
        if (!BuildExecutionPlan()) {
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

        // Input count check
        size_t num_args = plan_->input_slots.size();
        if (inputs.size() < num_args) {
            return ExecutionResult::Error("Input count mismatch: expected " +
                                          std::to_string(num_args) + " inputs, got " +
                                          std::to_string(inputs.size()));
        }

        // Handle identity functions (no steps, but outputs reference inputs)
        if (plan_->steps.empty() && !plan_->output_slots.empty() && !inputs.empty()) {
            ExecutionResult result;
            for (size_t i = 0; i < plan_->output_slots.size(); i++) {
                SlotId slot = plan_->output_slots[i];
                // Find the input index that maps to this slot
                int input_idx = -1;
                for (size_t j = 0; j < plan_->input_slots.size(); j++) {
                    if (plan_->input_slots[j] == slot) {
                        input_idx = (int)j;
                        break;
                    }
                }
                if (input_idx < 0 || !inputs[input_idx]) {
                    return ExecutionResult::Error("Identity function with unmapped output slot");
                }
                MpsBuffer* input = inputs[input_idx];
                size_t byte_size = input->byte_size();
                id<MTLBuffer> src = (__bridge id<MTLBuffer>)input->metal_buffer();
                if (!src) {
                    return ExecutionResult::Error("Identity function input has no Metal buffer");
                }
                id<MTLBuffer> dst = [mtl_device newBufferWithBytes:src.contents
                                                            length:byte_size
                                                           options:MTLResourceStorageModeShared];
                if (!dst) {
                    return ExecutionResult::Error("Failed to allocate buffer for identity output");
                }

                auto tensorType = mlir::cast<mlir::RankedTensorType>(plan_->return_types[i]);
                std::vector<int64_t> dims(tensorType.getShape().begin(),
                                          tensorType.getShape().end());
                int dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
                result.buffers.push_back(
                    std::make_unique<MpsBuffer>(device, (__bridge void*)dst, dtype, dims));
            }
            return result;
        }

        // Function with no outputs
        if (plan_->output_slots.empty()) {
            return ExecutionResult{};
        }

        // Get command queue
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)client_->command_queue();
        if (!commandQueue) {
            return ExecutionResult::Error("No Metal command queue available");
        }

        // Allocate slot buffer array
        std::vector<id<MTLBuffer>> slot_bufs(plan_->slots.size(), nil);

        // Assign input buffers to input slots
        for (size_t i = 0; i < num_args; i++) {
            MpsBuffer* input = inputs[i];
            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }
            id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)input->metal_buffer();
            if (!mtl_buf) {
                return ExecutionResult::Error("Input buffer at index " + std::to_string(i) +
                                              " has no Metal buffer");
            }
            SlotId slot = plan_->input_slots[i];
            slot_bufs[slot] = mtl_buf;
        }

        // Pre-allocate intermediate MTLBuffers for graph-step target slots.
        for (const auto& gs : plan_->graph_steps) {
            for (const auto& [slot, tensor] : gs.targets) {
                if (slot_bufs[slot])
                    continue;  // Already assigned (input slot).
                size_t byte_size = plan_->slots[slot].byte_size;
                id<MTLBuffer> buf = [mtl_device newBufferWithLength:byte_size
                                                            options:MTLResourceStorageModeShared];
                if (!buf) {
                    return ExecutionResult::Error("Failed to pre-allocate slot buffer of " +
                                                  std::to_string(byte_size) + " bytes");
                }
                slot_bufs[slot] = buf;
            }
        }

        // Single command buffer for all steps. MPSCommandBuffer is required by
        // MPSGraph's encodeToCommandBuffer: and is accepted by native MPS kernels
        // via id<MTLCommandBuffer> conformance.
        MPSCommandBuffer* cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];

        // Execute steps
        std::string step_error;
        for (const auto& step : plan_->steps) {
            if (step.kind == Step::GRAPH) {
                auto& gs = plan_->graph_steps[step.index];

                // Build feeds dictionary
                NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
                    [NSMutableDictionary dictionary];
                for (auto& [slot, tensor] : gs.feeds) {
                    if (!slot_bufs[slot]) {
                        step_error = "Slot buffer " + std::to_string(slot) +
                                     " is nil during feed construction";
                        break;
                    }
                    MPSGraphTensorData* data =
                        [[MPSGraphTensorData alloc] initWithMTLBuffer:slot_bufs[slot]
                                                                shape:tensor.shape
                                                             dataType:tensor.dataType];
                    feeds[tensor] = data;
                }
                if (!step_error.empty())
                    break;

                // Build target tensors array
                NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray array];
                for (auto& [slot, tensor] : gs.targets) {
                    [targets addObject:tensor];
                }

                NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* result_dict = nil;
                @try {
                    result_dict = [gs.graph encodeToCommandBuffer:cmdBuf
                                                            feeds:feeds
                                                    targetTensors:targets
                                                 targetOperations:nil
                                              executionDescriptor:nil];
                } @catch (NSException* exception) {
                    step_error = "MPS graph execution failed: " +
                                 std::string([[exception reason] UTF8String]);
                    break;
                }

                // Export results GPU-side into pre-allocated slot buffers.
                for (size_t i = 0; i < gs.targets.size(); i++) {
                    SlotId slot = gs.targets[i].first;
                    MPSGraphTensor* target_tensor = gs.targets[i].second;
                    MPSGraphTensorData* result_data = result_dict[target_tensor];
                    if (!result_data) {
                        step_error = "MPS graph execution produced no result for output " +
                                     std::to_string(i);
                        break;
                    }

                    MPSNDArray* ndarray = [result_data mpsndarray];
                    if (!ndarray) {
                        step_error = "Failed to get MPSNDArray from result data";
                        break;
                    }
                    [ndarray exportDataWithCommandBuffer:cmdBuf
                                                toBuffer:slot_bufs[slot]
                                     destinationDataType:plan_->slots[slot].dtype
                                                  offset:0
                                              rowStrides:nil];
                }
                if (!step_error.empty())
                    break;
            } else {
                // ----- Native step -----
                auto& ns = plan_->native_steps[step.index];

                std::vector<id<MTLBuffer>> input_bufs;
                for (auto slot : ns.input_slots) {
                    input_bufs.push_back(slot_bufs[slot]);
                }

                id<MTLBuffer> output = ns.handler(mtl_device, cmdBuf, ns.op, input_bufs);
                if (!output) {
                    step_error = "Native op handler returned nil";
                    break;
                }

                slot_bufs[ns.output_slot] = output;
            }
        }

        // Always commit — releasing an uncommitted MPSCommandBuffer may abort.
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (!step_error.empty()) {
            return ExecutionResult::Error(step_error);
        }
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSString* desc = cmdBuf.error ? cmdBuf.error.localizedDescription : @"unknown error";
            return ExecutionResult::Error("Metal command buffer error: " +
                                          std::string([desc UTF8String]));
        }

        // Build output MpsBuffers
        ExecutionResult result;
        for (size_t i = 0; i < plan_->output_slots.size(); i++) {
            SlotId slot = plan_->output_slots[i];
            id<MTLBuffer> buf = slot_bufs[slot];

            auto tensorType = mlir::cast<mlir::RankedTensorType>(plan_->return_types[i]);
            mlir::Type elemType = tensorType.getElementType();
            int dtype = MlirTypeToPjrtDtype(elemType);
            if (dtype < 0) {
                return ExecutionResult::Error("Unsupported output type for result " +
                                              std::to_string(i));
            }

            std::vector<int64_t> dims(tensorType.getShape().begin(), tensorType.getShape().end());

            // If this slot came directly from an input (identity / pass-through), copy it.
            bool is_input_slot = false;
            for (auto in_slot : plan_->input_slots) {
                if (in_slot == slot) {
                    is_input_slot = true;
                    break;
                }
            }
            if (is_input_slot) {
                size_t byte_size = plan_->slots[slot].byte_size;
                id<MTLBuffer> copy = [mtl_device newBufferWithBytes:buf.contents
                                                             length:byte_size
                                                            options:MTLResourceStorageModeShared];
                if (!copy) {
                    return ExecutionResult::Error("Failed to allocate output buffer copy");
                }
                buf = copy;
            }

            result.buffers.push_back(
                std::make_unique<MpsBuffer>(device, (__bridge void*)buf, dtype, dims));
        }

        return result;
    }
}

}  // namespace jax_mps
