#import "pjrt_plugin/mps_executable.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
    void* shape;  // NSArray<NSNumber*>* (bridged, owned by plan)
    MPSDataType dtype;
    size_t byte_size;
};

struct GraphStep {
    void* graph;                                    // MPSGraph* (bridged retained)
    std::vector<std::pair<SlotId, void*>> feeds;    // slot -> MPSGraphTensor* (owned by graph)
    std::vector<std::pair<SlotId, void*>> targets;  // slot -> MPSGraphTensor* (owned by graph)
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
    // Release retained ObjC objects in the plan
    if (plan_) {
        // Release graphs
        for (auto& gs : plan_->graph_steps) {
            if (gs.graph) {
                (void)(__bridge_transfer MPSGraph*)gs.graph;  // ARC releases
            }
        }
        // Release slot shapes
        for (auto& slot : plan_->slots) {
            if (slot.shape) {
                (void)(__bridge_transfer NSArray<NSNumber*>*)slot.shape;  // ARC releases
            }
        }
    }
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

// Find the handler for an operation
static const OpHandler* FindHandler(mlir::Operation* op) {
    std::string op_name = op->getName().getStringRef().str();
    return OpRegistry::Find(op_name);
}

// Forward declaration for recursive processing
static ProcessResult processOperations(HandlerContext& ctx, mlir::Block& block);

// Adapter function for BlockProcessor signature
static ProcessResult processOperationsAdapter(HandlerContext& ctx, mlir::Block& block) {
    return processOperations(ctx, block);
}

// Process a func.call operation by looking up the callee and processing its body
static ProcessResult processCallOp(HandlerContext& ctx, mlir::func::CallOp callOp) {
    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive function");
    }

    // Look up the callee function
    auto callee = ctx.module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
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
        MPSGraphTensor* argTensor = GetTensor(ctx.values, callArg);
        if (!argTensor) {
            return ProcessResult::Error("Call argument " + std::to_string(i) + " not found");
        }

        // Map the function argument to the same tensor
        ctx.values[funcArg.getAsOpaquePointer()] = argTensor;
    }

    // Process the callee function's body
    HandlerContext calleeCtx(ctx.graph, nullptr, ctx.values, ctx.module, ctx.depth + 1,
                             processOperationsAdapter);
    ProcessResult calleeResult = processOperations(calleeCtx, calleeBlock);
    if (!calleeResult.ok()) {
        return calleeResult;
    }

    // Map return values back to call results
    for (size_t i = 0; i < callOp.getNumResults() && i < calleeResult.return_values.size(); i++) {
        mlir::Value returnValue = calleeResult.return_values[i];
        MPSGraphTensor* returnTensor = GetTensor(ctx.values, returnValue);
        if (!returnTensor) {
            return ProcessResult::Error("Return value " + std::to_string(i) +
                                        " not found in callee");
        }
        ctx.values[callOp.getResult(i).getAsOpaquePointer()] = returnTensor;
    }

    return ProcessResult{};
}

// Process operations in a block, handling func.call recursively
static ProcessResult processOperations(HandlerContext& ctx, mlir::Block& block) {
    ProcessResult result;

    for (mlir::Operation& operation : block) {
        mlir::Operation* op = &operation;
        std::string op_name = op->getName().getStringRef().str();
        MPS_LOG_DEBUG("Processing op: %s (results=%u)\n", op_name.c_str(), op->getNumResults());

        // Handle function and StableHLO region returns.
        if (mlir::isa<mlir::func::ReturnOp>(op) || mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
            for (mlir::Value operand : op->getOperands()) {
                result.return_values.push_back(operand);
            }
            continue;
        }

        // Handle func.call - process the callee function recursively
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            ProcessResult callResult = processCallOp(ctx, callOp);
            if (!callResult.ok()) {
                return callResult;
            }
            continue;
        }

        // Look up handler in registry
        const OpHandler* handler = FindHandler(op);
        if (!handler) {
            return ProcessResult::Error(UnsupportedOpsMessage({op_name}) +
                                        "\n\n"
                                        "Supported operations: " +
                                        OpRegistry::ListRegistered());
        }

        // Native ops should not appear in graph processing - they're handled separately
        if (handler->is_native()) {
            return ProcessResult::Error("Native op '" + op_name +
                                        "' encountered during graph building - "
                                        "this should have been handled by segmented execution");
        }

        // Create op context and call handler
        HandlerContext opCtx(ctx.graph, op, ctx.values, ctx.module, ctx.depth,
                             processOperationsAdapter);
        ProcessResult opResult = handler->graph_handler(opCtx);
        if (!opResult.ok())
            return opResult;
    }

    return result;
}

bool MpsExecutable::BuildExecutionPlan() {
    MPS_LOG_DEBUG("BuildExecutionPlan: start, plan_built_=%d\n", plan_built_);

    // Thread-safe lazy initialization of the execution plan
    std::scoped_lock lock(plan_mutex_);
    if (plan_built_)
        return true;

    @autoreleasepool {
        MPS_LOG_DEBUG("BuildExecutionPlan: in autoreleasepool\n");
        if (!client_) {
            error_ = "No MPS client available";
            return false;
        }

        MPS_LOG_DEBUG("BuildExecutionPlan: starting inline pass\n");
        // Manually inline func.call ops whose callees contain native ops.
        // JAX wraps ops like cholesky in private helper functions called via
        // func.call – the segmentation below only walks the entry block, so
        // native ops inside callees would be missed.  We splice the callee's
        // operations directly into the entry block, avoiding the MLIR inliner
        // pass (which requires DialectInlinerInterface on every dialect).
        {
            bool changed = true;
            int pass_num = 0;
            while (changed) {
                changed = false;
                MPS_LOG_DEBUG("BuildExecutionPlan: inline pass iteration %d\n", pass_num++);
                mlir::Block& block = entry_func_.front();
                // NOLINTNEXTLINE(modernize-loop-convert) - iterator invalidated by erase+break
                for (auto it = block.begin(), end = block.end(); it != end; ++it) {
                    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(&*it);
                    if (!callOp)
                        continue;
                    MPS_LOG_DEBUG("BuildExecutionPlan: found call to %s\n",
                                  callOp.getCallee().str().c_str());

                    auto callee = module_->lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                    if (!callee || callee.empty())
                        continue;

                    // Only inline callees that contain native ops.
                    bool has_native = false;
                    callee.walk([&](mlir::Operation* inner) {
                        if (has_native)
                            return;
                        std::string inner_name = inner->getName().getStringRef().str();
                        const OpHandler* h = OpRegistry::Find(inner_name);
                        if (h && h->is_native())
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
        MPS_LOG_DEBUG("BuildExecutionPlan: inline pass complete\n");

        auto plan = std::make_unique<ExecutionPlan>();

        mlir::Block& block = entry_func_.front();
        MPS_LOG_DEBUG("BuildExecutionPlan: got entry block with %zu ops\n",
                      std::distance(block.begin(), block.end()));

        // ------------------------------------------------------------------
        // Pass 1 – segment the ops into graph vs native
        // ------------------------------------------------------------------
        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 1 (segmentation)\n");
        struct SegmentInfo {
            bool is_native;
            std::vector<mlir::Operation*> ops;
        };

        std::vector<SegmentInfo> segments;
        SegmentInfo current_segment{false, {}};
        mlir::func::ReturnOp returnOp = nullptr;

        int op_count = 0;
        for (mlir::Operation& operation : block) {
            MPS_LOG_DEBUG("BuildExecutionPlan: Pass 1 op %d: %s\n", op_count++,
                          operation.getName().getStringRef().str().c_str());
            mlir::Operation* op = &operation;

            if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
                returnOp = ret;
                continue;
            }

            std::string op_name = op->getName().getStringRef().str();
            const OpHandler* handler = OpRegistry::Find(op_name);
            bool is_native = handler && handler->is_native();

            if (is_native) {
                // Validate: native ops must have exactly one result for now
                if (op->getNumResults() != 1) {
                    error_ = "Native op '" + op_name + "' must have exactly one result";
                    return false;
                }
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
        MPS_LOG_DEBUG("BuildExecutionPlan: Pass 1 complete, %zu segments\n", segments.size());

        // ------------------------------------------------------------------
        // Pass 1b – determine which values need materialized slots
        // ------------------------------------------------------------------
        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 1b (slot allocation)\n");
        // Build op → segment map
        std::unordered_map<mlir::Operation*, int> op_to_seg;
        for (int i = 0; i < (int)segments.size(); i++) {
            for (auto* op : segments[i].ops) {
                op_to_seg[op] = i;
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: op_to_seg map built\n");

        // Slot allocation helper
        std::unordered_map<void*, SlotId> value_to_slot;
        auto getOrCreateSlot = [&](mlir::Value value) -> SlotId {
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot called\n");
            void* key = value.getAsOpaquePointer();
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot key=%p\n", key);
            auto it = value_to_slot.find(key);
            if (it != value_to_slot.end()) {
                MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot found existing slot %d\n",
                              it->second);
                return it->second;
            }

            SlotId slot = (SlotId)plan->slots.size();
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot creating new slot %d\n", slot);
            value_to_slot[key] = slot;

            SlotInfo info;
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot getting shape\n");
            NSArray<NSNumber*>* shape = GetShapeFromType(value.getType());
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot shape=%p\n", (void*)shape);
            info.shape = (__bridge_retained void*)shape;  // Retain for later use
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot getting dtype\n");
            auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
            info.dtype =
                tensorType ? MlirTypeToMps(tensorType.getElementType()) : MPSDataTypeFloat32;
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot dtype=%d\n", (int)info.dtype);
            info.byte_size = ByteSizeFromType(value.getType());
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot byte_size=%zu\n", info.byte_size);
            plan->slots.push_back(info);
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot done, slot=%d\n", slot);
            return slot;
        };

        MPS_LOG_DEBUG("BuildExecutionPlan: allocating slots for %u function arguments\n",
                      entry_func_.getNumArguments());
        // Function arguments always get slots
        for (unsigned i = 0; i < entry_func_.getNumArguments(); i++) {
            MPS_LOG_DEBUG("BuildExecutionPlan: arg %u\n", i);
            auto arg = entry_func_.getArgument(i);
            SlotId slot = getOrCreateSlot(arg);
            plan->input_slots.push_back(slot);
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: %zu input slots\n", plan->input_slots.size());

        // Return values always get slots
        MPS_LOG_DEBUG("BuildExecutionPlan: allocating return value slots\n");
        if (returnOp) {
            MPS_LOG_DEBUG("BuildExecutionPlan: returnOp has %u operands\n",
                          returnOp->getNumOperands());
            for (mlir::Value operand : returnOp->getOperands()) {
                SlotId slot = getOrCreateSlot(operand);
                plan->output_slots.push_back(slot);
                plan->return_types.push_back(operand.getType());
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: %zu output slots\n", plan->output_slots.size());

        // Values that cross segment boundaries need slots
        MPS_LOG_DEBUG("BuildExecutionPlan: checking segment boundary crossings\n");
        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            MPS_LOG_DEBUG("BuildExecutionPlan: segment %d, %zu ops\n", seg_idx,
                          segments[seg_idx].ops.size());
            int op_idx = 0;
            for (auto* op : segments[seg_idx].ops) {
                MPS_LOG_DEBUG("BuildExecutionPlan:   op %d: %s\n", op_idx++,
                              op->getName().getStringRef().str().c_str());
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
                MPS_LOG_DEBUG("BuildExecutionPlan:   operands done, checking results\n");
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
                MPS_LOG_DEBUG("BuildExecutionPlan:   results done\n");
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: segment boundary crossings done, %zu slots\n",
                      plan->slots.size());

        // ------------------------------------------------------------------
        // Pass 2 – build graph / native steps
        // ------------------------------------------------------------------
        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 2 (step building)\n");
        // NOLINTNEXTLINE(modernize-loop-convert) - need seg_idx for debug logging
        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            auto& seg = segments[seg_idx];
            MPS_LOG_DEBUG("BuildExecutionPlan: Pass 2 segment %d, is_native=%d, %zu ops\n", seg_idx,
                          seg.is_native, seg.ops.size());

            if (seg.is_native) {
                // ----- Native step -----
                mlir::Operation* op = seg.ops[0];
                std::string op_name = op->getName().getStringRef().str();
                const OpHandler* handler = OpRegistry::Find(op_name);

                NativeStep ns;
                ns.handler = handler->native_handler;
                ns.op = op;
                for (mlir::Value operand : op->getOperands()) {
                    ns.input_slots.push_back(value_to_slot[operand.getAsOpaquePointer()]);
                }
                // single-result assumption (validated earlier)
                ns.output_slot = value_to_slot[op->getResult(0).getAsOpaquePointer()];

                plan->native_steps.push_back(ns);
                plan->steps.push_back({Step::NATIVE, plan->native_steps.size() - 1});
            } else {
                // ----- Graph step -----
                MPS_LOG_DEBUG("BuildExecutionPlan: creating MPSGraph\n");
                MPSGraph* graph = [[MPSGraph alloc] init];
                if (!graph) {
                    error_ = "Failed to create MPSGraph";
                    return false;
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: MPSGraph created at %p\n", (void*)graph);
                ValueMap values;
                GraphStep gs;
                gs.graph = (__bridge_retained void*)graph;  // Retain for later use
                MPS_LOG_DEBUG("BuildExecutionPlan: gs.graph set to %p\n", gs.graph);

                // Create placeholders for values entering this segment from outside.
                // Values defined within this segment (including nested regions) should NOT
                // get placeholders. We need to walk ALL operations including those in nested
                // regions to find external dependencies.

                // First, collect ALL ops in this segment including nested ones
                MPS_LOG_DEBUG("BuildExecutionPlan: collecting all ops in segment\n");
                std::unordered_set<mlir::Operation*> all_seg_ops;
                for (auto* op : seg.ops) {
                    all_seg_ops.insert(op);
                    op->walk([&](mlir::Operation* nestedOp) { all_seg_ops.insert(nestedOp); });
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: collected %zu ops in segment\n",
                              all_seg_ops.size());

                std::unordered_set<void*> created_ph;

                // Helper to create placeholder for a value if needed
                auto maybeCreatePlaceholder = [&](mlir::Value operand) {
                    // Skip non-entry block arguments - they're handled by control flow.
                    // Entry block arguments (function arguments) DO need placeholders.
                    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                        if (blockArg.getOwner() != &entry_func_.front()) {
                            return;  // Block argument in nested region, handled by control flow
                        }
                        // Fall through - this is a function argument, needs a placeholder
                    }

                    mlir::Operation* defOp = operand.getDefiningOp();
                    // Skip values produced within this segment (including nested regions)
                    if (defOp && all_seg_ops.count(defOp)) {
                        return;
                    }

                    void* key = operand.getAsOpaquePointer();
                    if (created_ph.count(key)) {
                        return;  // Already created
                    }

                    // Get shape and dtype from the MLIR type
                    NSArray<NSNumber*>* shape = GetShapeFromType(operand.getType());
                    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
                    MPSDataType mps_dtype = tensorType ? MlirTypeToMps(tensorType.getElementType())
                                                       : MPSDataTypeFloat32;

                    MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                                     dataType:mps_dtype
                                                                         name:nil];
                    values[key] = placeholder;
                    created_ph.insert(key);

                    // If this value has a slot, register it as a feed
                    auto slot_it = value_to_slot.find(key);
                    if (slot_it != value_to_slot.end()) {
                        gs.feeds.emplace_back(slot_it->second, (__bridge void*)placeholder);
                    }
                };

                // Walk all operations including those in nested regions
                MPS_LOG_DEBUG("BuildExecutionPlan: creating placeholders\n");
                int ph_count = 0;
                for (auto* op : seg.ops) {
                    op->walk([&](mlir::Operation* nestedOp) {
                        for (mlir::Value operand : nestedOp->getOperands()) {
                            maybeCreatePlaceholder(operand);
                        }
                    });
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: created %zu placeholders\n", created_ph.size());

                // Process ops within this segment
                MPS_LOG_DEBUG("BuildExecutionPlan: processing ops in segment\n");
                @try {
                    int processed = 0;
                    for (auto* op : seg.ops) {
                        std::string op_name = op->getName().getStringRef().str();
                        MPS_LOG_DEBUG("BuildExecutionPlan: processing op %d: %s\n", processed++,
                                      op_name.c_str());

                        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
                            HandlerContext callCtx(graph, op, values, *module_, 0,
                                                   processOperationsAdapter);
                            ProcessResult callResult = processCallOp(callCtx, callOp);
                            if (!callResult.ok()) {
                                error_ = callResult.error;
                                return false;
                            }
                            continue;
                        }

                        // Look up handler
                        const OpHandler* handler = FindHandler(op);
                        if (!handler) {
                            error_ = UnsupportedOpsMessage({op_name}) +
                                     "\n\nSupported operations: " + OpRegistry::ListRegistered();
                            return false;
                        }

                        // Create op context and call handler
                        HandlerContext opCtx(graph, op, values, *module_, 0,
                                             processOperationsAdapter);
                        ProcessResult opResult = handler->graph_handler(opCtx);
                        if (!opResult.ok()) {
                            error_ = opResult.error;
                            return false;
                        }
                    }
                } @catch (NSException* exception) {
                    error_ =
                        "MPS graph build failed: " + std::string([[exception reason] UTF8String]);
                    return false;
                }

                // Identify target values (values leaving this segment)
                MPS_LOG_DEBUG("BuildExecutionPlan: identifying target values\n");
                for (auto* op : seg.ops) {
                    for (mlir::Value result : op->getResults()) {
                        void* key = result.getAsOpaquePointer();
                        if (value_to_slot.count(key)) {
                            SlotId slot = value_to_slot[key];
                            MPSGraphTensor* tensor = values[key];
                            MPS_LOG_DEBUG("BuildExecutionPlan: target slot=%d, key=%p, tensor=%p\n",
                                          slot, key, (void*)tensor);
                            if (tensor) {
                                MPS_LOG_DEBUG("BuildExecutionPlan: tensor shape=%s, dtype=%d\n",
                                              [[tensor.shape description] UTF8String],
                                              (int)tensor.dataType);
                                gs.targets.emplace_back(slot, (__bridge void*)tensor);
                            } else {
                                MPS_LOG_DEBUG("BuildExecutionPlan: WARNING tensor is nil!\n");
                            }
                        }
                    }
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: identified %zu targets\n", gs.targets.size());

                plan->graph_steps.push_back(gs);
                plan->steps.push_back({Step::GRAPH, plan->graph_steps.size() - 1});
            }
        }

        MPS_LOG_DEBUG("BuildExecutionPlan: setting plan_ and plan_built_=true\n");
        MPS_LOG_DEBUG(
            "BuildExecutionPlan: %zu steps, %zu graph_steps, %zu native_steps, %zu slots\n",
            plan->steps.size(), plan->graph_steps.size(), plan->native_steps.size(),
            plan->slots.size());
        plan_ = std::move(plan);
        plan_built_ = true;
    }
    MPS_LOG_DEBUG("BuildExecutionPlan: returning true\n");
    return true;
}

ExecutionResult MpsExecutable::Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device) {
    // Check for compilation errors
    if (!valid_) {
        return ExecutionResult::Error("Cannot execute: compilation failed - " + error_);
    }

    // Build plan on first execution (lazy initialization)
    MPS_LOG_DEBUG("Execute: plan_built_=%d\n", plan_built_);
    if (!plan_built_) {
        MPS_LOG_DEBUG("Execute: building execution plan\n");
        if (!BuildExecutionPlan()) {
            return ExecutionResult::Error("Execution plan build failed: " + error_);
        }
        MPS_LOG_DEBUG("Execute: plan built successfully\n");
    }

    @autoreleasepool {
        MPS_LOG_DEBUG("Execute: entering autoreleasepool\n");
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
                // MpsBuffer retains, release our +1 from newBufferWithBytes
                CFRelease((__bridge CFTypeRef)dst);
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
        if (!cmdBuf) {
            return ExecutionResult::Error("Failed to create command buffer");
        }

        // Execute steps
        MPS_LOG_DEBUG("Executing %zu steps\n", plan_->steps.size());
        for (size_t step_idx = 0; step_idx < plan_->steps.size(); step_idx++) {
            const auto& step = plan_->steps[step_idx];
            MPS_LOG_DEBUG("Step %zu: kind=%s\n", step_idx,
                          step.kind == Step::GRAPH ? "GRAPH" : "NATIVE");
            if (step.kind == Step::GRAPH) {
                auto& gs = plan_->graph_steps[step.index];
                MPS_LOG_DEBUG("  graph_step index=%zu, feeds=%zu, targets=%zu\n", step.index,
                              gs.feeds.size(), gs.targets.size());
                MPSGraph* graph = (__bridge MPSGraph*)gs.graph;
                if (!graph) {
                    return ExecutionResult::Error("Graph is nil at step " +
                                                  std::to_string(step_idx));
                }

                // Build feeds dictionary
                MPS_LOG_DEBUG("  Building feeds dictionary\n");
                NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
                    [NSMutableDictionary dictionary];
                for (auto& [slot, tensor_ptr] : gs.feeds) {
                    MPS_LOG_DEBUG("    Feed slot=%d, tensor_ptr=%p\n", slot, tensor_ptr);
                    if (!slot_bufs[slot]) {
                        return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                      " is nil during feed construction");
                    }
                    MPSGraphTensor* tensor = (__bridge MPSGraphTensor*)tensor_ptr;
                    if (!tensor) {
                        return ExecutionResult::Error("Feed tensor is nil for slot " +
                                                      std::to_string(slot));
                    }
                    MPS_LOG_DEBUG("    Tensor shape: %s, dtype: %d\n",
                                  [[tensor.shape description] UTF8String], (int)tensor.dataType);
                    MPSGraphTensorData* data =
                        [[MPSGraphTensorData alloc] initWithMTLBuffer:slot_bufs[slot]
                                                                shape:tensor.shape
                                                             dataType:tensor.dataType];
                    feeds[tensor] = data;
                }
                MPS_LOG_DEBUG("  Feeds built: %lu entries\n", (unsigned long)[feeds count]);

                // Build target tensors array
                MPS_LOG_DEBUG("  Building targets array\n");
                NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray array];
                for (auto& [slot, tensor_ptr] : gs.targets) {
                    MPS_LOG_DEBUG("    Target slot=%d, tensor_ptr=%p\n", slot, tensor_ptr);
                    MPSGraphTensor* tensor = (__bridge MPSGraphTensor*)tensor_ptr;
                    if (!tensor) {
                        return ExecutionResult::Error("Target tensor is nil for slot " +
                                                      std::to_string(slot));
                    }
                    [targets addObject:tensor];
                }
                MPS_LOG_DEBUG("  Targets built: %lu entries\n", (unsigned long)[targets count]);

                MPS_LOG_DEBUG("  Encoding graph to command buffer\n");
                MPS_LOG_DEBUG("  graph=%p, cmdBuf=%p, feeds count=%lu, targets count=%lu\n",
                              (void*)graph, (void*)cmdBuf, (unsigned long)[feeds count],
                              (unsigned long)[targets count]);
                // Log target tensor details
                for (MPSGraphTensor* t in targets) {
                    MPS_LOG_DEBUG("  target tensor=%p, shape=%s, dtype=%d\n", (void*)t,
                                  [[t.shape description] UTF8String], (int)t.dataType);
                }
                NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* result_dict = nil;
                @try {
                    MPS_LOG_DEBUG("  calling encodeToCommandBuffer...\n");
                    result_dict = [graph encodeToCommandBuffer:cmdBuf
                                                         feeds:feeds
                                                 targetTensors:targets
                                              targetOperations:nil
                                           executionDescriptor:nil];
                    MPS_LOG_DEBUG("  encodeToCommandBuffer returned\n");
                } @catch (NSException* exception) {
                    return ExecutionResult::Error("MPS graph execution failed: " +
                                                  std::string([[exception reason] UTF8String]));
                }
                MPS_LOG_DEBUG("  Graph encoded\n");

                // Export results GPU-side into pre-allocated slot buffers.
                MPS_LOG_DEBUG("  Exporting results to slot buffers\n");
                for (size_t i = 0; i < gs.targets.size(); i++) {
                    SlotId slot = gs.targets[i].first;
                    MPSGraphTensor* target_tensor = (__bridge MPSGraphTensor*)gs.targets[i].second;
                    MPS_LOG_DEBUG("    Exporting target %zu, slot=%d\n", i, slot);
                    MPSGraphTensorData* result_data = result_dict[target_tensor];
                    if (!result_data) {
                        return ExecutionResult::Error(
                            "MPS graph execution produced no result for output " +
                            std::to_string(i));
                    }

                    MPSNDArray* ndarray = [result_data mpsndarray];
                    if (!ndarray) {
                        return ExecutionResult::Error("Failed to get MPSNDArray from result data");
                    }
                    [ndarray exportDataWithCommandBuffer:cmdBuf
                                                toBuffer:slot_bufs[slot]
                                     destinationDataType:plan_->slots[slot].dtype
                                                  offset:0
                                              rowStrides:nil];
                }
                MPS_LOG_DEBUG("  Graph step %zu complete\n", step_idx);
            } else {
                // ----- Native step -----
                auto& ns = plan_->native_steps[step.index];

                std::vector<id<MTLBuffer>> input_bufs;
                input_bufs.reserve(ns.input_slots.size());
                for (auto slot : ns.input_slots) {
                    input_bufs.push_back(slot_bufs[slot]);
                }

                id<MTLBuffer> output = ns.handler(mtl_device, cmdBuf, ns.op, input_bufs);
                if (!output) {
                    return ExecutionResult::Error("Native op handler returned nil");
                }

                slot_bufs[ns.output_slot] = output;
            }
        }

        // Commit and wait for all GPU work to complete.
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
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
                result.buffers.push_back(
                    std::make_unique<MpsBuffer>(device, (__bridge void*)copy, dtype, dims));
                // MpsBuffer retains, release our +1 from newBufferWithBytes
                CFRelease((__bridge CFTypeRef)copy);
            } else {
                result.buffers.push_back(
                    std::make_unique<MpsBuffer>(device, (__bridge void*)buf, dtype, dims));
            }
        }

        return result;
    }
}

}  // namespace jax_mps
