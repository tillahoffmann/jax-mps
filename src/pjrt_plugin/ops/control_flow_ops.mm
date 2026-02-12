// Control flow operations: while, case

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

namespace {

// Bind block arguments to tensors from an array
void BindBlockArguments(mlir::Block& block, NSArray<MPSGraphTensor*>* tensors, ValueMap& values) {
    for (NSUInteger i = 0; i < tensors.count && i < block.getNumArguments(); ++i) {
        values[block.getArgument(i).getAsOpaquePointer()] = tensors[i];
    }
}

// Evaluate while loop condition block
MPSGraphTensor* EvaluateWhileCond(MPSGraph* graph, mlir::Block& condBlock, const ValueMap& values,
                                  NSArray<MPSGraphTensor*>* inputTensors,
                                  NSMutableArray<MPSGraphTensor*>* resultTensors,
                                  mlir::ModuleOp module, int depth, std::string* blockError,
                                  BlockProcessor processBlock) {
    ValueMap condValues = values;
    BindBlockArguments(condBlock, inputTensors, condValues);

    HandlerContext condCtx(graph, nullptr, condValues, module, depth + 1, processBlock);
    ProcessResult condResult = processBlock(condCtx, condBlock);
    if (!condResult.ok() || condResult.return_values.empty()) {
        *blockError = condResult.ok() ? "while cond returned no predicate" : condResult.error;
        [resultTensors addObjectsFromArray:inputTensors];
        return [graph constantWithScalar:0 dataType:MPSDataTypeBool];
    }

    [resultTensors addObjectsFromArray:inputTensors];
    MPSGraphTensor* pred = GetTensor(condValues, condResult.return_values[0]);
    if (!pred) {
        *blockError = "while cond predicate tensor not found";
        return [graph constantWithScalar:0 dataType:MPSDataTypeBool];
    }
    return pred;
}

// Evaluate while loop body block
NSArray<MPSGraphTensor*>* EvaluateWhileBody(MPSGraph* graph, mlir::Block& bodyBlock,
                                            const ValueMap& values,
                                            NSArray<MPSGraphTensor*>* bodyArgs,
                                            mlir::ModuleOp module, int depth,
                                            std::string* blockError, BlockProcessor processBlock) {
    ValueMap bodyValues = values;
    BindBlockArguments(bodyBlock, bodyArgs, bodyValues);

    HandlerContext bodyCtx(graph, nullptr, bodyValues, module, depth + 1, processBlock);
    ProcessResult bodyResult = processBlock(bodyCtx, bodyBlock);
    if (!bodyResult.ok()) {
        *blockError = bodyResult.error;
        return bodyArgs;
    }

    NSMutableArray<MPSGraphTensor*>* out = [NSMutableArray array];
    for (mlir::Value value : bodyResult.return_values) {
        MPSGraphTensor* tensor = GetTensor(bodyValues, value);
        if (!tensor) {
            *blockError = "while body return tensor not found";
            return bodyArgs;
        }
        [out addObject:tensor];
    }
    return out;
}

}  // namespace

static ProcessResult HandleWhileOp(HandlerContext& ctx) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(ctx.op);
    if (!whileOp) {
        return ProcessResult::Error("Expected stablehlo.while operation");
    }

    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive while");
    }

    NSMutableArray<MPSGraphTensor*>* initialInputs = [NSMutableArray array];
    for (mlir::Value operand : whileOp->getOperands()) {
        MPSGraphTensor* t = GetTensor(ctx.values, operand);
        if (!t)
            return ProcessResult::Error("While operand tensor not found");
        [initialInputs addObject:t];
    }

    if (whileOp.getCond().empty() || whileOp.getBody().empty()) {
        return ProcessResult::Error("stablehlo.while requires non-empty cond/body regions");
    }
    mlir::Block& condBlock = whileOp.getCond().front();
    mlir::Block& bodyBlock = whileOp.getBody().front();

    // Capture context for use in blocks
    MPSGraph* graph = ctx.graph;
    ValueMap& values = ctx.values;
    mlir::ModuleOp module = ctx.module;
    int depth = ctx.depth;
    BlockProcessor processBlock = ctx.processBlock;

    __block std::string blockError;

    NSArray<MPSGraphTensor*>* outputs = [graph whileWithInitialInputs:initialInputs
        before:^MPSGraphTensor*(NSArray<MPSGraphTensor*>* inputTensors,
                                NSMutableArray<MPSGraphTensor*>* resultTensors) {
          return EvaluateWhileCond(graph, condBlock, values, inputTensors, resultTensors, module,
                                   depth, &blockError, processBlock);
        }
        after:^NSArray<MPSGraphTensor*>*(NSArray<MPSGraphTensor*>* bodyArgs) {
          return EvaluateWhileBody(graph, bodyBlock, values, bodyArgs, module, depth, &blockError,
                                   processBlock);
        }
        name:nil];

    if (!blockError.empty())
        return ProcessResult::Error(blockError);
    if (!outputs)
        return ProcessResult::Error("whileWithInitialInputs returned null");
    if ((NSUInteger)whileOp->getNumResults() != outputs.count) {
        return ProcessResult::Error("while output arity mismatch");
    }

    for (NSUInteger i = 0; i < outputs.count; ++i) {
        ctx.values[whileOp->getResult(i).getAsOpaquePointer()] = outputs[i];
    }
    return ProcessResult{};
}

static ProcessResult HandleCaseOp(HandlerContext& ctx) {
    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive case");
    }
    if (ctx.op->getNumOperands() < 1) {
        return ProcessResult::Error("stablehlo.case requires selector operand");
    }
    if (ctx.op->getNumRegions() < 1) {
        return ProcessResult::Error("stablehlo.case requires at least one branch region");
    }

    MPSGraphTensor* selector = GetTensor(ctx.values, ctx.op->getOperand(0));
    if (!selector) {
        return ProcessResult::Error("stablehlo.case selector tensor not found");
    }

    const size_t numResults = ctx.op->getNumResults();
    const size_t numBranches = ctx.op->getNumRegions();
    const size_t numBranchOperands = ctx.op->getNumOperands() - 1;

    std::vector<std::vector<MPSGraphTensor*>> branchOutputs(numBranches);
    for (size_t b = 0; b < numBranches; ++b) {
        mlir::Region& region = ctx.op->getRegion((unsigned)b);
        if (region.empty()) {
            return ProcessResult::Error("stablehlo.case branch region is empty");
        }

        mlir::Block& block = region.front();
        if (block.getNumArguments() > numBranchOperands) {
            return ProcessResult::Error(
                "stablehlo.case branch expects more operands than provided");
        }

        ValueMap branchValues = ctx.values;
        for (size_t i = 0; i < block.getNumArguments(); ++i) {
            mlir::Value branchOperand = ctx.op->getOperand(1 + i);
            MPSGraphTensor* argTensor = GetTensor(ctx.values, branchOperand);
            if (!argTensor) {
                return ProcessResult::Error("stablehlo.case branch operand tensor not found");
            }
            branchValues[block.getArgument((unsigned)i).getAsOpaquePointer()] = argTensor;
        }

        HandlerContext branchCtx(ctx.graph, nullptr, branchValues, ctx.module, ctx.depth + 1,
                                 ctx.processBlock);
        ProcessResult branchResult = ctx.processBlock(branchCtx, block);
        if (!branchResult.ok()) {
            return branchResult;
        }
        if (branchResult.return_values.size() != numResults) {
            return ProcessResult::Error("stablehlo.case branch result arity mismatch");
        }

        branchOutputs[b].reserve(numResults);
        for (size_t r = 0; r < numResults; ++r) {
            MPSGraphTensor* t = GetTensor(branchValues, branchResult.return_values[r]);
            if (!t) {
                return ProcessResult::Error("stablehlo.case branch return tensor not found");
            }
            branchOutputs[b].push_back(t);
        }
    }

    for (size_t r = 0; r < numResults; ++r) {
        MPSGraphTensor* selected = branchOutputs[numBranches - 1][r];
        for (size_t i = numBranches - 1; i > 0; --i) {
            MPSGraphTensor* branchIndex = [ctx.graph constantWithScalar:static_cast<double>(i - 1)
                                                               dataType:selector.dataType];
            MPSGraphTensor* pred = [ctx.graph equalWithPrimaryTensor:selector
                                                     secondaryTensor:branchIndex
                                                                name:nil];
            selected = [ctx.graph selectWithPredicateTensor:pred
                                        truePredicateTensor:branchOutputs[i - 1][r]
                                       falsePredicateTensor:selected
                                                       name:nil];
        }
        ctx.values[ctx.op->getResult((unsigned)r).getAsOpaquePointer()] = selected;
    }

    return ProcessResult{};
}

static ProcessResult HandleCustomCall(HandlerContext& ctx) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(ctx.op);
    if (!customCallOp) {
        return ProcessResult::Error("custom_call: expected CustomCallOp");
    }

    std::string target = customCallOp.getCallTargetName().str();

    const OpHandler* handler = CustomCallRegistry::Find(target);
    if (!handler) {
        return ProcessResult::Error(
            "Unsupported custom call target: " + target +
            "\n"
            "See CONTRIBUTING.md for how to add support for new operations.");
    }

    // Delegate to the target-specific handler
    return handler->graph_handler(ctx);
}

// Register control flow ops as regular GRAPH ops
REGISTER_MPS_OP("stablehlo.while", HandleWhileOp);
REGISTER_MPS_OP("stablehlo.case", HandleCaseOp);

// Register custom_call as a regular GRAPH op - it dispatches to CustomCallRegistry internally
REGISTER_MPS_OP("stablehlo.custom_call", HandleCustomCall);

}  // namespace jax_mps
