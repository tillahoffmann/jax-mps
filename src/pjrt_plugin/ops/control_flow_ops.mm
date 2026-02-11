#import "pjrt_plugin/ops/control_flow_ops.h"

#include "stablehlo/dialect/StablehloOps.h"

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

    ProcessResult condResult = processBlock(graph, condBlock, condValues, module, depth + 1);
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

    ProcessResult bodyResult = processBlock(graph, bodyBlock, bodyValues, module, depth + 1);
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

ProcessResult HandleWhileOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values,
                            mlir::ModuleOp module, int depth, BlockProcessor processBlock) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op);
    if (!whileOp) {
        return ProcessResult::Error("Expected stablehlo.while operation");
    }

    if (depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive while");
    }

    NSMutableArray<MPSGraphTensor*>* initialInputs = [NSMutableArray array];
    for (mlir::Value operand : whileOp->getOperands()) {
        MPSGraphTensor* t = GetTensor(values, operand);
        if (!t)
            return ProcessResult::Error("While operand tensor not found");
        [initialInputs addObject:t];
    }

    if (whileOp.getCond().empty() || whileOp.getBody().empty()) {
        return ProcessResult::Error("stablehlo.while requires non-empty cond/body regions");
    }
    mlir::Block& condBlock = whileOp.getCond().front();
    mlir::Block& bodyBlock = whileOp.getBody().front();

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
        values[whileOp->getResult(i).getAsOpaquePointer()] = outputs[i];
    }
    return ProcessResult{};
}

ProcessResult HandleCaseOp(MPSGraph* graph, mlir::Operation* op, ValueMap& values,
                           mlir::ModuleOp module, int depth, BlockProcessor processBlock) {
    if (depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive case");
    }
    if (op->getNumOperands() < 1) {
        return ProcessResult::Error("stablehlo.case requires selector operand");
    }
    if (op->getNumRegions() < 1) {
        return ProcessResult::Error("stablehlo.case requires at least one branch region");
    }

    MPSGraphTensor* selector = GetTensor(values, op->getOperand(0));
    if (!selector) {
        return ProcessResult::Error("stablehlo.case selector tensor not found");
    }

    const size_t numResults = op->getNumResults();
    const size_t numBranches = op->getNumRegions();
    const size_t numBranchOperands = op->getNumOperands() - 1;

    std::vector<std::vector<MPSGraphTensor*>> branchOutputs(numBranches);
    for (size_t b = 0; b < numBranches; ++b) {
        mlir::Region& region = op->getRegion((unsigned)b);
        if (region.empty()) {
            return ProcessResult::Error("stablehlo.case branch region is empty");
        }

        mlir::Block& block = region.front();
        if (block.getNumArguments() > numBranchOperands) {
            return ProcessResult::Error(
                "stablehlo.case branch expects more operands than provided");
        }

        ValueMap branchValues = values;
        for (size_t i = 0; i < block.getNumArguments(); ++i) {
            mlir::Value branchOperand = op->getOperand(1 + i);
            MPSGraphTensor* argTensor = GetTensor(values, branchOperand);
            if (!argTensor) {
                return ProcessResult::Error("stablehlo.case branch operand tensor not found");
            }
            branchValues[block.getArgument((unsigned)i).getAsOpaquePointer()] = argTensor;
        }

        ProcessResult branchResult = processBlock(graph, block, branchValues, module, depth + 1);
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
            MPSGraphTensor* branchIndex = [graph constantWithScalar:static_cast<double>(i - 1)
                                                           dataType:selector.dataType];
            MPSGraphTensor* pred = [graph equalWithPrimaryTensor:selector
                                                 secondaryTensor:branchIndex
                                                            name:nil];
            selected = [graph selectWithPredicateTensor:pred
                                    truePredicateTensor:branchOutputs[i - 1][r]
                                   falsePredicateTensor:selected
                                                   name:nil];
        }
        values[op->getResult((unsigned)r).getAsOpaquePointer()] = selected;
    }

    return ProcessResult{};
}

bool IsControlFlowOp(const std::string& op_name) {
    return op_name == "stablehlo.while" || op_name == "stablehlo.case";
}

// Register control flow ops as special ops in OpRegistry
// This allows getSupportedOps() to find them without needing a separate registry
// The actual dispatch happens in mps_executable.mm via HandleWhileOp/HandleCaseOp
REGISTER_SPECIAL_MPS_OP("stablehlo.while", stablehlo_while);
REGISTER_SPECIAL_MPS_OP("stablehlo.case", stablehlo_case);

// Register stablehlo.custom_call so the parser accepts it
// Actual dispatch happens in mps_executable.mm by looking up the target in CustomCallRegistry
REGISTER_SPECIAL_MPS_OP("stablehlo.custom_call", stablehlo_custom_call);

}  // namespace jax_mps
