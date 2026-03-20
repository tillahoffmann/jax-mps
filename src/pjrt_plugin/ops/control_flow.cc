// Control flow op handlers (while, case, call, return, composite, custom_call,
// optimization_barrier).

#include <mlx/compile.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Common helper for return-like operations (func.return, stablehlo.return)
bool CollectReturnValues(mlir::Operation* op, ValueMap& values,
                         std::vector<mlx::core::array>& outputs, const char* opName) {
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("%s: operand not found in value map\n", opName);
            return false;
        }
        outputs.push_back(val_opt->get());
    }
    return true;
}

// Handler for func.return
bool HandleReturn(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                  ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "func.return");
}

// Handler for stablehlo.return (region terminator)
bool HandleStablehloReturn(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    return CollectReturnValues(op, values, outputs, "stablehlo.return");
}

// Handler for stablehlo.while
bool HandleWhile(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op);
    if (!whileOp) {
        MPS_LOG_ERROR("stablehlo.while: failed to cast\n");
        return false;
    }

    if (ctx.inside_compile) {
        throw CompileIncompatibleError("stablehlo.while requires eval()");
    }

    std::vector<mlx::core::array> loopVars;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.while: operand not found in value map\n");
            return false;
        }
        loopVars.push_back(val_opt->get());
    }

    auto& condRegion = whileOp.getCond();
    auto& bodyRegion = whileOp.getBody();

    using CompiledFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;
    CompiledFn compiledCond;
    CompiledFn compiledBody;
    bool useCompiled = false;
    try {
        compiledCond = mlx::core::compile(
            [&condRegion, &ctx, &values](
                const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                auto args = inputs;
                std::vector<mlx::core::array> results;
                ExecContext compileCtx;
                compileCtx.module = ctx.module;
                compileCtx.inside_compile = true;
                if (!ExecuteRegion(condRegion, args, results, compileCtx, &values)) {
                    return {};
                }
                return results;
            });

        compiledBody = mlx::core::compile(
            [&bodyRegion, &ctx, &values](
                const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                auto args = inputs;
                std::vector<mlx::core::array> results;
                ExecContext compileCtx;
                compileCtx.module = ctx.module;
                compileCtx.inside_compile = true;
                if (!ExecuteRegion(bodyRegion, args, results, compileCtx, &values)) {
                    return {};
                }
                return results;
            });

        // Probe: run compiled cond+body once to verify they work
        auto testCond = compiledCond(loopVars);
        if (testCond.size() == 1) {
            auto testBody = compiledBody(loopVars);
            if (testBody.size() == loopVars.size()) {
                std::vector<mlx::core::array> toEval;
                toEval.insert(toEval.end(), testCond.begin(), testCond.end());
                toEval.insert(toEval.end(), testBody.begin(), testBody.end());
                mlx::core::eval(toEval);
                useCompiled = true;
            }
        }
    } catch (...) {
        useCompiled = false;
    }

    while (true) {
        std::vector<mlx::core::array> condResults;
        if (useCompiled) {
            condResults = compiledCond(loopVars);
        } else {
            if (!ExecuteRegion(condRegion, loopVars, condResults, ctx, &values)) {
                MPS_LOG_ERROR("stablehlo.while: failed to execute cond region\n");
                return false;
            }
        }

        if (condResults.size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: cond region should return 1 value, got %zu\n",
                          condResults.size());
            return false;
        }

        loopVars.push_back(condResults[0]);
        mlx::core::eval(loopVars);
        auto condVal = std::move(loopVars.back());
        loopVars.pop_back();

        if (condVal.size() != 1) {
            MPS_LOG_ERROR("stablehlo.while: condition must be a scalar, got size %zu\n",
                          condVal.size());
            return false;
        }

        if (!condVal.item<bool>()) {
            break;
        }

        std::vector<mlx::core::array> bodyResults;
        if (useCompiled) {
            bodyResults = compiledBody(loopVars);
        } else {
            if (!ExecuteRegion(bodyRegion, loopVars, bodyResults, ctx, &values)) {
                MPS_LOG_ERROR("stablehlo.while: failed to execute body region\n");
                return false;
            }
        }

        if (bodyResults.size() != loopVars.size()) {
            MPS_LOG_ERROR("stablehlo.while: body returned %zu values, expected %zu\n",
                          bodyResults.size(), loopVars.size());
            return false;
        }

        loopVars = std::move(bodyResults);
    }

    for (size_t i = 0; i < loopVars.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(loopVars[i]));
    }

    return true;
}

// Handler for stablehlo.case (implements lax.cond and lax.switch)
bool HandleCase(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto caseOp = mlir::dyn_cast<mlir::stablehlo::CaseOp>(op);
    if (!caseOp) {
        MPS_LOG_ERROR("stablehlo.case: failed to cast\n");
        return false;
    }

    auto index_opt = GetValue(values, caseOp.getIndex());
    if (!index_opt) {
        MPS_LOG_ERROR("stablehlo.case: index operand not found\n");
        return false;
    }

    if (ctx.inside_compile) {
        throw CompileIncompatibleError("stablehlo.case requires eval()");
    }
    mlx::core::eval(index_opt->get());
    int branchIdx = index_opt->get().item<int>();

    int numBranches = static_cast<int>(caseOp.getBranches().size());
    if (branchIdx < 0 || branchIdx >= numBranches) {
        branchIdx = numBranches - 1;
    }

    auto& branch = caseOp.getBranches()[branchIdx];
    std::vector<mlx::core::array> branchArgs;
    std::vector<mlx::core::array> branchResults;
    if (!ExecuteRegion(branch, branchArgs, branchResults, ctx, &values)) {
        MPS_LOG_ERROR("stablehlo.case: failed to execute branch %d\n", branchIdx);
        return false;
    }

    for (size_t i = 0; i < branchResults.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(branchResults[i]));
    }

    return true;
}

// Handler for func.call
bool HandleCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                ExecContext& ctx) {
    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!callOp) {
        MPS_LOG_ERROR("func.call: failed to cast\n");
        return false;
    }

    auto calleeName = callOp.getCallee();
    auto calleeFunc = ctx.module.lookupSymbol<mlir::func::FuncOp>(calleeName);
    if (!calleeFunc) {
        MPS_LOG_ERROR("func.call: callee function '%s' not found\n", calleeName.str().c_str());
        return false;
    }

    std::vector<mlx::core::array> callInputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("func.call: operand not found in value map\n");
            return false;
        }
        callInputs.push_back(val_opt->get());
    }

    std::vector<mlx::core::array> callOutputs;
    if (!ExecuteFunction(calleeFunc, callInputs, callOutputs, ctx)) {
        MPS_LOG_ERROR("func.call: failed to execute callee '%s'\n", calleeName.str().c_str());
        return false;
    }

    if (callOutputs.size() != op->getNumResults()) {
        MPS_LOG_ERROR("func.call: result count mismatch\n");
        return false;
    }

    for (size_t i = 0; i < callOutputs.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(callOutputs[i]));
    }

    return true;
}

// Handler for stablehlo.custom_call
bool HandleCustomCall(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                      ExecContext& ctx) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        MPS_LOG_ERROR("stablehlo.custom_call: failed to cast\n");
        return false;
    }

    auto callTargetName = customCallOp.getCallTargetName().str();

    // Handle Sharding annotation - just pass input through
    if (callTargetName == "Sharding") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: expected 1 input and 1 output\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call Sharding: operand not found\n");
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    // Handle SPMDFullToShardShape and SPMDShardToFullShape
    if (callTargetName == "SPMDFullToShardShape" || callTargetName == "SPMDShardToFullShape") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: operand not found\n", callTargetName.c_str());
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), input_opt->get());
        return true;
    }

    // Handle unary mhlo.* custom calls
    static const std::unordered_map<std::string, UnaryMlxFn> unaryCustomCalls = {
        {"mhlo.erf", mlx::core::erf},       {"mhlo.sinh", mlx::core::sinh},
        {"mhlo.cosh", mlx::core::cosh},     {"mhlo.asin", mlx::core::arcsin},
        {"mhlo.acos", mlx::core::arccos},   {"mhlo.atan", mlx::core::arctan},
        {"mhlo.asinh", mlx::core::arcsinh}, {"mhlo.acosh", mlx::core::arccosh},
        {"mhlo.atanh", mlx::core::arctanh}, {"mhlo.erf_inv", mlx::core::erfinv},
    };

    auto unaryIt = unaryCustomCalls.find(callTargetName);
    if (unaryIt != unaryCustomCalls.end()) {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: operand not found\n", callTargetName.c_str());
            return false;
        }
        values.emplace(ToKey(op->getResult(0)), unaryIt->second(input_opt->get(), {}));
        return true;
    }

    // Handle mhlo.topk
    if (callTargetName == "mhlo.topk") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.topk: expected 1 input and 2 outputs\n");
            return false;
        }
        auto input_opt = GetValue(values, op->getOperand(0));
        if (!input_opt) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.topk: operand not found\n");
            return false;
        }

        auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        int k = static_cast<int>(resultType.getShape().back());
        int axis = static_cast<int>(input_opt->get().ndim()) - 1;

        auto input = mlx::core::contiguous(input_opt->get());
        auto sortedIndices = mlx::core::argsort(mlx::core::negative(input), axis);

        mlx::core::Shape starts(sortedIndices.ndim(), 0);
        mlx::core::Shape stops(sortedIndices.shape().begin(), sortedIndices.shape().end());
        stops[axis] = k;
        auto indices = mlx::core::slice(sortedIndices, starts, stops);

        auto topValues = mlx::core::take_along_axis(input, indices, axis);
        values.emplace(ToKey(op->getResult(0)), std::move(topValues));
        values.emplace(ToKey(op->getResult(1)), mlx::core::astype(indices, mlx::core::int32));
        return true;
    }

    MPS_LOG_ERROR("stablehlo.custom_call: unsupported target '%s'\n", callTargetName.c_str());
    return false;
}

// Handler for stablehlo.composite
bool HandleComposite(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto compositeOp = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op);
    if (!compositeOp) {
        MPS_LOG_ERROR("stablehlo.composite: failed to cast\n");
        return false;
    }

    auto compositeName = compositeOp.getName().str();
    auto decompositionName = compositeOp.getDecomposition().str();

    MPS_LOG_DEBUG("stablehlo.composite: name=%s decomposition=%s\n", compositeName.c_str(),
                  decompositionName.c_str());

    // Gather inputs
    std::vector<mlx::core::array> inputs;
    for (auto operand : op->getOperands()) {
        auto val_opt = GetValue(values, operand);
        if (!val_opt) {
            MPS_LOG_ERROR("stablehlo.composite %s: operand not found\n", compositeName.c_str());
            return false;
        }
        inputs.push_back(val_opt->get());
    }

    // Try native MLX dispatch for known single-input CHLO composite ops.
    if (inputs.size() == 1 && op->getNumResults() == 1) {
        // clang-format off
        static const std::unordered_map<std::string, UnaryMlxFn> nativeUnary{
            {"chlo.asin",    mlx::core::arcsin},   {"chlo.acos",    mlx::core::arccos},
            {"chlo.atan",    mlx::core::arctan},   {"chlo.asinh",   mlx::core::arcsinh},
            {"chlo.acosh",   mlx::core::arccosh},  {"chlo.atanh",   mlx::core::arctanh},
            {"chlo.sinh",    mlx::core::sinh},     {"chlo.cosh",    mlx::core::cosh},
            {"chlo.tan",     mlx::core::tan},      {"chlo.erf",     mlx::core::erf},
            {"chlo.erf_inv", mlx::core::erfinv},
        };
        // clang-format on

        auto it = nativeUnary.find(compositeName);
        if (it != nativeUnary.end()) {
            values.emplace(ToKey(op->getResult(0)), it->second(inputs[0], {}));
            return true;
        }
    }

    // Try native MLX dispatch for known two-input CHLO composite ops
    if (inputs.size() == 2 && op->getNumResults() == 1 && compositeName == "chlo.atan2") {
        values.emplace(ToKey(op->getResult(0)), mlx::core::arctan2(inputs[0], inputs[1], {}));
        return true;
    }

    // Fallback: execute the decomposition function
    auto decompFunc = ctx.module.lookupSymbol<mlir::func::FuncOp>(decompositionName);
    if (!decompFunc) {
        MPS_LOG_ERROR("stablehlo.composite %s: decomposition function '%s' not found\n",
                      compositeName.c_str(), decompositionName.c_str());
        return false;
    }

    std::vector<mlx::core::array> decompOutputs;
    if (!ExecuteFunction(decompFunc, inputs, decompOutputs, ctx)) {
        MPS_LOG_ERROR("stablehlo.composite %s: decomposition execution failed\n",
                      compositeName.c_str());
        return false;
    }

    if (decompOutputs.size() != op->getNumResults()) {
        MPS_LOG_ERROR("stablehlo.composite %s: result count mismatch (got %zu, expected %u)\n",
                      compositeName.c_str(), decompOutputs.size(), op->getNumResults());
        return false;
    }

    for (size_t i = 0; i < decompOutputs.size(); ++i) {
        values.emplace(ToKey(op->getResult(i)), std::move(decompOutputs[i]));
    }

    return true;
}

// Handler for stablehlo.optimization_barrier
bool HandleOptimizationBarrier(mlir::Operation* op, ValueMap& values,
                               std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto operand_opt = GetValue(values, op->getOperand(i));
        if (!operand_opt) {
            MPS_LOG_ERROR("stablehlo.optimization_barrier: operand %u not found\n", i);
            return false;
        }
        values.emplace(ToKey(op->getResult(i)), operand_opt->get());
    }
    return true;
}

}  // namespace

void RegisterControlFlowHandlers(std::unordered_map<std::string, OpHandler>& handlers) {
    handlers.insert({"func.return", HandleReturn});
    handlers.insert({"func.call", HandleCall});
    handlers.insert({"stablehlo.return", HandleStablehloReturn});
    handlers.insert({"stablehlo.custom_call", HandleCustomCall});
    handlers.insert({"stablehlo.composite", HandleComposite});
    handlers.insert({"stablehlo.optimization_barrier", HandleOptimizationBarrier});
    handlers.insert({"stablehlo.while", HandleWhile});
    handlers.insert({"stablehlo.case", HandleCase});
}

}  // namespace jax_mps
