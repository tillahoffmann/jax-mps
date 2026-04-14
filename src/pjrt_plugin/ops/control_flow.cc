// Control flow op handlers (while, case, call, return, composite, custom_call,
// optimization_barrier).

#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/linalg.h>
#include <mlx/primitives.h>
#include <mlx/random.h>
#include <mlx/transforms.h>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pjrt_plugin/ops/handler_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Shared unary op table used by both the mhlo.* custom_call path (HandleCustomCall)
// and the chlo.* composite path (HandleComposite). Keyed by the bare op name as a
// string_view pointing into the static string literals below, so callers can
// probe with a string_view into the original op name without allocating.
const std::unordered_map<std::string_view, UnaryMlxFn>& UnaryMlxOps() {
    static const std::unordered_map<std::string_view, UnaryMlxFn> kOps = {
        {"sinh", mlx::core::sinh},      {"cosh", mlx::core::cosh},
        {"tan", mlx::core::tan},        {"asin", mlx::core::arcsin},
        {"acos", mlx::core::arccos},    {"atan", mlx::core::arctan},
        {"asinh", mlx::core::arcsinh},  {"acosh", mlx::core::arccosh},
        {"atanh", mlx::core::arctanh},  {"erf", mlx::core::erf},
        {"erf_inv", mlx::core::erfinv},
    };
    return kOps;
}

// Parse the custom_call's backend_config (a JSON string, emitted by our Python
// lowerings in src/jax_plugins/mps/ops.py) into a JSON object. Returns an empty
// object on any failure, so callers can freely probe for keys with defaults.
//
// Note: unregistered MLIR attributes (e.g. `mhlo.backend_config = {...}`) do
// not round-trip through StableHLO portable artifacts, so we can't pass a
// typed DictAttr here — JSON-in-a-string is the stable wire format.
llvm::json::Object ParseBackendConfig(mlir::stablehlo::CustomCallOp op) {
    auto bcAttr = op.getBackendConfig();
    if (!bcAttr)
        return {};
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(*bcAttr);
    if (!strAttr || strAttr.getValue().empty())
        return {};
    auto parsed = llvm::json::parse(strAttr.getValue());
    if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return {};
    }
    if (auto* obj = parsed->getAsObject())
        return std::move(*obj);
    return {};
}

// Convert a boolean mask to an additive attention mask (true -> 0, false -> -1e9)
// cast to the given dtype so MLX's SDPA type check passes with float16.
mlx::core::array BoolMaskToAdditive(const mlx::core::array& mask, mlx::core::Dtype dtype) {
    return mlx::core::astype(
        mlx::core::where(mask, mlx::core::array(0.0F), mlx::core::array(-1e9F)), dtype);
}

// Common helper for return-like operations (func.return, stablehlo.return)
bool CollectReturnValues(mlir::Operation* op, ValueMap& values,
                         std::vector<mlx::core::array>& outputs, const char* opName) {
    for (auto operand : op->getOperands()) {
        auto* val = RequireValue(values, operand, opName);
        if (!val)
            return false;
        outputs.push_back(*val);
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

// Collect arrays referenced inside a region but defined outside it.
// These must be passed as explicit inputs to mx::compile to avoid
// the "uncaptured inputs" error. An optional `seen` set enables
// deduplication across multiple regions (e.g., body + cond).
void CollectExternalValues(mlir::Region& region, const ValueMap& values, std::vector<void*>& keys,
                           std::vector<mlx::core::array>& arrays,
                           std::unordered_set<void*>* seen = nullptr) {
    std::unordered_set<void*> blockArgKeys;
    for (auto arg : region.front().getArguments()) {
        blockArgKeys.insert(ToKey(arg));
    }

    std::unordered_set<void*> localSeen;
    auto& seenRef = seen ? *seen : localSeen;
    // Walk all ops recursively (including nested regions like reduce/if bodies)
    // to capture external values used anywhere inside the while body/cond.
    region.walk([&](mlir::Operation* op) {
        for (auto operand : op->getOperands()) {
            auto* key = ToKey(operand);
            if (blockArgKeys.count(key) || seenRef.count(key))
                continue;
            // Skip values defined inside this region (by any op within it)
            if (operand.getDefiningOp() && operand.getDefiningOp()->getParentRegion() == &region)
                continue;
            auto it = values.find(key);
            if (it != values.end()) {
                seenRef.insert(key);
                keys.push_back(key);
                arrays.push_back(it->second);
            }
        }
    });
}

// Custom MLX primitive that encapsulates a while-loop with compiled body
// and per-step eval. This is opaque to mx::compile (won't be fused) but
// doesn't prevent outer compilation. When eval is called on the compiled
// graph, this primitive runs the loop with per-step eval internally.
//
// Stream placement: This primitive MUST run on the CPU stream, but this does
// NOT move computation to CPU. The compiled body (compiledBody_) internally
// dispatches GPU kernels via MLX's default GPU stream — the CPU side only
// orchestrates the loop: call body → async_eval() to flush GPU work → update
// loop vars → repeat. This is the same pattern as MLX's own eval() scheduler.
//
// Why not the GPU stream: eval()/async_eval() synchronize the GPU command
// queue. Calling them from within a GPU stream eval callback would re-enter
// the GPU scheduler, causing a deadlock. The CPU stream avoids this.
class WhileLoopPrimitive : public mlx::core::Primitive {
public:
    using BodyFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;
    using CondFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;
    using CompiledFn =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;

    // Constructor for counted loops (tripCount >= 0)
    WhileLoopPrimitive(mlx::core::Stream stream, BodyFn bodyFn, size_t nLoopVars, size_t nExt,
                       size_t counterIdx, int64_t tripCount)
        : Primitive(stream),
          compiledBody_(mlx::core::compile(std::move(bodyFn))),
          nLoopVars_(nLoopVars),
          nExt_(nExt),
          counterIdx_(counterIdx),
          tripCount_(tripCount) {
        if (stream.device != mlx::core::Device::cpu) {
            throw std::runtime_error(
                "WhileLoopPrimitive must be on the CPU stream — GPU stream "
                "placement would deadlock on internal eval() calls");
        }
    }

    // Constructor for dynamic condition loops (tripCount < 0, with condFn)
    WhileLoopPrimitive(mlx::core::Stream stream, BodyFn bodyFn, CondFn condFn, size_t nLoopVars,
                       size_t nExt)
        : Primitive(stream),
          compiledBody_(mlx::core::compile(std::move(bodyFn))),
          compiledCond_(mlx::core::compile(std::move(condFn))),
          nLoopVars_(nLoopVars),
          nExt_(nExt),
          counterIdx_(0),
          tripCount_(-1) {
        if (stream.device != mlx::core::Device::cpu) {
            throw std::runtime_error(
                "WhileLoopPrimitive must be on the CPU stream — GPU stream "
                "placement would deadlock on internal eval() calls");
        }
    }

    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override {
        eval_impl(inputs, outputs);
    }

    void eval_gpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override {
        eval_impl(inputs, outputs);
    }

    const char* name() const override {
        return "WhileLoop";
    }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        return false;
    }

    std::vector<mlx::core::Shape> output_shapes(
        const std::vector<mlx::core::array>& inputs) override {
        std::vector<mlx::core::Shape> shapes;
        shapes.reserve(nLoopVars_);
        for (size_t i = 0; i < nLoopVars_; ++i)
            shapes.push_back(inputs[i].shape());
        return shapes;
    }

private:
    void eval_impl(const std::vector<mlx::core::array>& inputs,
                   std::vector<mlx::core::array>& outputs) {
        // inputs = loopVars (nLoopVars_) + external values (nExt_)
        std::vector<mlx::core::array> current(inputs.begin(), inputs.end());

        // Validate input arity: loop vars + external captures.
        if (current.size() != nLoopVars_ + nExt_)
            throw std::runtime_error("WhileLoopPrimitive: expected " +
                                     std::to_string(nLoopVars_ + nExt_) + " inputs, got " +
                                     std::to_string(current.size()));

        if (tripCount_ >= 0) {
            // --- COUNTED LOOP ---
            // Validate counter is a scalar integer before extracting.
            auto& counterArr = current[counterIdx_];
            if (counterArr.size() != 1 ||
                !mlx::core::issubdtype(counterArr.dtype(), mlx::core::integer)) {
                throw std::runtime_error("WhileLoopPrimitive: counter must be a scalar integer");
            }
            mlx::core::eval(counterArr);
            int64_t counter = counterArr.item<int64_t>();

            for (int64_t i = counter; i < tripCount_; ++i) {
                auto bodyResults = compiledBody_(current);
                if (bodyResults.size() != nLoopVars_)
                    throw std::runtime_error("WhileLoopPrimitive: body returned " +
                                             std::to_string(bodyResults.size()) +
                                             " results, expected " + std::to_string(nLoopVars_));
                mlx::core::async_eval(bodyResults);
                for (size_t j = 0; j < nLoopVars_; ++j)
                    current[j] = std::move(bodyResults[j]);
            }
        } else {
            // --- DYNAMIC CONDITION LOOP ---
            while (true) {
                auto condResult = compiledCond_(current);
                if (condResult.size() != 1)
                    throw std::runtime_error(
                        "WhileLoopPrimitive: cond must return exactly 1 result, got " +
                        std::to_string(condResult.size()));
                if (condResult[0].size() != 1)
                    throw std::runtime_error(
                        "WhileLoopPrimitive: cond result must be scalar, got size " +
                        std::to_string(condResult[0].size()));
                mlx::core::eval(condResult[0]);
                if (!condResult[0].item<bool>())
                    break;

                auto bodyResults = compiledBody_(current);
                if (bodyResults.size() != nLoopVars_)
                    throw std::runtime_error("WhileLoopPrimitive: body returned " +
                                             std::to_string(bodyResults.size()) +
                                             " results, expected " + std::to_string(nLoopVars_));
                mlx::core::async_eval(bodyResults);
                for (size_t j = 0; j < nLoopVars_; ++j)
                    current[j] = std::move(bodyResults[j]);
            }
        }

        // Synchronize final results
        using Diff = std::vector<mlx::core::array>::difference_type;
        std::vector<mlx::core::array> finalVars(current.begin(),
                                                current.begin() + static_cast<Diff>(nLoopVars_));
        mlx::core::eval(finalVars);

        for (size_t i = 0; i < nLoopVars_; ++i)
            outputs[i].copy_shared_buffer(current[i]);
    }

    CompiledFn compiledBody_;
    CompiledFn compiledCond_;  // empty for counted loops
    size_t nLoopVars_;
    size_t nExt_;
    size_t counterIdx_;
    int64_t tripCount_;
};

// Handler for stablehlo.while
bool HandleWhile(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                 ExecContext& ctx) {
    auto whileOp = CastOp<mlir::stablehlo::WhileOp>(op, "stablehlo.while");
    if (!whileOp)
        return false;

    if (ctx.inside_compile) {
        // --- Custom primitive approach ---
        // Instead of throwing CompileIncompatibleError, we create a
        // WhileLoopPrimitive that is opaque to mx::compile but runs the
        // loop with compiled body + per-step eval when eval'd.

        std::vector<mlx::core::array> loopVars;
        for (auto operand : op->getOperands()) {
            auto* val = RequireValue(values, operand, "stablehlo.while");
            if (!val)
                return false;
            loopVars.push_back(*val);
        }

        auto& condRegion = whileOp.getCond();
        auto& bodyRegion = whileOp.getBody();
        const size_t nLoopVars = loopVars.size();

        // Counted-loop detection: check if cond is "counter < constant"
        // AND the body increments the counter by exactly +1.
        // This is conservative to avoid misclassifying loops where the counter
        // is updated by != 1, or the cond has additional logic beyond the compare.
        int64_t tripCount = -1;
        size_t counterIdx = 0;
        {
            auto& condBlock = condRegion.front();
            mlir::Operation* cmpOp = nullptr;
            for (auto& innerOp : condBlock.getOperations()) {
                if (innerOp.getName().getStringRef() == "stablehlo.compare") {
                    cmpOp = &innerOp;
                    break;
                }
            }
            if (cmpOp) {
                auto cmpDir = cmpOp->getAttrOfType<mlir::stablehlo::ComparisonDirectionAttr>(
                    "comparison_direction");
                if (cmpDir && cmpDir.getValue() == mlir::stablehlo::ComparisonDirection::LT) {
                    auto lhs = cmpOp->getOperand(0);
                    auto rhs = cmpOp->getOperand(1);
                    if (auto* defOp = rhs.getDefiningOp()) {
                        if (defOp->getName().getStringRef() == "stablehlo.constant") {
                            auto attr = defOp->getAttrOfType<mlir::DenseElementsAttr>("value");
                            if (attr && attr.isSplat() && attr.getElementType().isIntOrIndex()) {
                                int64_t rhsVal = attr.getSplatValue<mlir::APInt>().getSExtValue();
                                for (size_t i = 0; i < condBlock.getNumArguments(); ++i) {
                                    if (lhs == condBlock.getArgument(i)) {
                                        counterIdx = i;
                                        tripCount = rhsVal;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                // Verify the cond return depends only on the compare result.
                // The cond block should end with `stablehlo.return(compare_result)`.
                if (tripCount >= 0) {
                    auto& condTerminator = condBlock.back();
                    if (condTerminator.getName().getStringRef() != "stablehlo.return" ||
                        condTerminator.getNumOperands() != 1 ||
                        condTerminator.getOperand(0) != cmpOp->getResult(0)) {
                        tripCount = -1;  // cond has extra logic, fall back
                    }
                }
            }

            // Verify the body increments the counter by exactly +1.
            // Look for: body return operand[counterIdx] = add(blockArg[counterIdx], constant(1))
            if (tripCount >= 0) {
                auto& bodyBlock = bodyRegion.front();
                auto& bodyTerminator = bodyBlock.back();
                bool validIncrement = false;

                if (bodyTerminator.getName().getStringRef() == "stablehlo.return" &&
                    bodyTerminator.getNumOperands() > counterIdx) {
                    auto counterResult = bodyTerminator.getOperand(counterIdx);
                    if (auto* addOp = counterResult.getDefiningOp()) {
                        if (addOp->getName().getStringRef() == "stablehlo.add" &&
                            addOp->getNumOperands() == 2) {
                            auto addLhs = addOp->getOperand(0);
                            auto addRhs = addOp->getOperand(1);
                            // Check: one operand is blockArg[counterIdx], the other is constant(1)
                            for (int swap = 0; swap < 2; ++swap) {
                                auto argSide = swap == 0 ? addLhs : addRhs;
                                auto constSide = swap == 0 ? addRhs : addLhs;
                                if (argSide == bodyBlock.getArgument(counterIdx)) {
                                    if (auto* constOp = constSide.getDefiningOp()) {
                                        if (constOp->getName().getStringRef() ==
                                            "stablehlo.constant") {
                                            auto cAttr =
                                                constOp->getAttrOfType<mlir::DenseElementsAttr>(
                                                    "value");
                                            if (cAttr && cAttr.isSplat() &&
                                                cAttr.getElementType().isIntOrIndex() &&
                                                cAttr.getSplatValue<mlir::APInt>().getSExtValue() ==
                                                    1) {
                                                validIncrement = true;
                                            }
                                        }
                                    }
                                }
                                if (validIncrement)
                                    break;
                            }
                        }
                    }
                }

                if (!validIncrement)
                    tripCount = -1;  // body doesn't increment by +1, fall back
            }
        }

        // Counted-loop path skips the cond region; only body captures are needed.
        // For dynamic-cond loops, include cond-region captures as well.
        std::vector<void*> extKeys;
        std::vector<mlx::core::array> extArrays;
        std::unordered_set<void*> seen;
        CollectExternalValues(bodyRegion, values, extKeys, extArrays, &seen);
        if (tripCount < 0)
            CollectExternalValues(condRegion, values, extKeys, extArrays, &seen);

        const size_t nExt = extKeys.size();

        // Build body function
        auto module = ctx.module;
        auto bodyFn =
            [&bodyRegion, module, extKeys, nLoopVars,
             nExt](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            using Diff = std::vector<mlx::core::array>::difference_type;
            auto args = std::vector<mlx::core::array>(
                inputs.begin(), inputs.begin() + static_cast<Diff>(nLoopVars));
            ValueMap parentVals;
            for (size_t i = 0; i < nExt; ++i)
                parentVals.emplace(extKeys[i], inputs[nLoopVars + i]);
            std::vector<mlx::core::array> results;
            ExecContext compileCtx;
            compileCtx.module = module;
            compileCtx.inside_compile = true;
            if (!ExecuteRegion(bodyRegion, args, results, compileCtx, &parentVals))
                throw std::runtime_error("WhileLoopPrimitive: body region execution failed");
            return results;
        };

        // Build primitive inputs: loopVars + external values
        std::vector<mlx::core::array> primInputs = loopVars;
        primInputs.insert(primInputs.end(), extArrays.begin(), extArrays.end());

        // Build output shapes/dtypes (same as loopVars)
        std::vector<mlx::core::Shape> outShapes;
        outShapes.reserve(nLoopVars);
        std::vector<mlx::core::Dtype> outDtypes;
        outDtypes.reserve(nLoopVars);
        for (size_t i = 0; i < nLoopVars; ++i) {
            outShapes.push_back(loopVars[i].shape());
            outDtypes.push_back(loopVars[i].dtype());
        }

        // CPU stream for orchestration only — the compiled body dispatches GPU
        // kernels internally. eval()/async_eval() calls inside the loop would
        // deadlock on a GPU stream (see WhileLoopPrimitive class comment).
        auto cpuStream = mlx::core::default_stream(mlx::core::Device::cpu);
        std::shared_ptr<WhileLoopPrimitive> prim;

        if (tripCount >= 0) {
            prim = std::make_shared<WhileLoopPrimitive>(cpuStream, std::move(bodyFn), nLoopVars,
                                                        nExt, counterIdx, tripCount);
        } else {
            // Build condFn for dynamic condition loops
            auto condFn =
                [&condRegion, module, extKeys, nLoopVars, nExt](
                    const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
                auto args = std::vector<mlx::core::array>(
                    inputs.begin(),
                    inputs.begin() +
                        static_cast<std::vector<mlx::core::array>::difference_type>(nLoopVars));
                ValueMap parentVals;
                for (size_t i = 0; i < nExt; ++i)
                    parentVals.emplace(extKeys[i], inputs[nLoopVars + i]);
                std::vector<mlx::core::array> results;
                ExecContext compileCtx;
                compileCtx.module = module;
                compileCtx.inside_compile = true;
                if (!ExecuteRegion(condRegion, args, results, compileCtx, &parentVals))
                    throw std::runtime_error("WhileLoopPrimitive: cond region execution failed");
                return results;
            };
            prim = std::make_shared<WhileLoopPrimitive>(cpuStream, std::move(bodyFn),
                                                        std::move(condFn), nLoopVars, nExt);
        }

        auto outputArrays =
            mlx::core::array::make_arrays(std::move(outShapes), outDtypes, prim, primInputs);
        for (size_t i = 0; i < nLoopVars; ++i)
            values.emplace(ToKey(op->getResult(i)), std::move(outputArrays[i]));
        return true;
    }

    std::vector<mlx::core::array> loopVars;
    for (auto operand : op->getOperands()) {
        auto* val = RequireValue(values, operand, "stablehlo.while");
        if (!val)
            return false;
        loopVars.push_back(*val);
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

        // Only eval the condition — keep loop vars lazy until needed.
        mlx::core::eval(condResults[0]);
        auto condVal = condResults[0];

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
    auto caseOp = CastOp<mlir::stablehlo::CaseOp>(op, "stablehlo.case");
    if (!caseOp)
        return false;

    auto* index = RequireValue(values, caseOp.getIndex(), "stablehlo.case");
    if (!index)
        return false;

    if (ctx.inside_compile) {
        throw CompileIncompatibleError("stablehlo.case requires eval()");
    }
    mlx::core::eval(*index);
    int branchIdx = index->item<int>();

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
    auto callOp = CastOp<mlir::func::CallOp>(op, "func.call");
    if (!callOp)
        return false;

    auto calleeName = callOp.getCallee();
    auto calleeFunc = ctx.module.lookupSymbol<mlir::func::FuncOp>(calleeName);
    if (!calleeFunc) {
        MPS_LOG_ERROR("func.call: callee function '%s' not found\n", calleeName.str().c_str());
        return false;
    }

    std::vector<mlx::core::array> callInputs;
    for (auto operand : op->getOperands()) {
        auto* val = RequireValue(values, operand, "func.call");
        if (!val)
            return false;
        callInputs.push_back(*val);
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
    auto customCallOp = CastOp<mlir::stablehlo::CustomCallOp>(op, "stablehlo.custom_call");
    if (!customCallOp)
        return false;

    auto callTargetName = customCallOp.getCallTargetName().str();

    // Handle Sharding annotation and SPMD shape ops - just pass input through
    if (callTargetName == "Sharding" || callTargetName == "SPMDFullToShardShape" ||
        callTargetName == "SPMDShardToFullShape") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                          callTargetName.c_str());
            return false;
        }
        auto* input = RequireValue(values, op->getOperand(0), callTargetName.c_str());
        if (!input)
            return false;
        values.emplace(ToKey(op->getResult(0)), *input);
        return true;
    }

    // Handle unary mhlo.* custom calls via the shared unary op table.
    // Use string_view throughout so the lookup doesn't allocate.
    static constexpr std::string_view kMhloPrefix = "mhlo.";
    std::string_view callName{callTargetName};
    if (callName.substr(0, kMhloPrefix.size()) == kMhloPrefix) {
        const auto& unaryOps = UnaryMlxOps();
        auto unaryIt = unaryOps.find(callName.substr(kMhloPrefix.size()));
        if (unaryIt != unaryOps.end()) {
            if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
                MPS_LOG_ERROR("stablehlo.custom_call %s: expected 1 input and 1 output\n",
                              callTargetName.c_str());
                return false;
            }
            auto* input = RequireValue(values, op->getOperand(0), callTargetName.c_str());
            if (!input)
                return false;
            values.emplace(ToKey(op->getResult(0)), unaryIt->second(*input, {}));
            return true;
        }
    }

    // Handle mhlo.topk
    if (callTargetName == "mhlo.topk") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("stablehlo.custom_call mhlo.topk: expected 1 input and 2 outputs\n");
            return false;
        }
        auto* input = RequireValue(values, op->getOperand(0), "mhlo.topk");
        if (!input)
            return false;

        auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        int k = static_cast<int>(resultType.getShape().back());
        auto [topValues, indices] = TopKImpl(*input, k);
        values.emplace(ToKey(op->getResult(0)), std::move(topValues));
        values.emplace(ToKey(op->getResult(1)), std::move(indices));
        return true;
    }

    // Handle mps.sdpa — fused scaled dot-product attention via mlx::core::fast.
    // Inputs: queries (B, N, T, H), keys (B, N_kv, S, H), values (B, N_kv, S, H),
    //         mask (boolean, broadcastable to (B, N, T, S))
    // backend_config: {"scale": <float>}
    if (callTargetName == "mps.sdpa") {
        if (op->getNumOperands() != 4 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.sdpa: expected 4 inputs and 1 output\n");
            return false;
        }
        auto* queries = RequireValue(values, op->getOperand(0), "mps.sdpa");
        auto* keys = RequireValue(values, op->getOperand(1), "mps.sdpa");
        auto* vals = RequireValue(values, op->getOperand(2), "mps.sdpa");
        auto* mask = RequireValue(values, op->getOperand(3), "mps.sdpa");
        if (!queries || !keys || !vals || !mask)
            return false;

        float scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("scale"))
            scale = static_cast<float>(*v);

        auto additive_mask = BoolMaskToAdditive(*mask, queries->dtype());
        auto result = mlx::core::fast::scaled_dot_product_attention(*queries, *keys, *vals, scale,
                                                                    "", additive_mask);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.sdpa_causal — fused SDPA with causal masking.
    if (callTargetName == "mps.sdpa_causal") {
        if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.sdpa_causal: expected 3 inputs and 1 output\n");
            return false;
        }
        auto* queries = RequireValue(values, op->getOperand(0), "mps.sdpa_causal");
        auto* keys = RequireValue(values, op->getOperand(1), "mps.sdpa_causal");
        auto* vals = RequireValue(values, op->getOperand(2), "mps.sdpa_causal");
        if (!queries || !keys || !vals)
            return false;

        float scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("scale"))
            scale = static_cast<float>(*v);

        auto result =
            mlx::core::fast::scaled_dot_product_attention(*queries, *keys, *vals, scale, "causal");
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.rms_norm — fused RMS normalization via mlx::core::fast.
    // Inputs: x, weight
    // backend_config: {"eps": <float>}
    if (callTargetName == "mps.rms_norm") {
        if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.rms_norm: expected 2 inputs and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.rms_norm");
        auto* weight = RequireValue(values, op->getOperand(1), "mps.rms_norm");
        if (!x || !weight)
            return false;

        float eps = 1e-6F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("eps"))
            eps = static_cast<float>(*v);

        // MLX rms_norm applies: x / sqrt(mean(x^2) + eps) * weight
        // Gemma's RMSNorm uses (1 + weight), so we pass (1 + weight) here.
        // The Python side is responsible for passing the correct weight.
        auto result = mlx::core::fast::rms_norm(*x, *weight, eps);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.addmm — fused matmul + bias emitted by the fuse_bias_add
    // StableHLO rewrite pattern. Inputs: lhs, rhs, bias (1D, size = trailing
    // output dim). Computes bias + lhs @ rhs with the standard matmul layout
    // (contracting dims = last of lhs / first of rhs after any leading batch
    // dims).
    if (callTargetName == "mps.addmm") {
        if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.addmm: expected 3 inputs and 1 output\n");
            return false;
        }
        auto* lhs = RequireValue(values, op->getOperand(0), "mps.addmm");
        auto* rhs = RequireValue(values, op->getOperand(1), "mps.addmm");
        auto* bias = RequireValue(values, op->getOperand(2), "mps.addmm");
        if (!lhs || !rhs || !bias)
            return false;

        auto result = mlx::core::addmm(*bias, *lhs, *rhs);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.softmax — fused numerically-stable softmax emitted by
    // fuse_softmax. One input, one output; axis comes from backend_config.
    if (callTargetName == "mps.softmax") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.softmax: expected 1 input and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.softmax");
        if (!x)
            return false;
        const int ndim = static_cast<int>(x->ndim());
        int axis = ndim - 1;  // default = trailing
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("axis"))
            axis = static_cast<int>(*v);
        if (axis < 0)
            axis += ndim;
        if (axis < 0 || axis >= ndim) {
            MPS_LOG_ERROR("mps.softmax: axis %d out of range for rank %d\n", axis, ndim);
            return false;
        }
        auto result = mlx::core::softmax(*x, std::vector<int>{axis}, /*precise=*/false);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.layer_norm — fused layer normalization via mlx::core::fast.
    // Inputs: x, weight, bias
    // backend_config: {"eps": <float>}
    if (callTargetName == "mps.layer_norm") {
        if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.layer_norm: expected 3 inputs and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.layer_norm");
        auto* weight = RequireValue(values, op->getOperand(1), "mps.layer_norm");
        auto* bias = RequireValue(values, op->getOperand(2), "mps.layer_norm");
        if (!x || !weight || !bias)
            return false;

        float eps = 1e-5F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("eps"))
            eps = static_cast<float>(*v);

        auto result = mlx::core::fast::layer_norm(*x, *weight, *bias, eps);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.layer_norm_bwd — backward for layer norm via mlx::core::vjp.
    // Inputs: x, weight, bias, grad_out; Outputs: dx, dweight, dbias
    if (callTargetName == "mps.layer_norm_bwd") {
        if (op->getNumOperands() != 4 || op->getNumResults() != 3) {
            MPS_LOG_ERROR("mps.layer_norm_bwd: expected 4 inputs and 3 outputs\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.layer_norm_bwd");
        auto* w = RequireValue(values, op->getOperand(1), "mps.layer_norm_bwd");
        auto* b = RequireValue(values, op->getOperand(2), "mps.layer_norm_bwd");
        auto* g = RequireValue(values, op->getOperand(3), "mps.layer_norm_bwd");
        if (!x || !w || !b || !g)
            return false;

        float eps = 1e-5F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("eps"))
            eps = static_cast<float>(*v);

        auto vjp_fn = [eps](const std::vector<mlx::core::array>& primals) {
            return std::vector<mlx::core::array>{
                mlx::core::fast::layer_norm(primals[0], primals[1], primals[2], eps)};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*x, *w, *b}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        values.emplace(ToKey(op->getResult(1)), std::move(grads[1]));
        values.emplace(ToKey(op->getResult(2)), std::move(grads[2]));
        return true;
    }

    // Handle mps.rope — fused rotary position embeddings via mlx::core::fast.
    // Inputs: x (..., T, D), offset (int32 scalar)
    // backend_config: {"dims": <int>, "traditional": <bool>, "base": <float>,
    //                  "rope_scale": <float>}
    if (callTargetName == "mps.rope") {
        if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.rope: expected 2 inputs and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.rope");
        auto* offsetArr = RequireValue(values, op->getOperand(1), "mps.rope");
        if (!x || !offsetArr)
            return false;

        int dims = 0;
        bool traditional = false;
        float base = 10000.0F;
        float rope_scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getInteger("dims"))
            dims = static_cast<int>(*v);
        if (auto v = bc.getNumber("base"))
            base = static_cast<float>(*v);
        if (auto v = bc.getNumber("rope_scale"))
            rope_scale = static_cast<float>(*v);
        if (auto v = bc.getBoolean("traditional"))
            traditional = *v;

        auto result = mlx::core::fast::rope(*x, dims, traditional, base, rope_scale, *offsetArr);
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.gelu — approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // MLX's compiler will fuse these element-wise ops into a single Metal kernel.
    if (callTargetName == "mps.gelu") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.gelu: expected 1 input and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.gelu");
        if (!x)
            return false;

        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Cast constants to input dtype to avoid f16→f32 promotion.
        auto dt = x->dtype();
        auto c = [dt](float v) { return mlx::core::astype(mlx::core::array(v), dt); };
        auto x3 = mlx::core::multiply(mlx::core::multiply(*x, *x, {}), *x, {});
        auto inner = mlx::core::multiply(
            c(0.7978845608F), mlx::core::add(*x, mlx::core::multiply(c(0.044715F), x3, {}), {}),
            {});
        auto result = mlx::core::multiply(
            c(0.5F),
            mlx::core::multiply(*x, mlx::core::add(c(1.0F), mlx::core::tanh(inner, {}), {}), {}),
            {});
        values.emplace(ToKey(op->getResult(0)), std::move(result));
        return true;
    }

    // Handle mps.sdpa_bwd — backward for SDPA via mlx::core::vjp.
    // Inputs: q, k, v, mask, grad_out; backend_config: {"scale": <float>}
    // Outputs: dq, dk, dv
    if (callTargetName == "mps.sdpa_bwd") {
        if (op->getNumOperands() != 5 || op->getNumResults() != 3) {
            MPS_LOG_ERROR("mps.sdpa_bwd: expected 5 inputs and 3 outputs\n");
            return false;
        }
        auto* q = RequireValue(values, op->getOperand(0), "mps.sdpa_bwd");
        auto* k = RequireValue(values, op->getOperand(1), "mps.sdpa_bwd");
        auto* v = RequireValue(values, op->getOperand(2), "mps.sdpa_bwd");
        auto* mask = RequireValue(values, op->getOperand(3), "mps.sdpa_bwd");
        auto* g = RequireValue(values, op->getOperand(4), "mps.sdpa_bwd");
        if (!q || !k || !v || !mask || !g)
            return false;

        float scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("scale"))
            scale = static_cast<float>(*v);

        auto additive_mask = BoolMaskToAdditive(*mask, q->dtype());
        auto vjp_fn = [scale, &additive_mask](const std::vector<mlx::core::array>& primals) {
            return std::vector<mlx::core::array>{mlx::core::fast::scaled_dot_product_attention(
                primals[0], primals[1], primals[2], scale, "", additive_mask)};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*q, *k, *v}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        values.emplace(ToKey(op->getResult(1)), std::move(grads[1]));
        values.emplace(ToKey(op->getResult(2)), std::move(grads[2]));
        return true;
    }

    // Handle mps.sdpa_causal_bwd — backward for causal SDPA via mlx::core::vjp.
    if (callTargetName == "mps.sdpa_causal_bwd") {
        if (op->getNumOperands() != 4 || op->getNumResults() != 3) {
            MPS_LOG_ERROR("mps.sdpa_causal_bwd: expected 4 inputs and 3 outputs\n");
            return false;
        }
        auto* q = RequireValue(values, op->getOperand(0), "mps.sdpa_causal_bwd");
        auto* k = RequireValue(values, op->getOperand(1), "mps.sdpa_causal_bwd");
        auto* v = RequireValue(values, op->getOperand(2), "mps.sdpa_causal_bwd");
        auto* g = RequireValue(values, op->getOperand(3), "mps.sdpa_causal_bwd");
        if (!q || !k || !v || !g)
            return false;

        float scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("scale"))
            scale = static_cast<float>(*v);

        auto vjp_fn = [scale](const std::vector<mlx::core::array>& primals) {
            return std::vector<mlx::core::array>{mlx::core::fast::scaled_dot_product_attention(
                primals[0], primals[1], primals[2], scale, "causal")};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*q, *k, *v}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        values.emplace(ToKey(op->getResult(1)), std::move(grads[1]));
        values.emplace(ToKey(op->getResult(2)), std::move(grads[2]));
        return true;
    }

    // Handle mps.rms_norm_bwd — backward for RMS norm via mlx::core::vjp.
    // Inputs: x, weight, grad_out; backend_config: {"eps": <float>}
    // Outputs: dx, dweight
    if (callTargetName == "mps.rms_norm_bwd") {
        if (op->getNumOperands() != 3 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("mps.rms_norm_bwd: expected 3 inputs and 2 outputs\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.rms_norm_bwd");
        auto* w = RequireValue(values, op->getOperand(1), "mps.rms_norm_bwd");
        auto* g = RequireValue(values, op->getOperand(2), "mps.rms_norm_bwd");
        if (!x || !w || !g)
            return false;

        float eps = 1e-6F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getNumber("eps"))
            eps = static_cast<float>(*v);

        auto vjp_fn = [eps](const std::vector<mlx::core::array>& primals) {
            return std::vector<mlx::core::array>{
                mlx::core::fast::rms_norm(primals[0], primals[1], eps)};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*x, *w}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        values.emplace(ToKey(op->getResult(1)), std::move(grads[1]));
        return true;
    }

    // Handle mps.rope_bwd — backward for RoPE via mlx::core::vjp.
    // Inputs: x, offset (int32 scalar), grad_out; backend_config same as mps.rope
    // Outputs: dx
    if (callTargetName == "mps.rope_bwd") {
        if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.rope_bwd: expected 3 inputs and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.rope_bwd");
        auto* offsetArr = RequireValue(values, op->getOperand(1), "mps.rope_bwd");
        auto* g = RequireValue(values, op->getOperand(2), "mps.rope_bwd");
        if (!x || !offsetArr || !g)
            return false;

        int dims = 0;
        bool traditional = false;
        float base = 10000.0F;
        float rope_scale = 1.0F;
        auto bc = ParseBackendConfig(customCallOp);
        if (auto v = bc.getInteger("dims"))
            dims = static_cast<int>(*v);
        if (auto v = bc.getNumber("base"))
            base = static_cast<float>(*v);
        if (auto v = bc.getNumber("rope_scale"))
            rope_scale = static_cast<float>(*v);
        if (auto v = bc.getBoolean("traditional"))
            traditional = *v;

        auto vjp_fn = [dims, traditional, base, rope_scale,
                       &offsetArr](const std::vector<mlx::core::array>& primals) {
            return std::vector<mlx::core::array>{
                mlx::core::fast::rope(primals[0], dims, traditional, base, rope_scale, *offsetArr)};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*x}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        return true;
    }

    // Handle mps.gelu_bwd — backward for GELU via mlx::core::vjp.
    // Inputs: x, grad_out; Outputs: dx
    if (callTargetName == "mps.gelu_bwd") {
        if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
            MPS_LOG_ERROR("mps.gelu_bwd: expected 2 inputs and 1 output\n");
            return false;
        }
        auto* x = RequireValue(values, op->getOperand(0), "mps.gelu_bwd");
        auto* g = RequireValue(values, op->getOperand(1), "mps.gelu_bwd");
        if (!x || !g)
            return false;

        auto vjp_fn = [](const std::vector<mlx::core::array>& primals) {
            const auto& x = primals[0];
            auto dt = x.dtype();
            auto c = [dt](float v) { return mlx::core::astype(mlx::core::array(v), dt); };
            auto x3 = mlx::core::multiply(mlx::core::multiply(x, x, {}), x, {});
            auto inner = mlx::core::multiply(
                c(0.7978845608F), mlx::core::add(x, mlx::core::multiply(c(0.044715F), x3, {}), {}),
                {});
            return std::vector<mlx::core::array>{mlx::core::multiply(
                c(0.5F),
                mlx::core::multiply(x, mlx::core::add(c(1.0F), mlx::core::tanh(inner, {}), {}), {}),
                {})};
        };
        auto [fwd_out, grads] = mlx::core::vjp(vjp_fn, {*x}, {*g});
        values.emplace(ToKey(op->getResult(0)), std::move(grads[0]));
        return true;
    }

    // Handle mps.eigh — symmetric eigendecomposition via MLX.
    // Inputs: a (symmetric matrix).  Outputs: eigenvectors, eigenvalues.
    if (callTargetName == "mps.eigh") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("mps.eigh: expected 1 input and 2 outputs\n");
            return false;
        }
        auto* a = RequireValue(values, op->getOperand(0), "mps.eigh");
        if (!a)
            return false;

        bool lower = true;
        if (auto v = ParseBackendConfig(customCallOp).getBoolean("lower"))
            lower = *v;

        auto a_contig = mlx::core::contiguous(*a);
        auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(a_contig, lower ? "L" : "U");
        values.emplace(ToKey(op->getResult(0)), std::move(eigenvectors));
        values.emplace(ToKey(op->getResult(1)), std::move(eigenvalues));
        return true;
    }

    // Handle mps.qr — QR decomposition via MLX.
    // Inputs: a.  Outputs: Q, R.
    if (callTargetName == "mps.qr") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 2) {
            MPS_LOG_ERROR("mps.qr: expected 1 input and 2 outputs\n");
            return false;
        }
        auto* a = RequireValue(values, op->getOperand(0), "mps.qr");
        if (!a)
            return false;

        bool full_matrices = false;
        if (auto v = ParseBackendConfig(customCallOp).getBoolean("full_matrices"))
            full_matrices = *v;

        auto a_contig = mlx::core::contiguous(*a);
        auto [Q, R] = mlx::core::linalg::qr(a_contig);

        if (full_matrices) {
            // MLX QR returns thin: Q is M×K, R is K×N (K=min(M,N)).
            // For full_matrices: Q should be M×M, R should be M×N.
            // Pad with zeros (the extra Q columns are not a proper
            // orthogonal complement, but match JAX's convention for
            // reconstruction: Q @ R = A regardless).
            auto ndim = Q.ndim();
            auto M = a->shape(static_cast<int>(a->ndim()) - 2);
            auto N = a->shape(static_cast<int>(a->ndim()) - 1);
            auto K = std::min(M, N);

            if (Q.shape(static_cast<int>(ndim) - 1) != M) {
                // Pad Q columns: M×K → M×M
                auto pad_width = mlx::core::zeros({Q.shape(0), M - K}, Q.dtype());
                if (ndim == 2) {
                    Q = mlx::core::concatenate({Q, pad_width}, 1);
                } else {
                    // Batched: reshape pad to match batch dims
                    auto pw_shape = Q.shape();
                    pw_shape[static_cast<int>(ndim) - 1] = M - K;
                    pad_width = mlx::core::zeros(pw_shape, Q.dtype());
                    Q = mlx::core::concatenate({Q, pad_width}, static_cast<int>(ndim) - 1);
                }
            }
            if (R.shape(static_cast<int>(ndim) - 2) != M) {
                // Pad R rows: K×N → M×N
                auto pw_shape = R.shape();
                pw_shape[static_cast<int>(ndim) - 2] = M - K;
                auto pad_rows = mlx::core::zeros(pw_shape, R.dtype());
                R = mlx::core::concatenate({R, pad_rows}, static_cast<int>(ndim) - 2);
            }
        }

        values.emplace(ToKey(op->getResult(0)), std::move(Q));
        values.emplace(ToKey(op->getResult(1)), std::move(R));
        return true;
    }

    // Handle mps.svd — SVD via MLX.
    // Inputs: a.  Outputs: U, S, Vt (compute_uv=true) or S (compute_uv=false).
    if (callTargetName == "mps.svd") {
        if (op->getNumOperands() != 1) {
            MPS_LOG_ERROR("mps.svd: expected 1 input\n");
            return false;
        }
        auto* a = RequireValue(values, op->getOperand(0), "mps.svd");
        if (!a)
            return false;

        if (op->getNumResults() != 1 && op->getNumResults() != 3) {
            MPS_LOG_ERROR("mps.svd: expected 1 or 3 outputs, got %u\n", op->getNumResults());
            return false;
        }
        bool compute_uv = (op->getNumResults() == 3);

        bool full_matrices = true;
        if (auto v = ParseBackendConfig(customCallOp).getBoolean("full_matrices"))
            full_matrices = *v;

        auto a_contig = mlx::core::contiguous(*a);
        auto results = mlx::core::linalg::svd(a_contig, compute_uv, {});

        if (compute_uv) {
            if (results.size() != 3) {
                MPS_LOG_ERROR("mps.svd: MLX returned %zu results, expected 3\n", results.size());
                return false;
            }
            auto U = results[0];   // M×M (full)
            auto S = results[1];   // K
            auto Vt = results[2];  // N×N (full)

            if (!full_matrices) {
                // Slice to thin: U → M×K, Vt → K×N
                auto ndim = static_cast<int>(U.ndim());
                auto K = static_cast<int>(S.shape(static_cast<int>(S.ndim()) - 1));

                // U[:, :K]
                mlx::core::Shape u_start(ndim, 0);
                auto u_stop = U.shape();
                u_stop[ndim - 1] = K;
                U = mlx::core::slice(U, u_start, u_stop);

                // Vt[:K, :]
                mlx::core::Shape vt_start(ndim, 0);
                auto vt_stop = Vt.shape();
                vt_stop[ndim - 2] = K;
                Vt = mlx::core::slice(Vt, vt_start, vt_stop);
            }

            // JAX svd_p returns (s, u, vt) – singular values first!
            values.emplace(ToKey(op->getResult(0)), std::move(S));
            values.emplace(ToKey(op->getResult(1)), std::move(U));
            values.emplace(ToKey(op->getResult(2)), std::move(Vt));
        } else {
            if (results.size() != 1) {
                MPS_LOG_ERROR("mps.svd: MLX returned %zu results, expected 1\n", results.size());
                return false;
            }
            values.emplace(ToKey(op->getResult(0)), std::move(results[0]));  // S
        }
        return true;
    }

    MPS_LOG_ERROR("stablehlo.custom_call: unsupported target '%s'\n", callTargetName.c_str());
    return false;
}

// Handler for stablehlo.composite
bool HandleComposite(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs,
                     ExecContext& ctx) {
    auto compositeOp = CastOp<mlir::stablehlo::CompositeOp>(op, "stablehlo.composite");
    if (!compositeOp)
        return false;

    auto compositeName = compositeOp.getName().str();
    auto decompositionName = compositeOp.getDecomposition().str();

    MPS_LOG_DEBUG("stablehlo.composite: name=%s decomposition=%s\n", compositeName.c_str(),
                  decompositionName.c_str());

    // Gather inputs
    std::vector<mlx::core::array> inputs;
    for (auto operand : op->getOperands()) {
        auto* val = RequireValue(values, operand, compositeName.c_str());
        if (!val)
            return false;
        inputs.push_back(*val);
    }

    // Try native MLX dispatch for known single-input CHLO composite ops,
    // via the shared unary op table. Use string_view to avoid per-lookup allocation.
    static constexpr std::string_view kChloPrefix = "chlo.";
    std::string_view compName{compositeName};
    if (inputs.size() == 1 && op->getNumResults() == 1 &&
        compName.substr(0, kChloPrefix.size()) == kChloPrefix) {
        const auto& unaryOps = UnaryMlxOps();
        auto it = unaryOps.find(compName.substr(kChloPrefix.size()));
        if (it != unaryOps.end()) {
            values.emplace(ToKey(op->getResult(0)), it->second(inputs[0], {}));
            return true;
        }
    }

    // Try native MLX dispatch for known two-input CHLO composite ops
    if (inputs.size() == 2 && op->getNumResults() == 1 && compositeName == "chlo.atan2") {
        values.emplace(ToKey(op->getResult(0)), mlx::core::arctan2(inputs[0], inputs[1], {}));
        return true;
    }

    // Handle chlo.top_k natively using shared TopKImpl
    if (inputs.size() == 1 && op->getNumResults() == 2 && compositeName == "chlo.top_k") {
        auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
        int k = static_cast<int>(resultType.getShape().back());
        auto [topValues, indices] = TopKImpl(inputs[0], k);
        values.emplace(ToKey(op->getResult(0)), std::move(topValues));
        values.emplace(ToKey(op->getResult(1)), std::move(indices));
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
        auto* val = RequireValue(values, op->getOperand(i), "stablehlo.optimization_barrier");
        if (!val)
            return false;
        values.emplace(ToKey(op->getResult(i)), *val);
    }
    return true;
}

// Handler for stablehlo.rng_bit_generator.
// Uses MLX's PRNG seeded from the state. Random values will differ from CPU/GPU backends
// but the state is correctly tracked (counter incremented by ceil(output_bytes / 16)).
bool HandleRngBitGenerator(mlir::Operation* op, ValueMap& values,
                           std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto rngOp = CastOp<mlir::stablehlo::RngBitGeneratorOp>(op, "stablehlo.rng_bit_generator");
    if (!rngOp)
        return false;

    auto* state = RequireValue(values, op->getOperand(0), "stablehlo.rng_bit_generator");
    if (!state)
        return false;

    // Get shapes/dtypes from MLIR types (available at graph construction time)
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
    if (!outputType) {
        MPS_LOG_ERROR("stablehlo.rng_bit_generator: result type is not RankedTensorType\n");
        return false;
    }
    auto outputShape = GetShape(outputType);
    auto outputDtype = MlirTypeToMlxDtype(outputType.getElementType());

    // Derive MLX random key (uint32[2]) from the state.
    // State is uint64[2] (from JAX lowering). View as uint32 and take first 2 elements as key.
    auto state_u32 = mlx::core::view(mlx::core::flatten(*state), mlx::core::uint32);
    auto key = mlx::core::slice(state_u32, {0}, {2});

    // Generate random bits
    size_t target_bytes = GetDtypeSize(outputDtype);
    mlx::core::array random_output(0);
    if (target_bytes <= 4) {
        random_output = mlx::core::random::bits(outputShape, 4, key);
        random_output = mlx::core::astype(random_output, outputDtype);
    } else {
        // For uint64 output: generate 2x uint32 per element and reinterpret
        auto expandedShape = outputShape;
        expandedShape.push_back(2);
        auto bits32 = mlx::core::random::bits(expandedShape, 4, key);
        random_output = mlx::core::view(bits32, outputDtype);
        random_output = mlx::core::reshape(random_output, outputShape);
    }

    // Compute counter increment: 4 uint32 outputs per counter step (16 bytes).
    size_t total_elements = 1;
    for (auto d : outputShape)
        total_elements *= static_cast<size_t>(d);
    size_t total_bytes = total_elements * target_bytes;
    uint64_t counter_incr = (total_bytes + 15) / 16;

    // Update state: state is uint64[2] = [key, counter]. Increment counter (element 1).
    auto key_part = mlx::core::slice(*state, {0}, {1});
    auto counter = mlx::core::add(mlx::core::slice(*state, {1}, {2}),
                                  mlx::core::array(counter_incr, mlx::core::uint64));
    auto output_state = mlx::core::concatenate({key_part, counter}, 0);

    // Result 0: output_state, Result 1: random output
    values.emplace(ToKey(op->getResult(0)), std::move(output_state));
    values.emplace(ToKey(op->getResult(1)), std::move(random_output));
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
    handlers.insert({"stablehlo.rng_bit_generator", HandleRngBitGenerator});
}

}  // namespace jax_mps
