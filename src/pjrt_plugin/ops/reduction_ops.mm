// Reduction operations: reduce (sum, product, max, min, and, or, argmax, argmin)
// reduce_window — cumulative ops (Tier 1) and pooling (Tier 2)
//
// Future tiers for reduce_window:
// - Tier 3: Sum-reduce as convolution with all-ones kernel (arbitrary ranks)
// - Tier 4: General reduce_window via im2col or custom Metal kernel

#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Helper to identify the reduction operation type from the region body
// Returns the operation name if it's a simple binary reduction, empty string otherwise
std::string GetReductionOpType(mlir::Region& body) {
    if (body.empty())
        return "";

    mlir::Block& block = body.front();

    // The reduction body should have exactly one operation (plus terminator)
    // that is the reduction function
    for (mlir::Operation& op : block) {
        std::string opName = op.getName().getStringRef().str();

        // Skip the terminator (stablehlo.return)
        if (opName == "stablehlo.return")
            continue;

        // Return the first binary operation we find
        return opName;
    }

    return "";
}

// For detecting argmax/argmin patterns in multi-result reduce
enum class ArgReduceKind { kUnknown, kMax, kMin };

bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        // If the compare operands are swapped, the max/min interpretation flips.
        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

// Helper: check if all values in an array are 1
bool AllOnes(llvm::ArrayRef<int64_t> arr) {
    return llvm::all_of(arr, [](int64_t v) { return v == 1; });
}

}  // namespace

// Single-result reduce: sum, product, max, min, and, or
static ProcessResult HandleSingleResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }

    // Get the input tensor (first operand)
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("reduce: input tensor not found");
    }

    // Get reduction dimensions
    auto dimensions = reduceOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    // Identify the reduction operation from the body
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [ctx.graph reductionSumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [ctx.graph reductionProductWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [ctx.graph reductionMaximumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [ctx.graph reductionMinimumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.and") {
        result = [ctx.graph reductionAndWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.or") {
        result = [ctx.graph reductionOrWithTensor:input axes:axes name:nil];
    } else {
        return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
    }

    // MPS Graph reduction keeps dimensions (with size 1), but StableHLO reduce removes them
    // Reshape to the expected output shape from the MLIR operation
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && result) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }

    return Result(ctx, result, "reduce");
}

// Multi-result reduce: argmax/argmin patterns
static ProcessResult HandleMultiResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }
    if (ctx.op->getNumResults() != 2 || ctx.op->getNumOperands() < 2) {
        return ProcessResult::Error("reduce: unsupported multi-result shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(ctx, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce: value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("reduce: only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("reduce: unsupported multi-result reduce body");
    }

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (kind == ArgReduceKind::kMax) {
        valueOut = [ctx.graph reductionMaximumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMaximumWithTensor:valueInput axis:axis name:nil];
    } else {
        valueOut = [ctx.graph reductionMinimumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMinimumWithTensor:valueInput axis:axis name:nil];
    }
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("reduce: failed to lower multi-result reduce");
    }

    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [ctx.graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [ctx.graph castTensor:indexOut toType:indexType name:nil];
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(ctx.op, 0);
    if (valueShape && valueOut) {
        valueOut = [ctx.graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(ctx.op, 1);
    if (indexShape && indexOut) {
        indexOut = [ctx.graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = valueOut;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

// Unified reduce handler - dispatches based on result count
static ProcessResult HandleReduce(HandlerContext& ctx) {
    if (ctx.op->getNumResults() > 1) {
        return HandleMultiResultReduce(ctx);
    }
    return HandleSingleResultReduce(ctx);
}
REGISTER_MPS_OP("stablehlo.reduce", HandleReduce);

// ---------------------------------------------------------------------------
// reduce_window: Tier 1 — cumulative patterns (cumsum, cumprod, cummax, cummin)
// ---------------------------------------------------------------------------

static ProcessResult HandleCumulativeReduceWindow(HandlerContext& ctx,
                                                  mlir::stablehlo::ReduceWindowOp rwOp,
                                                  MPSGraphTensor* input,
                                                  llvm::ArrayRef<int64_t> inputShape,
                                                  int64_t rank) {
    auto windowDims = rwOp.getWindowDimensions();
    auto paddingAttr = rwOp.getPaddingAttr();

    // Find the single cumulative axis: exactly one axis must have
    // window_dim == input_shape[axis], rest must be 1.
    int64_t cumAxis = -1;
    for (int64_t i = 0; i < rank; ++i) {
        if (windowDims[i] == 1)
            continue;
        if (windowDims[i] == inputShape[i] && cumAxis == -1) {
            cumAxis = i;
        } else {
            return ProcessResult::Error(
                "reduce_window: unsupported window dimensions (not a cumulative pattern)");
        }
    }
    if (cumAxis == -1)
        return ProcessResult::Error("reduce_window: no cumulative axis found");

    // Check padding to determine forward/reverse and exclusive/inclusive.
    int64_t padLow = 0, padHigh = 0;
    if (paddingAttr) {
        auto values = paddingAttr.getValues<int64_t>();
        padLow = values[{(uint64_t)cumAxis, 0}];
        padHigh = values[{(uint64_t)cumAxis, 1}];
        // All other axes must have zero padding.
        for (int64_t i = 0; i < rank; ++i) {
            if (i == cumAxis)
                continue;
            if (values[{(uint64_t)i, 0}] != 0 || values[{(uint64_t)i, 1}] != 0)
                return ProcessResult::Error(
                    "reduce_window: non-zero padding on non-cumulative axis");
        }
    }

    int64_t axisSize = inputShape[cumAxis];
    BOOL reverse = NO;
    BOOL exclusive = NO;

    if (padLow == axisSize - 1 && padHigh == 0) {
        reverse = NO;
        exclusive = NO;
    } else if (padLow == 0 && padHigh == axisSize - 1) {
        reverse = YES;
        exclusive = NO;
    } else if (padLow == axisSize && padHigh == -1) {
        reverse = NO;
        exclusive = YES;
    } else if (padLow == -1 && padHigh == axisSize) {
        reverse = YES;
        exclusive = YES;
    } else {
        return ProcessResult::Error(
            "reduce_window: unsupported padding pattern (not a cumulative op)");
    }

    std::string reductionType = GetReductionOpType(rwOp.getBody());

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [ctx.graph cumulativeSumWithTensor:input
                                               axis:(NSInteger)cumAxis
                                          exclusive:exclusive
                                            reverse:reverse
                                               name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [ctx.graph cumulativeProductWithTensor:input
                                                   axis:(NSInteger)cumAxis
                                              exclusive:exclusive
                                                reverse:reverse
                                                   name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [ctx.graph cumulativeMaximumWithTensor:input
                                                   axis:(NSInteger)cumAxis
                                              exclusive:exclusive
                                                reverse:reverse
                                                   name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [ctx.graph cumulativeMinimumWithTensor:input
                                                   axis:(NSInteger)cumAxis
                                              exclusive:exclusive
                                                reverse:reverse
                                                   name:nil];
    } else {
        return ProcessResult::Error("reduce_window: unsupported reduction type: " + reductionType);
    }

    return Result(ctx, result, "reduce_window");
}

// ---------------------------------------------------------------------------
// reduce_window: Tier 2 — pooling (max pool, sum pool via avg*count)
// ---------------------------------------------------------------------------

static ProcessResult HandlePoolingReduceWindow(HandlerContext& ctx,
                                               mlir::stablehlo::ReduceWindowOp rwOp,
                                               MPSGraphTensor* input,
                                               llvm::ArrayRef<int64_t> inputShape, int64_t rank) {
    auto windowDims = rwOp.getWindowDimensions();
    auto stridesOpt = rwOp.getWindowStrides();
    auto winDilOpt = rwOp.getWindowDilations();
    auto paddingAttr = rwOp.getPaddingAttr();

    // Collect per-axis attributes.
    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> winDil(rank, 1);
    std::vector<int64_t> padLow(rank, 0);
    std::vector<int64_t> padHigh(rank, 0);

    if (stridesOpt) {
        auto s = *stridesOpt;
        for (int64_t i = 0; i < rank; i++)
            strides[i] = s[i];
    }
    if (winDilOpt) {
        auto d = *winDilOpt;
        for (int64_t i = 0; i < rank; i++)
            winDil[i] = d[i];
    }
    if (paddingAttr) {
        auto vals = paddingAttr.getValues<int64_t>();
        for (int64_t i = 0; i < rank; i++) {
            padLow[i] = vals[{(uint64_t)i, 0}];
            padHigh[i] = vals[{(uint64_t)i, 1}];
        }
    }

    // All padding must be non-negative for pooling.
    for (int64_t i = 0; i < rank; i++) {
        if (padLow[i] < 0 || padHigh[i] < 0)
            return ProcessResult::Error(
                "reduce_window: negative padding not supported for pooling");
    }

    // Must have at least one spatial dim (window > 1).
    bool hasSpatial = false;
    for (int64_t i = 0; i < rank; i++) {
        if (windowDims[i] > 1)
            hasSpatial = true;
    }
    if (!hasSpatial)
        return ProcessResult::Error("reduce_window: no spatial dimensions for pooling");

    // Check reduction type — pooling supports max and add (sum via avg*count).
    std::string reductionType = GetReductionOpType(rwOp.getBody());
    bool isMax = (reductionType == "stablehlo.maximum");
    bool isAdd = (reductionType == "stablehlo.add");
    if (!isMax && !isAdd)
        return ProcessResult::Error("reduce_window: pooling supports maximum/add, got: " +
                                    reductionType);

    // MPS provides maxPooling4D / avgPooling4D which require exactly 4D input.
    // For rank <= 4: prepend size-1 dims. Rank > 4 not yet supported.
    if (rank > 4)
        return ProcessResult::Error("reduce_window: rank > 4 pooling not yet supported");

    // Build 4D arrays for the MPS pooling descriptor.
    int64_t pad4 = 4 - rank;
    NSMutableArray<NSNumber*>* kernelSizes = [NSMutableArray arrayWithCapacity:4];
    NSMutableArray<NSNumber*>* mpsStrides = [NSMutableArray arrayWithCapacity:4];
    NSMutableArray<NSNumber*>* dilationRates = [NSMutableArray arrayWithCapacity:4];
    // paddingValues: 8 elements [before_dim0, after_dim0, before_dim1, after_dim1, ...]
    NSMutableArray<NSNumber*>* paddingValues = [NSMutableArray arrayWithCapacity:8];
    NSMutableArray<NSNumber*>* reshapeShape = [NSMutableArray arrayWithCapacity:4];

    // Prepend trivial dims to reach 4D.
    for (int64_t i = 0; i < pad4; i++) {
        [kernelSizes addObject:@1];
        [mpsStrides addObject:@1];
        [dilationRates addObject:@1];
        [paddingValues addObject:@0];
        [paddingValues addObject:@0];
        [reshapeShape addObject:@1];
    }
    for (int64_t i = 0; i < rank; i++) {
        [kernelSizes addObject:@(windowDims[i])];
        [mpsStrides addObject:@(strides[i])];
        [dilationRates addObject:@(winDil[i])];
        [paddingValues addObject:@(padLow[i])];
        [paddingValues addObject:@(padHigh[i])];
        [reshapeShape addObject:@(inputShape[i])];
    }

    // Reshape input to 4D.
    MPSGraphTensor* input4D = [ctx.graph reshapeTensor:input withShape:reshapeShape name:nil];

    // Create the 4D pooling descriptor with explicit padding.
    MPSGraphPooling4DOpDescriptor* desc =
        [MPSGraphPooling4DOpDescriptor descriptorWithKernelSizes:kernelSizes
                                                         strides:mpsStrides
                                                   dilationRates:dilationRates
                                                   paddingValues:paddingValues
                                                    paddingStyle:MPSGraphPaddingStyleExplicit];

    MPSGraphTensor* result4D = nil;
    if (isMax) {
        result4D = [ctx.graph maxPooling4DWithSourceTensor:input4D descriptor:desc name:nil];
    } else {
        // Sum pooling = avg pooling * window_element_count.
        // With includeZeroPadToAverage=YES, avg = sum_of_window / K where
        // K = product of kernel sizes, so sum = avg * K.
        desc.includeZeroPadToAverage = YES;
        result4D = [ctx.graph avgPooling4DWithSourceTensor:input4D descriptor:desc name:nil];

        int64_t windowCount = 1;
        for (NSNumber* k in kernelSizes)
            windowCount *= k.longLongValue;

        MPSGraphTensor* countTensor = [ctx.graph constantWithScalar:(double)windowCount
                                                           dataType:result4D.dataType];
        result4D = [ctx.graph multiplicationWithPrimaryTensor:result4D
                                              secondaryTensor:countTensor
                                                         name:nil];
    }

    // Reshape back to the expected output shape.
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape)
        result4D = [ctx.graph reshapeTensor:result4D withShape:outputShape name:nil];

    return Result(ctx, result4D, "reduce_window");
}

// ---------------------------------------------------------------------------
// reduce_window: top-level dispatcher
// ---------------------------------------------------------------------------

static ProcessResult HandleReduceWindow(HandlerContext& ctx) {
    auto rwOp = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(ctx.op);
    if (!rwOp)
        return ProcessResult::Error("reduce_window: expected ReduceWindowOp");

    // Only support single-input / single-init / single-result reduce_window.
    if (rwOp.getInputs().size() != 1 || rwOp.getInitValues().size() != 1 ||
        rwOp->getNumResults() != 1)
        return ProcessResult::Error("reduce_window: only single-input reduce_window is supported");

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reduce_window: input tensor not found");

    // Validate the init value is the correct identity for the reduction type.
    // MPS pooling/cumulative ops assume identity padding; a non-identity init
    // would silently produce wrong results.
    std::string reductionType = GetReductionOpType(rwOp.getBody());
    mlir::Value initValue = rwOp.getInitValues()[0];
    if (auto* defOp = initValue.getDefiningOp()) {
        if (auto constOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(defOp)) {
            auto attr = constOp.getValue();
            if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
                if (denseAttr.isSplat()) {
                    auto elemType = denseAttr.getElementType();
                    bool valid = false;
                    if (mlir::isa<mlir::FloatType>(elemType)) {
                        double val = denseAttr.getSplatValue<llvm::APFloat>().convertToDouble();
                        if (reductionType == "stablehlo.add" && val == 0.0)
                            valid = true;
                        else if (reductionType == "stablehlo.multiply" && val == 1.0)
                            valid = true;
                        else if (reductionType == "stablehlo.maximum" && std::isinf(val) && val < 0)
                            valid = true;
                        else if (reductionType == "stablehlo.minimum" && std::isinf(val) && val > 0)
                            valid = true;
                    } else if (mlir::isa<mlir::IntegerType>(elemType)) {
                        int64_t val = denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
                        if (reductionType == "stablehlo.add" && val == 0)
                            valid = true;
                        else if (reductionType == "stablehlo.multiply" && val == 1)
                            valid = true;
                        // For integer max/min, the init should be the type's
                        // min/max value respectively. Accept any value here since
                        // verifying exact type bounds is complex for all int widths.
                        else if (reductionType == "stablehlo.maximum" ||
                                 reductionType == "stablehlo.minimum")
                            valid = true;
                    }
                    if (!valid)
                        return ProcessResult::Error(
                            "reduce_window: init value is not the identity for reduction type " +
                            reductionType);
                }
            }
        }
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(rwOp.getInputs()[0].getType());
    if (!inputType)
        return ProcessResult::Error("reduce_window: unranked input");
    auto inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    auto windowDims = rwOp.getWindowDimensions();
    auto stridesOpt = rwOp.getWindowStrides();
    auto baseDilOpt = rwOp.getBaseDilations();
    auto winDilOpt = rwOp.getWindowDilations();

    // Classify the pattern:
    // Cumulative: all strides=1, all dilations=1, exactly one axis has
    //   window_dim == input_shape[axis] and the rest have window_dim == 1.
    // Pooling: base_dilations all 1, non-negative padding, at least one
    //   axis with window > 1 (but not matching the cumulative pattern).
    bool allStridesOne = !stridesOpt || AllOnes(*stridesOpt);
    bool allBaseDilOne = !baseDilOpt || AllOnes(*baseDilOpt);
    bool allWinDilOne = !winDilOpt || AllOnes(*winDilOpt);

    // Check for cumulative pattern: exactly one axis where window == input_shape.
    bool isCumulative = false;
    if (allStridesOne && allBaseDilOne && allWinDilOne) {
        int64_t cumCount = 0;
        for (int64_t i = 0; i < rank; i++) {
            if (windowDims[i] == 1)
                continue;
            if (windowDims[i] == inputShape[i])
                cumCount++;
            else
                cumCount = -1;  // non-cumulative spatial dim
        }
        isCumulative = (cumCount == 1);
    }

    if (isCumulative)
        return HandleCumulativeReduceWindow(ctx, rwOp, input, inputShape, rank);

    // Pooling: base_dilations must be all 1.
    if (allBaseDilOne)
        return HandlePoolingReduceWindow(ctx, rwOp, input, inputShape, rank);

    return ProcessResult::Error("reduce_window: unsupported pattern (not cumulative or pooling)");
}
REGISTER_MPS_OP("stablehlo.reduce_window", HandleReduceWindow);

// stablehlo.return is a terminator used inside regions (e.g., reduce body)
// It's handled implicitly by parent operations, not executed directly
static ProcessResult HandleReturn(HandlerContext& ctx) {
    // This should never be called directly - it's handled by the parent operation
    // But we register it so it's not flagged as unsupported during module verification
    MPS_LOG_WARN("stablehlo.return should not be called directly\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", HandleReturn);

}  // namespace jax_mps
