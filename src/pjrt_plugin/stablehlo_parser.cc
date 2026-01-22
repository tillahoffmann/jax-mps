// MLIR-based StableHLO parser
// Uses proper MLIR/StableHLO libraries for robust parsing

#include "pjrt_plugin/stablehlo_parser.h"

#include <sstream>
#include <unordered_map>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mps {

namespace {

// Register all dialects needed for StableHLO parsing
void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    // Allow unknown dialects (e.g., sdy/Shardy for sharding) to pass through
    context.allowUnregisteredDialects();
}

// Convert MLIR element type to string
std::string elementTypeToString(mlir::Type type) {
    if (type.isF32())
        return "f32";
    if (type.isF64())
        return "f64";
    if (type.isF16())
        return "f16";
    if (type.isBF16())
        return "bf16";

    // Handle integer types - check for signedness
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        unsigned width = intType.getWidth();
        bool isUnsigned = intType.isUnsigned();
        std::string prefix = isUnsigned ? "ui" : "i";
        // Also treat signless as signed for compatibility
        if (intType.isSignless()) {
            // StableHLO uses signless integers, which we treat as signed
            prefix = "si";
        }
        return prefix + std::to_string(width);
    }

    // Default fallback
    std::string str;
    llvm::raw_string_ostream os(str);
    type.print(os);
    return str;
}

// Convert MLIR ranked tensor type to TensorType
TensorType convertTensorType(mlir::RankedTensorType tensorType) {
    TensorType result;

    // Get shape
    for (int64_t dim : tensorType.getShape()) {
        result.shape.push_back(dim);
    }

    // Get element type
    result.element_type = elementTypeToString(tensorType.getElementType());

    return result;
}

// Map StableHLO operation to OpKind
// Returns the op name string for use in error messages when Unknown
std::pair<OpKind, std::string> getOpKindWithName(mlir::Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    std::string name_str = name.str();

    if (name == "stablehlo.add")
        return {OpKind::Add, name_str};
    if (name == "stablehlo.multiply")
        return {OpKind::Multiply, name_str};
    if (name == "stablehlo.subtract")
        return {OpKind::Subtract, name_str};
    if (name == "stablehlo.divide")
        return {OpKind::Divide, name_str};
    if (name == "stablehlo.maximum")
        return {OpKind::Maximum, name_str};
    if (name == "stablehlo.minimum")
        return {OpKind::Minimum, name_str};
    if (name == "stablehlo.tanh")
        return {OpKind::Tanh, name_str};
    if (name == "stablehlo.exponential")
        return {OpKind::Exp, name_str};
    if (name == "stablehlo.log")
        return {OpKind::Log, name_str};
    if (name == "stablehlo.negate")
        return {OpKind::Negate, name_str};
    if (name == "stablehlo.abs")
        return {OpKind::Abs, name_str};
    if (name == "stablehlo.sqrt")
        return {OpKind::Sqrt, name_str};
    if (name == "stablehlo.log_plus_one")
        return {OpKind::LogPlusOne, name_str};
    if (name == "stablehlo.compare")
        return {OpKind::Compare, name_str};
    if (name == "stablehlo.select")
        return {OpKind::Select, name_str};
    if (name == "stablehlo.dot")
        return {OpKind::Dot, name_str};
    if (name == "stablehlo.dot_general")
        return {OpKind::DotGeneral, name_str};
    if (name == "stablehlo.reshape")
        return {OpKind::Reshape, name_str};
    if (name == "stablehlo.transpose")
        return {OpKind::Transpose, name_str};
    if (name == "stablehlo.broadcast")
        return {OpKind::Broadcast, name_str};
    if (name == "stablehlo.broadcast_in_dim")
        return {OpKind::BroadcastInDim, name_str};
    if (name == "stablehlo.reduce")
        return {OpKind::Reduce, name_str};
    if (name == "stablehlo.convert")
        return {OpKind::Convert, name_str};
    if (name == "stablehlo.constant")
        return {OpKind::Constant, name_str};
    if (name == "func.return" || name == "return")
        return {OpKind::Return, name_str};
    if (name == "func.call" || name == "call")
        return {OpKind::Call, name_str};

    // Bitwise operations (needed for RNG)
    if (name == "stablehlo.and")
        return {OpKind::And, name_str};
    if (name == "stablehlo.or")
        return {OpKind::Or, name_str};
    if (name == "stablehlo.xor")
        return {OpKind::Xor, name_str};
    if (name == "stablehlo.shift_right_logical")
        return {OpKind::ShiftRightLogical, name_str};
    if (name == "stablehlo.shift_left")
        return {OpKind::ShiftLeft, name_str};

    // Other operations
    if (name == "stablehlo.concatenate")
        return {OpKind::Concatenate, name_str};
    if (name == "stablehlo.slice")
        return {OpKind::Slice, name_str};
    if (name == "stablehlo.dynamic_slice")
        return {OpKind::DynamicSlice, name_str};
    if (name == "stablehlo.iota")
        return {OpKind::Iota, name_str};
    if (name == "stablehlo.bitcast_convert")
        return {OpKind::BitcastConvert, name_str};
    if (name == "stablehlo.custom_call")
        return {OpKind::CustomCall, name_str};

    return {OpKind::Unknown, name_str};
}

// Parse a function from MLIR
// unsupported_ops collects the names of any unsupported operations encountered
bool parseFunction(mlir::func::FuncOp funcOp, StableHLOFunction& result,
                   std::vector<std::string>& unsupported_ops) {
    result.name = funcOp.getName().str();

    // Parse argument types
    for (mlir::Type argType : funcOp.getArgumentTypes()) {
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(argType)) {
            result.arg_types.push_back(convertTensorType(tensorType));
        }
    }

    // Parse result types
    for (mlir::Type resultType : funcOp.getResultTypes()) {
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
            result.result_types.push_back(convertTensorType(tensorType));
        }
    }

    // Parse operations
    // Track mapping from MLIR Value to result name
    std::unordered_map<void*, std::string> valueToName;
    int opCounter = 0;
    funcOp.walk([&](mlir::Operation* op) {
        // Skip the function op itself
        if (mlir::isa<mlir::func::FuncOp>(op))
            return;

        StableHLOOp shloOp;
        auto [kind, op_name] = getOpKindWithName(op);
        shloOp.kind = kind;
        shloOp.op_name = op_name;

        // Track unsupported operations
        if (kind == OpKind::Unknown) {
            // Check if we've already recorded this op type
            bool already_recorded = false;
            for (const auto& recorded : unsupported_ops) {
                if (recorded == op_name) {
                    already_recorded = true;
                    break;
                }
            }
            if (!already_recorded) {
                unsupported_ops.push_back(op_name);
            }
        }

        // Generate result name and track it
        if (op->getNumResults() > 0) {
            shloOp.name = "%" + std::to_string(opCounter++);
            // Map all operation results to their names
            // For single-result ops: %N
            // For multi-result ops: %N.0, %N.1, etc.
            if (op->getNumResults() == 1) {
                valueToName[op->getResult(0).getAsOpaquePointer()] = shloOp.name;
            } else {
                for (unsigned i = 0; i < op->getNumResults(); ++i) {
                    std::string resultName = shloOp.name + "." + std::to_string(i);
                    valueToName[op->getResult(i).getAsOpaquePointer()] = resultName;
                }
            }
        }

        // Get operands
        for (mlir::Value operand : op->getOperands()) {
            Operand opnd;
            // Try to get a meaningful name
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                // Check which block the argument belongs to
                mlir::Block* parentBlock = blockArg.getOwner();
                mlir::Operation* parentOp = parentBlock->getParentOp();
                if (mlir::isa<mlir::func::FuncOp>(parentOp)) {
                    // This is a function argument
                    opnd.name = "%arg" + std::to_string(blockArg.getArgNumber());
                } else {
                    // This is a block argument from a nested region (e.g., reduce body)
                    // For now, mark as unknown since we don't handle nested regions yet
                    opnd.name = "%nested_arg" + std::to_string(blockArg.getArgNumber());
                }
            } else {
                // Look up the operand's name from our mapping
                auto it = valueToName.find(operand.getAsOpaquePointer());
                if (it != valueToName.end()) {
                    opnd.name = it->second;
                } else {
                    opnd.name = "%unknown";
                }
            }
            shloOp.operands.push_back(opnd);
        }

        // Get result type
        if (op->getNumResults() > 0) {
            if (auto tensorType =
                    mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())) {
                shloOp.result_type = convertTensorType(tensorType);
            }
        }

        // Handle specific operation attributes
        if (auto dotGeneral = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
            auto dimNumbers = dotGeneral.getDotDimensionNumbers();
            for (int64_t d : dimNumbers.getLhsBatchingDimensions()) {
                shloOp.lhs_batching_dims.push_back(d);
            }
            for (int64_t d : dimNumbers.getRhsBatchingDimensions()) {
                shloOp.rhs_batching_dims.push_back(d);
            }
            for (int64_t d : dimNumbers.getLhsContractingDimensions()) {
                shloOp.lhs_contracting_dims.push_back(d);
            }
            for (int64_t d : dimNumbers.getRhsContractingDimensions()) {
                shloOp.rhs_contracting_dims.push_back(d);
            }
        }

        if (auto broadcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
            for (int64_t d : broadcast.getBroadcastDimensions()) {
                shloOp.broadcast_dimensions.push_back(d);
            }
        }

        if (auto transpose = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
            for (int64_t d : transpose.getPermutation()) {
                shloOp.permutation.push_back(d);
            }
        }

        if (auto concat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
            shloOp.concatenate_dimension = concat.getDimension();
        }

        if (auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
            for (int64_t d : slice.getStartIndices()) {
                shloOp.slice_starts.push_back(d);
            }
            for (int64_t d : slice.getLimitIndices()) {
                shloOp.slice_limits.push_back(d);
            }
            for (int64_t d : slice.getStrides()) {
                shloOp.slice_strides.push_back(d);
            }
        }

        if (auto dynSlice = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op)) {
            for (int64_t d : dynSlice.getSliceSizes()) {
                shloOp.slice_sizes.push_back(d);
            }
        }

        if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
            shloOp.iota_dimension = iota.getIotaDimension();
        }

        if (auto compare = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
            auto direction = compare.getComparisonDirection();
            shloOp.compare_direction =
                mlir::stablehlo::stringifyComparisonDirection(direction).str();
        }

        if (auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
            shloOp.custom_call_target = customCall.getCallTargetName().str();
        }

        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            shloOp.call_target = callOp.getCallee().str();
        }

        // Handle constant operations - extract the actual constant value
        if (auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
            auto value = constantOp.getValue();
            if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
                auto elemType = denseAttr.getElementType();

                // Check if it's a splat (single value broadcast to all elements)
                if (denseAttr.isSplat()) {
                    shloOp.is_scalar_constant = true;
                    if (elemType.isF32()) {
                        shloOp.constant_scalar = denseAttr.getSplatValue<float>();
                    } else if (elemType.isF64()) {
                        shloOp.constant_scalar =
                            static_cast<float>(denseAttr.getSplatValue<double>());
                    } else if (elemType.isF16()) {
                        // F16 values come as APFloat, convert to float
                        auto apVal = denseAttr.getSplatValue<llvm::APFloat>();
                        shloOp.constant_scalar = apVal.convertToFloat();
                    } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                        // Handle all integer types via APInt
                        auto apInt = denseAttr.getSplatValue<llvm::APInt>();
                        shloOp.constant_scalar_raw = apInt.getZExtValue();
                        shloOp.uses_raw_data = true;
                    }
                } else {
                    // Non-splat dense constant - use raw data approach for all types
                    auto rawData = denseAttr.getRawData();
                    shloOp.constant_raw.assign(rawData.begin(), rawData.end());
                    shloOp.uses_raw_data = true;
                }
            }
        }

        result.ops.push_back(std::move(shloOp));
    });

    return true;
}

// Parse module from MLIR module op
bool parseModule(mlir::ModuleOp moduleOp, StableHLOModule& module) {
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        StableHLOFunction func;
        if (parseFunction(funcOp, func, module.unsupported_ops)) {
            // Set entry function (usually "main")
            if (func.name == "main") {
                module.entry_function = func.name;
            }
            module.functions.push_back(std::move(func));
        }
    });

    // Default to first function if no "main"
    if (module.entry_function.empty() && !module.functions.empty()) {
        module.entry_function = module.functions[0].name;
    }

    return !module.functions.empty();
}

}  // namespace

bool parseStableHLOBytecode(const char* data, size_t size, StableHLOModule& module) {
    mlir::MLIRContext context;
    registerDialects(context);

    // Create memory buffer from data
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                   /*BufferName=*/"stablehlo_bytecode",
                                                   /*RequiresNullTerminator=*/false);

    // Try to deserialize as portable artifact first (handles VHLO versioning)
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {
        // Fall back to regular bytecode reading
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!moduleOp) {
        return false;
    }

    return parseModule(*moduleOp, module);
}

bool parseStableHLOText(const std::string& text, StableHLOModule& module) {
    mlir::MLIRContext context;
    registerDialects(context);

    // Parse text format
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::parseSourceString<mlir::ModuleOp>(text, &context);

    if (!moduleOp) {
        return false;
    }

    return parseModule(*moduleOp, module);
}

std::string bytecodeToText(const char* data, size_t size) {
    mlir::MLIRContext context;
    registerDialects(context);

    // Create memory buffer
    auto buffer =
        llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);

    // Deserialize
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {
        // Try regular parsing
        llvm::SourceMgr sourceMgr;
        auto bufferCopy = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                           "stablehlo_bytecode", false);
        sourceMgr.AddNewSourceBuffer(std::move(bufferCopy), llvm::SMLoc());
        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!moduleOp) {
        return "";
    }

    // Print to string
    std::string result;
    llvm::raw_string_ostream os(result);
    moduleOp->print(os);
    return result;
}

}  // namespace mps
