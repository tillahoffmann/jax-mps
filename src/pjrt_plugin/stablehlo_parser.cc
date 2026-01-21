// MLIR-based StableHLO parser
// Uses proper MLIR/StableHLO libraries for robust parsing

#include "pjrt_plugin/stablehlo_parser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <unordered_map>

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
    if (type.isF32()) return "f32";
    if (type.isF64()) return "f64";
    if (type.isF16()) return "f16";
    if (type.isBF16()) return "bf16";
    if (type.isInteger(32)) return "i32";
    if (type.isInteger(64)) return "i64";
    if (type.isInteger(1)) return "i1";
    if (type.isInteger(8)) return "i8";
    if (type.isInteger(16)) return "i16";

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

    if (name == "stablehlo.add") return {OpKind::Add, name_str};
    if (name == "stablehlo.multiply") return {OpKind::Multiply, name_str};
    if (name == "stablehlo.subtract") return {OpKind::Subtract, name_str};
    if (name == "stablehlo.divide") return {OpKind::Divide, name_str};
    if (name == "stablehlo.maximum") return {OpKind::Maximum, name_str};
    if (name == "stablehlo.minimum") return {OpKind::Minimum, name_str};
    if (name == "stablehlo.tanh") return {OpKind::Tanh, name_str};
    if (name == "stablehlo.exponential") return {OpKind::Exp, name_str};
    if (name == "stablehlo.log") return {OpKind::Log, name_str};
    if (name == "stablehlo.negate") return {OpKind::Negate, name_str};
    if (name == "stablehlo.abs") return {OpKind::Abs, name_str};
    if (name == "stablehlo.dot") return {OpKind::Dot, name_str};
    if (name == "stablehlo.dot_general") return {OpKind::DotGeneral, name_str};
    if (name == "stablehlo.reshape") return {OpKind::Reshape, name_str};
    if (name == "stablehlo.transpose") return {OpKind::Transpose, name_str};
    if (name == "stablehlo.broadcast") return {OpKind::Broadcast, name_str};
    if (name == "stablehlo.broadcast_in_dim") return {OpKind::BroadcastInDim, name_str};
    if (name == "stablehlo.reduce") return {OpKind::Reduce, name_str};
    if (name == "stablehlo.convert") return {OpKind::Convert, name_str};
    if (name == "stablehlo.constant") return {OpKind::Constant, name_str};
    if (name == "func.return" || name == "return") return {OpKind::Return, name_str};
    if (name == "func.call" || name == "call") return {OpKind::Call, name_str};

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
        if (mlir::isa<mlir::func::FuncOp>(op)) return;

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
            // Map this operation's result to its name
            valueToName[op->getResult(0).getAsOpaquePointer()] = shloOp.name;
        }

        // Get operands
        for (mlir::Value operand : op->getOperands()) {
            Operand opnd;
            // Try to get a meaningful name
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                opnd.name = "%arg" + std::to_string(blockArg.getArgNumber());
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
            if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
                    op->getResult(0).getType())) {
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
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(data, size),
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
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(data, size),
        "stablehlo_bytecode",
        false);

    // Deserialize
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {
        // Try regular parsing
        llvm::SourceMgr sourceMgr;
        auto bufferCopy = llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(data, size), "stablehlo_bytecode", false);
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
