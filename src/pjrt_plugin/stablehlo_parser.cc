#include "pjrt_plugin/stablehlo_parser.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <fstream>
#include <unistd.h>

namespace mps {

namespace {

// Parse a tensor type string like "tensor<2x3xf32>"
TensorType parseTensorTypeString(const std::string& str) {
    TensorType result;

    // Match tensor<...xf32> or tensor<f32>
    std::regex tensor_regex(R"(tensor<([^>]*)>)");
    std::smatch match;
    if (!std::regex_search(str, match, tensor_regex)) {
        return result;
    }

    std::string inner = match[1].str();

    // Find the element type (last part after 'x' or the whole thing)
    size_t last_x = inner.rfind('x');
    std::string elem_type;
    std::string shape_str;

    if (last_x != std::string::npos) {
        elem_type = inner.substr(last_x + 1);
        shape_str = inner.substr(0, last_x);
    } else {
        elem_type = inner;
        shape_str = "";
    }

    result.element_type = elem_type;

    // Parse shape dimensions
    if (!shape_str.empty()) {
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, 'x')) {
            if (dim == "?") {
                result.shape.push_back(-1);  // Dynamic dimension
            } else {
                result.shape.push_back(std::stoll(dim));
            }
        }
    }

    return result;
}

// Map operation name to OpKind
OpKind parseOpKind(const std::string& op_name) {
    if (op_name == "stablehlo.add") return OpKind::Add;
    if (op_name == "stablehlo.multiply") return OpKind::Multiply;
    if (op_name == "stablehlo.subtract") return OpKind::Subtract;
    if (op_name == "stablehlo.divide") return OpKind::Divide;
    if (op_name == "stablehlo.tanh") return OpKind::Tanh;
    if (op_name == "stablehlo.exponential") return OpKind::Exp;
    if (op_name == "stablehlo.log") return OpKind::Log;
    if (op_name == "stablehlo.negate") return OpKind::Negate;
    if (op_name == "stablehlo.abs") return OpKind::Abs;
    if (op_name == "stablehlo.dot") return OpKind::Dot;
    if (op_name == "stablehlo.dot_general") return OpKind::DotGeneral;
    if (op_name == "stablehlo.reshape") return OpKind::Reshape;
    if (op_name == "stablehlo.transpose") return OpKind::Transpose;
    if (op_name == "stablehlo.broadcast") return OpKind::Broadcast;
    if (op_name == "stablehlo.broadcast_in_dim") return OpKind::BroadcastInDim;
    if (op_name == "stablehlo.reduce") return OpKind::Reduce;
    if (op_name == "stablehlo.convert") return OpKind::Convert;
    if (op_name == "stablehlo.constant") return OpKind::Constant;
    if (op_name == "func.return" || op_name == "return") return OpKind::Return;
    if (op_name == "func.call" || op_name == "call") return OpKind::Call;
    return OpKind::Unknown;
}

// Simple text-based parser for MLIR/StableHLO
// This parses the text form that we get from deserializing the bytecode
class TextParser {
public:
    explicit TextParser(const std::string& text) : text_(text), pos_(0) {}

    bool parse(StableHLOModule& module) {
        // Skip to module
        if (!skipTo("module")) return false;

        // Find all functions
        while (skipTo("func.func")) {
            StableHLOFunction func;
            if (parseFunction(func)) {
                if (func.name == "main" || func.name == "@main") {
                    module.entry_function = func.name;
                }
                module.functions.push_back(std::move(func));
            }
        }

        return !module.functions.empty();
    }

private:
    bool parseFunction(StableHLOFunction& func) {
        // Parse function signature: @name(%arg0: tensor<...>, ...) -> (tensor<...>)
        skipWhitespace();

        // Get visibility (public/private)
        if (peek() == 'p') {
            if (text_.substr(pos_, 6) == "public") {
                pos_ += 6;
                skipWhitespace();
            } else if (text_.substr(pos_, 7) == "private") {
                pos_ += 7;
                skipWhitespace();
            }
        }

        // Get function name
        if (peek() != '@') return false;
        pos_++;  // skip @

        std::string name;
        while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '_')) {
            name += text_[pos_++];
        }
        func.name = name;

        // Parse arguments
        skipWhitespace();
        if (peek() != '(') return false;
        pos_++;  // skip (

        while (peek() != ')') {
            skipWhitespace();
            // Skip %argN:
            if (peek() == '%') {
                while (pos_ < text_.size() && text_[pos_] != ':') pos_++;
                pos_++;  // skip :
                skipWhitespace();
            }

            // Parse tensor type
            std::string type_str;
            int depth = 0;
            while (pos_ < text_.size()) {
                char c = text_[pos_];
                if (c == '<') depth++;
                if (c == '>') depth--;
                if ((c == ',' || c == ')') && depth == 0) break;
                type_str += c;
                pos_++;
            }

            // Remove attributes like {jax.arg_info = ...}
            size_t brace = type_str.find('{');
            if (brace != std::string::npos) {
                type_str = type_str.substr(0, brace);
            }

            // Trim whitespace
            while (!type_str.empty() && isspace(type_str.back())) {
                type_str.pop_back();
            }

            if (!type_str.empty()) {
                func.arg_types.push_back(parseTensorTypeString(type_str));
            }

            if (peek() == ',') pos_++;
            skipWhitespace();
        }
        pos_++;  // skip )

        // Parse return type
        skipWhitespace();
        if (text_.substr(pos_, 2) == "->") {
            pos_ += 2;
            skipWhitespace();

            // Parse return types (may be wrapped in parentheses)
            if (peek() == '(') {
                pos_++;
                while (peek() != ')') {
                    skipWhitespace();
                    std::string type_str;
                    int depth = 0;
                    while (pos_ < text_.size()) {
                        char c = text_[pos_];
                        if (c == '<') depth++;
                        if (c == '>') depth--;
                        if ((c == ',' || c == ')') && depth == 0) break;
                        type_str += c;
                        pos_++;
                    }

                    // Remove attributes
                    size_t brace = type_str.find('{');
                    if (brace != std::string::npos) {
                        type_str = type_str.substr(0, brace);
                    }
                    while (!type_str.empty() && isspace(type_str.back())) {
                        type_str.pop_back();
                    }

                    if (!type_str.empty()) {
                        func.result_types.push_back(parseTensorTypeString(type_str));
                    }

                    if (peek() == ',') pos_++;
                    skipWhitespace();
                }
                pos_++;  // skip )
            } else {
                // Single return type
                std::string type_str;
                int depth = 0;
                while (pos_ < text_.size()) {
                    char c = text_[pos_];
                    if (c == '<') depth++;
                    if (c == '>') depth--;
                    if ((c == '{' || isspace(c)) && depth == 0) break;
                    type_str += c;
                    pos_++;
                }
                if (!type_str.empty()) {
                    func.result_types.push_back(parseTensorTypeString(type_str));
                }
            }
        }

        // Find function body
        if (!skipTo("{")) return false;
        pos_++;  // skip {

        // Parse operations until closing brace
        int brace_depth = 1;
        while (brace_depth > 0 && pos_ < text_.size()) {
            skipWhitespace();

            // Check for nested braces
            if (peek() == '{') {
                brace_depth++;
                pos_++;
                continue;
            }
            if (peek() == '}') {
                brace_depth--;
                pos_++;
                continue;
            }

            // Try to parse an operation
            StableHLOOp op;
            if (parseOp(op)) {
                func.ops.push_back(std::move(op));
            } else {
                // Skip to next line
                while (pos_ < text_.size() && text_[pos_] != '\n') pos_++;
                if (pos_ < text_.size()) pos_++;
            }
        }

        return true;
    }

    bool parseOp(StableHLOOp& op) {
        skipWhitespace();

        // Check for result assignment: %name = op
        std::string result_name;
        if (peek() == '%') {
            pos_++;
            while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '_')) {
                result_name += text_[pos_++];
            }
            skipWhitespace();
            if (peek() != '=') {
                // This might be a return statement with operands
                pos_ -= result_name.size() + 1;
                result_name.clear();
            } else {
                pos_++;  // skip =
                skipWhitespace();
            }
        }

        // Parse operation name
        std::string op_name;
        while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '.' || text_[pos_] == '_')) {
            op_name += text_[pos_++];
        }

        if (op_name.empty()) return false;

        op.kind = parseOpKind(op_name);
        op.name = result_name.empty() ? "" : "%" + result_name;

        // Parse operands
        skipWhitespace();
        while (peek() == '%') {
            pos_++;
            std::string operand_name;
            while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '_')) {
                operand_name += text_[pos_++];
            }
            op.operands.push_back({"%" + operand_name});
            skipWhitespace();
            if (peek() == ',') {
                pos_++;
                skipWhitespace();
            }
        }

        // Parse call target for func.call
        if (op.kind == OpKind::Call && peek() == '@') {
            pos_++;
            std::string call_target;
            while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '_')) {
                call_target += text_[pos_++];
            }
            // Store call target (we'd need to extend the struct, but for now skip)
            skipWhitespace();
            if (peek() == '(') {
                pos_++;
                while (peek() == '%') {
                    pos_++;
                    std::string operand_name;
                    while (pos_ < text_.size() && (isalnum(text_[pos_]) || text_[pos_] == '_')) {
                        operand_name += text_[pos_++];
                    }
                    op.operands.push_back({"%" + operand_name});
                    skipWhitespace();
                    if (peek() == ',') {
                        pos_++;
                        skipWhitespace();
                    }
                }
                if (peek() == ')') pos_++;
            }
        }

        // Skip to result type (after ':')
        while (pos_ < text_.size() && text_[pos_] != ':' && text_[pos_] != '\n') {
            pos_++;
        }

        if (peek() == ':') {
            pos_++;
            skipWhitespace();

            // Parse result type
            std::string type_str;
            int depth = 0;
            while (pos_ < text_.size()) {
                char c = text_[pos_];
                if (c == '<' || c == '(') depth++;
                if (c == '>' || c == ')') depth--;
                if (c == '\n' && depth == 0) break;
                type_str += c;
                pos_++;
            }

            // Extract the result type (after '->' if present)
            size_t arrow = type_str.find("->");
            if (arrow != std::string::npos) {
                type_str = type_str.substr(arrow + 2);
            }

            // Trim and parse
            while (!type_str.empty() && isspace(type_str.front())) {
                type_str.erase(0, 1);
            }
            while (!type_str.empty() && isspace(type_str.back())) {
                type_str.pop_back();
            }

            // Handle tuple types by taking first element
            if (type_str.front() == '(') {
                type_str = type_str.substr(1);
                size_t comma = type_str.find(',');
                size_t paren = type_str.find(')');
                if (comma != std::string::npos && comma < paren) {
                    type_str = type_str.substr(0, comma);
                } else if (paren != std::string::npos) {
                    type_str = type_str.substr(0, paren);
                }
            }

            op.result_type = parseTensorTypeString(type_str);
        }

        return true;
    }

    void skipWhitespace() {
        while (pos_ < text_.size() && isspace(text_[pos_])) {
            pos_++;
        }
    }

    bool skipTo(const std::string& pattern) {
        size_t found = text_.find(pattern, pos_);
        if (found == std::string::npos) return false;
        pos_ = found + pattern.size();
        return true;
    }

    char peek() const {
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    const std::string& text_;
    size_t pos_;
};

}  // namespace

// Check if data looks like MLIR text (starts with "module" or similar text)
static bool looksLikeText(const char* data, size_t size) {
    if (size < 6) return false;
    // Check for common MLIR text starts
    return (strncmp(data, "module", 6) == 0 ||
            strncmp(data, "func", 4) == 0 ||
            strncmp(data, "// ", 3) == 0 ||
            strncmp(data, "#", 1) == 0);
}

std::string bytecodeToText(const char* data, size_t size) {
    // Check if it's already text format
    if (looksLikeText(data, size)) {
        return std::string(data, size);
    }

    std::string result;

    // Write bytecode to temporary file
    char temp_input[] = "/tmp/stablehlo_input_XXXXXX";
    int input_fd = mkstemp(temp_input);
    if (input_fd < 0) {
        fprintf(stderr, "[MPS] Failed to create temp input file\n");
        return result;
    }

    ssize_t written = write(input_fd, data, size);
    close(input_fd);

    if (written != static_cast<ssize_t>(size)) {
        fprintf(stderr, "[MPS] Failed to write bytecode to temp file\n");
        unlink(temp_input);
        return result;
    }

    // Create temp output file
    char temp_output[] = "/tmp/stablehlo_output_XXXXXX";
    int output_fd = mkstemp(temp_output);
    if (output_fd < 0) {
        fprintf(stderr, "[MPS] Failed to create temp output file\n");
        unlink(temp_input);
        return result;
    }
    close(output_fd);

    // Run stablehlo-translate to convert bytecode to text
    // The tool should be in our build directory
    std::string stablehlo_translate = "/Users/till/git/jax-mps/third_party/stablehlo/stablehlo-build/bin/stablehlo-translate";

    std::string cmd = stablehlo_translate + " --deserialize < " + std::string(temp_input) + " > " + std::string(temp_output) + " 2>/dev/null";

    int ret = system(cmd.c_str());

    if (ret != 0) {
        // Try mlir-opt as fallback (silently)
        std::string mlir_opt = "/Users/till/git/jax-mps/third_party/stablehlo/llvm-build/bin/mlir-opt";
        cmd = mlir_opt + " " + std::string(temp_input) + " > " + std::string(temp_output) + " 2>/dev/null";
        ret = system(cmd.c_str());
    }

    // Read output
    std::ifstream output_file(temp_output);
    if (output_file.is_open()) {
        std::stringstream buffer;
        buffer << output_file.rdbuf();
        result = buffer.str();
        output_file.close();
    }

    // Cleanup
    unlink(temp_input);
    unlink(temp_output);

    return result;
}

bool parseStableHLOBytecode(const char* data, size_t size, StableHLOModule& module) {
    // First convert bytecode to text
    std::string text = bytecodeToText(data, size);

    if (text.empty()) {
        // Silently fail - caller will use identity fallback
        return false;
    }

    // Parse the text
    return parseStableHLOText(text, module);
}

bool parseStableHLOText(const std::string& text, StableHLOModule& module) {
    TextParser parser(text);
    return parser.parse(module);
}

}  // namespace mps
