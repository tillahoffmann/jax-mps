#ifndef ISSUE_URL_H
#define ISSUE_URL_H

#include <string>
#include <vector>

namespace jax_mps {

/// Build a full error message for unsupported ops, including links to the
/// GitHub issue template and CONTRIBUTING.md.
inline std::string UnsupportedOpsMessage(const std::vector<std::string>& op_names) {
    // Build comma-separated list of op names.
    std::string op_list;
    for (size_t i = 0; i < op_names.size(); i++) {
        if (i > 0)
            op_list += ", ";
        op_list += op_names[i];
    }

    // URL-encode dots in the first op name for query parameters.
    std::string encoded = op_names[0];
    for (size_t pos = 0; (pos = encoded.find('.', pos)) != std::string::npos; pos += 3) {
        encoded.replace(pos, 1, "%2E");
    }

    return "Unsupported operation(s): " + op_list +
           ". The MPS backend does not have a handler for these operations.\n\n"
           "File an issue: "
           "https://github.com/tillahoffmann/jax-mps/issues/new?template=missing-op.yml"
           "&title=Missing+op:+" +
           encoded + "&op-name=" + encoded +
           "\n"
           "Add it yourself: https://github.com/tillahoffmann/jax-mps/blob/main/CONTRIBUTING.md";
}

}  // namespace jax_mps

#endif  // ISSUE_URL_H
