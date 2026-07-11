#include "pjrt_plugin/ops/metal_kernel_lib.h"

#include <stdexcept>
#include <tuple>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

MTL::DataType to_mtl_type(MklConstant::Type t) {
    switch (t) {
        case MklConstant::kBool:
            return MTL::DataType::DataTypeBool;
        case MklConstant::kInt:
            return MTL::DataType::DataTypeInt;
        case MklConstant::kUint:
            return MTL::DataType::DataTypeUInt;
        case MklConstant::kFloat:
            return MTL::DataType::DataTypeFloat;
    }
    return MTL::DataType::DataTypeBool;
}

// Multi-output primitive that dispatches one kernel from a precompiled metallib.
class MetalKernelLibKernel : public Primitive {
public:
    MetalKernelLibKernel(Stream s, std::string libpath, std::string kname, std::string hash_name,
                         std::array<int, 3> grid, std::array<int, 3> tg, bool by_threadgroups,
                         std::vector<MklBuffer> buffers, std::vector<MklConstant> constants)
        : Primitive(s),
          libpath_(std::move(libpath)),
          kname_(std::move(kname)),
          hash_name_(std::move(hash_name)),
          grid_(grid),
          tg_(tg),
          by_threadgroups_(by_threadgroups),
          buffers_(std::move(buffers)),
          constants_(std::move(constants)) {}

    void eval_cpu(const std::vector<array>& /*inputs*/, std::vector<array>& /*outputs*/) override {
        throw std::runtime_error("mps.metal_kernel_lib has no CPU implementation");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

    DEFINE_NAME(MetalKernelLib)
    bool is_equivalent(const Primitive& other) const override {
        const auto& o = static_cast<const MetalKernelLibKernel&>(other);
        return libpath_ == o.libpath_ && kname_ == o.kname_ && hash_name_ == o.hash_name_ &&
               grid_ == o.grid_ && tg_ == o.tg_ && by_threadgroups_ == o.by_threadgroups_ &&
               buffers_ == o.buffers_ && constants_ == o.constants_;
    }

private:
    std::string libpath_;
    std::string kname_;
    std::string hash_name_;
    std::array<int, 3> grid_;
    std::array<int, 3> tg_;
    bool by_threadgroups_;
    std::vector<MklBuffer> buffers_;
    std::vector<MklConstant> constants_;
};

void MetalKernelLibKernel::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    const auto& s = stream();
    auto& d = metal::device(s.device);

    // Make every input fully contiguous so the kernel can index with plain
    // strides. Reserve so &copies.back() stays valid across push_backs.
    std::vector<array> copies;
    copies.reserve(inputs.size());
    std::vector<const array*> in_ptrs;
    in_ptrs.reserve(inputs.size());
    for (const auto& a : inputs) {
        if (a.flags().row_contiguous) {
            in_ptrs.push_back(&a);
        } else {
            copies.push_back(contiguous_copy_gpu(a, s));
            in_ptrs.push_back(&copies.back());
        }
    }
    // Outputs are constructed row-contiguous by make_arrays; just back them.
    for (auto& out : outputs) {
        out.set_data(allocator::malloc(out.nbytes()));
    }

    // Specialize the pipeline on the requested function constants. The value
    // storage lives in constants_ (owned by this primitive), so the pointers we
    // hand to get_kernel stay valid through the call.
    metal::MTLFCList fclist;
    fclist.reserve(constants_.size());
    for (const auto& c : constants_) {
        fclist.emplace_back(c.value.data(), to_mtl_type(c.type),
                            static_cast<NS::UInteger>(c.index));
    }

    // Cache the library under its path so distinct metallibs never clash and the
    // same file is only mapped once.
    auto* lib = d.get_library(libpath_, libpath_);
    auto* kernel = d.get_kernel(kname_, lib, hash_name_.empty() ? kname_ : hash_name_, fclist);
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    if (buffers_.empty()) {
        // Default positional layout: inputs 0..N-1, outputs N..N+M-1.
        int idx = 0;
        for (const auto* a : in_ptrs) {
            enc.set_input_array(*a, idx++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, idx++);
        }
    } else {
        for (const auto& b : buffers_) {
            switch (b.kind) {
                case MklBuffer::kInput:
                    if (b.arg < 0 || static_cast<size_t>(b.arg) >= in_ptrs.size())
                        throw std::runtime_error("metal_kernel_lib: input buffer arg out of range");
                    enc.set_input_array(*in_ptrs[b.arg], b.slot);
                    break;
                case MklBuffer::kOutput:
                    if (b.arg < 0 || static_cast<size_t>(b.arg) >= outputs.size())
                        throw std::runtime_error(
                            "metal_kernel_lib: output buffer arg out of range");
                    enc.set_output_array(outputs[b.arg], b.slot);
                    break;
                case MklBuffer::kBytes:
                    enc.set_bytes(b.bytes.data(), static_cast<int>(b.bytes.size()), b.slot);
                    break;
            }
        }
    }

    MTL::Size grid(grid_[0], grid_[1], grid_[2]);
    MTL::Size tg(tg_[0], tg_[1], tg_[2]);
    if (by_threadgroups_) {
        enc.dispatch_threadgroups(grid, tg);
    } else {
        enc.dispatch_threads(grid, tg);
    }
    enc.add_temporaries(std::move(copies));
}

}  // namespace

std::vector<array> metal_kernel_lib(const std::vector<array>& inputs,
                                    const std::vector<Shape>& out_shapes,
                                    const std::vector<Dtype>& out_dtypes,
                                    const std::string& libpath, const std::string& kname,
                                    const std::string& hash_name, std::array<int, 3> grid,
                                    std::array<int, 3> threadgroup, bool by_threadgroups,
                                    std::vector<MklBuffer> buffers,
                                    std::vector<MklConstant> constants, StreamOrDevice s_) {
    auto s = to_stream(s_);
    return array::make_arrays(out_shapes, out_dtypes,
                              std::make_shared<MetalKernelLibKernel>(
                                  s, libpath, kname, hash_name, grid, threadgroup, by_threadgroups,
                                  std::move(buffers), std::move(constants)),
                              inputs);
}

}  // namespace mlx::core
