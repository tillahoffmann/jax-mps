// complex64 Cholesky, triangular solve and QR via Accelerate (issue #120).
// See complex_linalg.h.

#include "pjrt_plugin/ops/complex_linalg.h"

// Accelerate's current LAPACK/BLAS interface (the one MLX itself builds
// against); in C++ it types complex values as std::complex.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <mlx/allocator.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jax_mps {

namespace {

using cfloat = std::complex<float>;

// One complex64 LAPACK/BLAS routine applied to every matrix in a batch, run on
// MLX's CPU stream. Inputs are row-contiguous (the wrappers below ensure it);
// outputs are allocated here and filled by the dispatched task.
class ComplexLinalgPrimitive : public mlx::core::Primitive {
public:
    enum class Kind { kCholesky, kTriangularSolve, kQr };

    struct Params {
        Kind kind;
        bool lower = true;                                     // cholesky, triangular solve
        bool leftSide = true;                                  // triangular solve
        ComplexTranspose transpose = ComplexTranspose::kNone;  // triangular solve
        bool unitDiagonal = false;                             // triangular solve
        bool fullMatrices = false;                             // qr
    };

    explicit ComplexLinalgPrimitive(Params params)
        : Primitive(mlx::core::default_stream(mlx::core::Device::cpu)), params_(params) {}

    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override {
        for (const auto& in : inputs) {
            if (!in.flags().row_contiguous)
                throw std::runtime_error("ComplexLinalgPrimitive: inputs must be row-contiguous");
        }
        for (auto& out : outputs)
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        auto& encoder = mlx::core::cpu::get_command_encoder(stream());
        for (const auto& in : inputs)
            encoder.set_input_array(in);
        for (auto& out : outputs)
            encoder.set_output_array(out);

        // Capture the arrays by value so their buffers outlive the task.
        switch (params_.kind) {
            case Kind::kCholesky:
                encoder.dispatch([a = inputs[0], out = outputs[0],
                                  lower = params_.lower]() mutable { RunCholesky(a, out, lower); });
                break;
            case Kind::kTriangularSolve:
                encoder.dispatch([a = inputs[0], b = inputs[1], out = outputs[0],
                                  p = params_]() mutable { RunTriangularSolve(a, b, out, p); });
                break;
            case Kind::kQr:
                encoder.dispatch(
                    [a = inputs[0], q = outputs[0], r = outputs[1]]() mutable { RunQr(a, q, r); });
                break;
        }
    }

    void eval_gpu(const std::vector<mlx::core::array>& /*inputs*/,
                  std::vector<mlx::core::array>& /*outputs*/) override {
        throw std::runtime_error("ComplexLinalgPrimitive runs on the CPU stream only");
    }

    const char* name() const override {
        return "ComplexLinalg";
    }

private:
    static int Dim(const mlx::core::array& x, int axis) {
        return x.shape(axis < 0 ? static_cast<int>(x.ndim()) + axis : axis);
    }

    static void RunCholesky(const mlx::core::array& a, mlx::core::array& out, bool lower) {
        const int n = Dim(a, -1);
        const size_t nn = static_cast<size_t>(n) * n;
        const size_t batch = nn == 0 ? 0 : a.size() / nn;
        const cfloat* src = a.data<cfloat>();
        cfloat* dst = out.data<cfloat>();
        std::vector<cfloat> col(nn);
        const char uplo = lower ? 'L' : 'U';
        const cfloat nan(std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN());
        for (size_t m = 0; m < batch; ++m, src += nn, dst += nn) {
            // Row-major -> column-major so LAPACK's uplo means what it says.
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    col[static_cast<size_t>(j) * n + i] = src[static_cast<size_t>(i) * n + j];
            __LAPACK_int ln = n;
            __LAPACK_int info = 0;
            cpotrf_(&uplo, &ln, col.data(), &ln, &info);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const bool keep = lower ? j <= i : j >= i;
                    cfloat v = keep ? col[static_cast<size_t>(j) * n + i] : cfloat(0.0F);
                    dst[static_cast<size_t>(i) * n + j] = info == 0 ? v : nan;
                }
            }
        }
    }

    static void RunTriangularSolve(const mlx::core::array& a, const mlx::core::array& b,
                                   mlx::core::array& out, const Params& p) {
        const int rows = Dim(b, -2);
        const int cols = Dim(b, -1);
        const int k = Dim(a, -1);
        const size_t bStride = static_cast<size_t>(rows) * cols;
        const size_t aStride = static_cast<size_t>(k) * k;
        const size_t batch = bStride == 0 ? 0 : b.size() / bStride;
        const cfloat* aData = a.data<cfloat>();
        const cfloat* bData = b.data<cfloat>();
        cfloat* x = out.data<cfloat>();
        std::copy(bData, bData + b.size(), x);
        if (bStride == 0)
            return;

        CBLAS_TRANSPOSE trans = CblasNoTrans;
        if (p.transpose == ComplexTranspose::kTranspose)
            trans = CblasTrans;
        else if (p.transpose == ComplexTranspose::kAdjoint)
            trans = CblasConjTrans;
        const cfloat alpha(1.0F, 0.0F);
        for (size_t m = 0; m < batch; ++m) {
            cblas_ctrsm(CblasRowMajor, p.leftSide ? CblasLeft : CblasRight,
                        p.lower ? CblasLower : CblasUpper, trans,
                        p.unitDiagonal ? CblasUnit : CblasNonUnit, rows, cols, &alpha,
                        aData + m * aStride, k, x + m * bStride, cols);
        }
    }

    static void RunQr(const mlx::core::array& a, mlx::core::array& q, mlx::core::array& r) {
        const int M = Dim(a, -2);
        const int N = Dim(a, -1);
        const int K = std::min(M, N);
        const int qCols = Dim(q, -1);  // K (reduced) or M (full)
        const int rRows = Dim(r, -2);
        const size_t mn = static_cast<size_t>(M) * N;
        const size_t batch = mn == 0 ? 0 : a.size() / mn;
        const cfloat* src = a.data<cfloat>();
        cfloat* qOut = q.data<cfloat>();
        cfloat* rOut = r.data<cfloat>();
        if (batch == 0) {
            std::fill(qOut, qOut + q.size(), cfloat(0.0F));
            std::fill(rOut, rOut + r.size(), cfloat(0.0F));
            return;
        }

        // Column-major work matrix with room for every column of Q.
        const int workCols = std::max(N, qCols);
        std::vector<cfloat> col(static_cast<size_t>(M) * workCols);
        std::vector<cfloat> tau(std::max(K, 1));
        __LAPACK_int lm = M;
        __LAPACK_int ln = N;
        __LAPACK_int lk = K;
        __LAPACK_int lq = qCols;
        __LAPACK_int info = 0;

        // Workspace query for both routines.
        cfloat query;
        __LAPACK_int lwork = -1;
        cgeqrf_(&lm, &ln, col.data(), &lm, tau.data(), &query, &lwork, &info);
        __LAPACK_int need = static_cast<__LAPACK_int>(query.real());
        lwork = -1;
        cungqr_(&lm, &lq, &lk, col.data(), &lm, tau.data(), &query, &lwork, &info);
        need = std::max(need, static_cast<__LAPACK_int>(query.real()));
        lwork = std::max<__LAPACK_int>(need, 1);
        std::vector<cfloat> work(lwork);

        for (size_t m = 0; m < batch; ++m, src += mn) {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    col[static_cast<size_t>(j) * M + i] = src[static_cast<size_t>(i) * N + j];
            cgeqrf_(&lm, &ln, col.data(), &lm, tau.data(), work.data(), &lwork, &info);

            // R: upper triangle of the first rRows rows; zero below.
            cfloat* rm = rOut + m * static_cast<size_t>(rRows) * N;
            for (int i = 0; i < rRows; ++i)
                for (int j = 0; j < N; ++j)
                    rm[static_cast<size_t>(i) * N + j] =
                        j >= i ? col[static_cast<size_t>(j) * M + i] : cfloat(0.0F);

            // Q: accumulate the K reflectors into qCols orthonormal columns.
            cungqr_(&lm, &lq, &lk, col.data(), &lm, tau.data(), work.data(), &lwork, &info);
            cfloat* qm = qOut + m * static_cast<size_t>(M) * qCols;
            for (int i = 0; i < M; ++i)
                for (int c = 0; c < qCols; ++c)
                    qm[static_cast<size_t>(i) * qCols + c] = col[static_cast<size_t>(c) * M + i];
        }
    }

    Params params_;
};

mlx::core::Shape WithLastTwo(const mlx::core::Shape& shape, int rows, int cols) {
    mlx::core::Shape out = shape;
    out[out.size() - 2] = rows;
    out[out.size() - 1] = cols;
    return out;
}

}  // namespace

mlx::core::array ComplexCholesky(const mlx::core::array& a, bool lower) {
    auto prim = std::make_shared<ComplexLinalgPrimitive>(ComplexLinalgPrimitive::Params{
        .kind = ComplexLinalgPrimitive::Kind::kCholesky, .lower = lower});
    return mlx::core::array(a.shape(), mlx::core::complex64, std::move(prim),
                            {mlx::core::contiguous(a)});
}

mlx::core::array ComplexTriangularSolve(const mlx::core::array& a, const mlx::core::array& b,
                                        bool leftSide, bool lower, ComplexTranspose transpose,
                                        bool unitDiagonal) {
    auto prim = std::make_shared<ComplexLinalgPrimitive>(
        ComplexLinalgPrimitive::Params{.kind = ComplexLinalgPrimitive::Kind::kTriangularSolve,
                                       .lower = lower,
                                       .leftSide = leftSide,
                                       .transpose = transpose,
                                       .unitDiagonal = unitDiagonal});
    return mlx::core::array(b.shape(), mlx::core::complex64, std::move(prim),
                            {mlx::core::contiguous(a), mlx::core::contiguous(b)});
}

std::pair<mlx::core::array, mlx::core::array> ComplexQr(const mlx::core::array& a,
                                                        bool fullMatrices) {
    const int M = a.shape(-2);
    const int N = a.shape(-1);
    const int K = std::min(M, N);
    const int qCols = fullMatrices ? M : K;
    auto prim = std::make_shared<ComplexLinalgPrimitive>(ComplexLinalgPrimitive::Params{
        .kind = ComplexLinalgPrimitive::Kind::kQr, .fullMatrices = fullMatrices});
    auto outs = mlx::core::array::make_arrays(
        {WithLastTwo(a.shape(), M, qCols), WithLastTwo(a.shape(), qCols, N)},
        {mlx::core::complex64, mlx::core::complex64}, std::move(prim), {mlx::core::contiguous(a)});
    return {outs[0], outs[1]};
}

}  // namespace jax_mps
