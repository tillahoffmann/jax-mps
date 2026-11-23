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

// LAPACK's Householder QR routines, overloaded on the element type so the
// batch loops below can be written once for float32 and complex64.
void LapackGeqrf(const __LAPACK_int* m, const __LAPACK_int* n, float* a, const __LAPACK_int* lda,
                 float* tau, float* work, const __LAPACK_int* lwork, __LAPACK_int* info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
void LapackGeqrf(const __LAPACK_int* m, const __LAPACK_int* n, cfloat* a, const __LAPACK_int* lda,
                 cfloat* tau, cfloat* work, const __LAPACK_int* lwork, __LAPACK_int* info) {
    cgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
// Q from k elementary reflectors: orgqr (real) / ungqr (complex).
void LapackOrgqr(const __LAPACK_int* m, const __LAPACK_int* n, const __LAPACK_int* k, float* a,
                 const __LAPACK_int* lda, const float* tau, float* work, const __LAPACK_int* lwork,
                 __LAPACK_int* info) {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
}
void LapackOrgqr(const __LAPACK_int* m, const __LAPACK_int* n, const __LAPACK_int* k, cfloat* a,
                 const __LAPACK_int* lda, const cfloat* tau, cfloat* work,
                 const __LAPACK_int* lwork, __LAPACK_int* info) {
    cungqr_(m, n, k, a, lda, tau, work, lwork, info);
}
__LAPACK_int WorkSize(float query) {
    return static_cast<__LAPACK_int>(query);
}
__LAPACK_int WorkSize(cfloat query) {
    return static_cast<__LAPACK_int>(query.real());
}

// Copies a row-major rows x cols matrix into column-major storage with leading
// dimension `rows`, and back.
template <typename T>
void ToColMajor(const T* src, T* col, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            col[static_cast<size_t>(j) * rows + i] = src[static_cast<size_t>(i) * cols + j];
}
template <typename T>
void FromColMajor(const T* col, T* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[static_cast<size_t>(i) * cols + j] = col[static_cast<size_t>(j) * rows + i];
}

// Calls f(T{}) with T the C++ element type of `dtype` (float32 or complex64).
template <typename F>
void DispatchQrType(mlx::core::Dtype dtype, F&& f) {
    if (dtype == mlx::core::float32)
        f(float{});
    else if (dtype == mlx::core::complex64)
        f(cfloat{});
    else
        throw std::runtime_error("Householder QR: expected float32 or complex64");
}

// One LAPACK/BLAS routine applied to every matrix in a batch, run on
// MLX's CPU stream. Inputs are row-contiguous (the wrappers below ensure it);
// outputs are allocated here and filled by the dispatched task.
class ComplexLinalgPrimitive : public mlx::core::Primitive {
public:
    // Cholesky and triangular solve are complex64 only; the Householder kinds
    // (kQr, kGeqrf, kHouseholderProduct) take float32 or complex64.
    enum class Kind { kCholesky, kTriangularSolve, kQr, kGeqrf, kHouseholderProduct };

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
                encoder.dispatch([a = inputs[0], q = outputs[0], r = outputs[1]]() mutable {
                    DispatchQrType(a.dtype(), [&](auto t) { RunQr<decltype(t)>(a, q, r); });
                });
                break;
            case Kind::kGeqrf:
                encoder.dispatch([a = inputs[0], out = outputs[0], taus = outputs[1]]() mutable {
                    DispatchQrType(a.dtype(), [&](auto t) { RunGeqrf<decltype(t)>(a, out, taus); });
                });
                break;
            case Kind::kHouseholderProduct:
                encoder.dispatch([a = inputs[0], taus = inputs[1], out = outputs[0]]() mutable {
                    DispatchQrType(a.dtype(), [&](auto t) {
                        RunHouseholderProduct<decltype(t)>(a, taus, out);
                    });
                });
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

    // Workspace size covering geqrf on an m x n matrix and orgqr producing
    // qCols columns from k reflectors.
    template <typename T>
    static __LAPACK_int QrWorkSize(int m, int n, int k, int qCols, T* col, T* tau) {
        __LAPACK_int lm = m;
        __LAPACK_int ln = n;
        __LAPACK_int lk = k;
        __LAPACK_int lq = qCols;
        __LAPACK_int ld = std::max(m, 1);
        __LAPACK_int info = 0;
        __LAPACK_int lwork = -1;
        T query{};
        __LAPACK_int need = 1;
        if (n > 0) {
            LapackGeqrf(&lm, &ln, col, &ld, tau, &query, &lwork, &info);
            need = std::max(need, WorkSize(query));
        }
        if (qCols > 0) {
            lwork = -1;
            LapackOrgqr(&lm, &lq, &lk, col, &ld, tau, &query, &lwork, &info);
            need = std::max(need, WorkSize(query));
        }
        return need;
    }

    template <typename T>
    static void RunQr(const mlx::core::array& a, mlx::core::array& q, mlx::core::array& r) {
        const int M = Dim(a, -2);
        const int N = Dim(a, -1);
        const int K = std::min(M, N);
        const int qCols = Dim(q, -1);  // K (reduced) or M (full)
        const int rRows = Dim(r, -2);
        const size_t mn = static_cast<size_t>(M) * N;
        const size_t qStride = static_cast<size_t>(M) * qCols;
        const size_t batch = qStride == 0 ? 0 : q.size() / qStride;
        const T* src = a.data<T>();
        T* qOut = q.data<T>();
        T* rOut = r.data<T>();
        if (batch == 0) {
            std::fill(qOut, qOut + q.size(), T(0));
            std::fill(rOut, rOut + r.size(), T(0));
            return;
        }

        // Column-major work matrix with room for every column of Q.
        const int workCols = std::max(N, qCols);
        std::vector<T> col(static_cast<size_t>(M) * workCols);
        std::vector<T> tau(std::max(K, 1));
        __LAPACK_int lm = M;
        __LAPACK_int ln = N;
        __LAPACK_int lk = K;
        __LAPACK_int lq = qCols;
        __LAPACK_int info = 0;
        __LAPACK_int lwork = QrWorkSize<T>(M, N, K, qCols, col.data(), tau.data());
        std::vector<T> work(lwork);

        for (size_t m = 0; m < batch; ++m, src += mn) {
            std::fill(col.begin(), col.end(), T(0));
            ToColMajor(src, col.data(), M, N);
            if (N > 0)
                LapackGeqrf(&lm, &ln, col.data(), &lm, tau.data(), work.data(), &lwork, &info);

            // R: upper triangle of the first rRows rows; zero below.
            T* rm = rOut + m * static_cast<size_t>(rRows) * N;
            for (int i = 0; i < rRows; ++i)
                for (int j = 0; j < N; ++j)
                    rm[static_cast<size_t>(i) * N + j] =
                        j >= i ? col[static_cast<size_t>(j) * M + i] : T(0);

            // Q: accumulate the K reflectors into qCols orthonormal columns.
            // With qCols > N (full matrices, tall input) the extra columns
            // complete the orthonormal basis.
            LapackOrgqr(&lm, &lq, &lk, col.data(), &lm, tau.data(), work.data(), &lwork, &info);
            FromColMajor(col.data(), qOut + m * qStride, M, qCols);
        }
    }

    // LAPACK geqrf: packed reflectors (below the diagonal) and R (on and above
    // it), plus the K scalar factors tau.
    template <typename T>
    static void RunGeqrf(const mlx::core::array& a, mlx::core::array& out, mlx::core::array& taus) {
        const int M = Dim(a, -2);
        const int N = Dim(a, -1);
        const int K = std::min(M, N);
        const size_t mn = static_cast<size_t>(M) * N;
        const T* src = a.data<T>();
        T* aOut = out.data<T>();
        T* tOut = taus.data<T>();
        std::fill(tOut, tOut + taus.size(), T(0));
        if (mn == 0) {
            std::copy(src, src + a.size(), aOut);
            return;
        }
        const size_t batch = a.size() / mn;
        std::vector<T> col(mn);
        std::vector<T> tau(std::max(K, 1));
        __LAPACK_int lm = M;
        __LAPACK_int ln = N;
        __LAPACK_int info = 0;
        __LAPACK_int lwork = QrWorkSize<T>(M, N, K, 0, col.data(), tau.data());
        std::vector<T> work(lwork);
        for (size_t m = 0; m < batch; ++m) {
            ToColMajor(src + m * mn, col.data(), M, N);
            LapackGeqrf(&lm, &ln, col.data(), &lm, tau.data(), work.data(), &lwork, &info);
            FromColMajor(col.data(), aOut + m * mn, M, N);
            std::copy(tau.begin(), tau.begin() + K, tOut + m * static_cast<size_t>(K));
        }
    }

    // LAPACK orgqr/ungqr: the first N columns of H_1 ... H_k from k reflectors
    // stored below the diagonal of `a` (M >= N >= k).
    template <typename T>
    static void RunHouseholderProduct(const mlx::core::array& a, const mlx::core::array& taus,
                                      mlx::core::array& out) {
        const int M = Dim(a, -2);
        const int N = Dim(a, -1);
        const int K = Dim(taus, -1);
        const size_t mn = static_cast<size_t>(M) * N;
        const T* src = a.data<T>();
        const T* tSrc = taus.data<T>();
        T* dst = out.data<T>();
        if (mn == 0)
            return;
        const size_t batch = a.size() / mn;
        std::vector<T> col(mn);
        std::vector<T> tau(std::max(K, 1));
        __LAPACK_int lm = M;
        __LAPACK_int ln = N;
        __LAPACK_int lk = K;
        __LAPACK_int info = 0;
        __LAPACK_int lwork = QrWorkSize<T>(M, 0, K, N, col.data(), tau.data());
        std::vector<T> work(lwork);
        for (size_t m = 0; m < batch; ++m) {
            ToColMajor(src + m * mn, col.data(), M, N);
            std::copy(tSrc + m * static_cast<size_t>(K), tSrc + (m + 1) * static_cast<size_t>(K),
                      tau.begin());
            LapackOrgqr(&lm, &ln, &lk, col.data(), &lm, tau.data(), work.data(), &lwork, &info);
            FromColMajor(col.data(), dst + m * mn, M, N);
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

std::pair<mlx::core::array, mlx::core::array> HouseholderQr(const mlx::core::array& a,
                                                            bool fullMatrices) {
    const int M = a.shape(-2);
    const int N = a.shape(-1);
    const int K = std::min(M, N);
    const int qCols = fullMatrices ? M : K;
    auto prim = std::make_shared<ComplexLinalgPrimitive>(ComplexLinalgPrimitive::Params{
        .kind = ComplexLinalgPrimitive::Kind::kQr, .fullMatrices = fullMatrices});
    auto outs = mlx::core::array::make_arrays(
        {WithLastTwo(a.shape(), M, qCols), WithLastTwo(a.shape(), qCols, N)},
        {a.dtype(), a.dtype()}, std::move(prim), {mlx::core::contiguous(a)});
    return {outs[0], outs[1]};
}

std::pair<mlx::core::array, mlx::core::array> Geqrf(const mlx::core::array& a) {
    const int K = std::min(a.shape(-2), a.shape(-1));
    mlx::core::Shape tauShape(a.shape().begin(), a.shape().end() - 1);
    tauShape.back() = K;
    auto prim = std::make_shared<ComplexLinalgPrimitive>(
        ComplexLinalgPrimitive::Params{.kind = ComplexLinalgPrimitive::Kind::kGeqrf});
    auto outs = mlx::core::array::make_arrays({a.shape(), tauShape}, {a.dtype(), a.dtype()},
                                              std::move(prim), {mlx::core::contiguous(a)});
    return {outs[0], outs[1]};
}

mlx::core::array HouseholderProduct(const mlx::core::array& a, const mlx::core::array& taus) {
    auto prim = std::make_shared<ComplexLinalgPrimitive>(
        ComplexLinalgPrimitive::Params{.kind = ComplexLinalgPrimitive::Kind::kHouseholderProduct});
    return mlx::core::array(a.shape(), a.dtype(), std::move(prim),
                            {mlx::core::contiguous(a), mlx::core::contiguous(taus)});
}

}  // namespace jax_mps
