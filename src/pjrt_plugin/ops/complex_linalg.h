#pragma once

// complex64 decompositions and solves that MLX's linalg rejects (issue #120).
//
// MLX's linalg::cholesky / qr / solve_triangular only accept float32/float64,
// so complex64 inputs are routed here instead: a CPU-stream MLX primitive that
// calls Accelerate's LAPACK/BLAS directly (cpotrf, cblas_ctrsm,
// cgeqrf + cungqr) — the same routines XLA's CPU backend uses, so results match
// it. All functions take and return row-major arrays with arbitrary leading
// batch dimensions.

#include <mlx/array.h>

#include <utility>

namespace jax_mps {

// Cholesky factor of Hermitian positive-definite matrices. Only the `lower`
// (or upper) triangle of `a` is read; the other triangle of the result is
// zero. Matrices that are not positive definite yield all-NaN factors, as on
// XLA's CPU backend.
mlx::core::array ComplexCholesky(const mlx::core::array& a, bool lower);

// How a triangular matrix enters a solve: as-is, transposed, or conjugate
// transposed (stablehlo.triangular_solve's transpose_a).
enum class ComplexTranspose { kNone, kTranspose, kAdjoint };

// Solves op(a) x = b (leftSide) or x op(a) = b for triangular `a`.
mlx::core::array ComplexTriangularSolve(const mlx::core::array& a, const mlx::core::array& b,
                                        bool leftSide, bool lower, ComplexTranspose transpose,
                                        bool unitDiagonal);

// Householder QR: returns (Q, R) with Q of shape (..., M, K) and R of shape
// (..., K, N), K = min(M, N), or Q (..., M, M) and R (..., M, N) when
// fullMatrices.
std::pair<mlx::core::array, mlx::core::array> ComplexQr(const mlx::core::array& a,
                                                        bool fullMatrices);

}  // namespace jax_mps
