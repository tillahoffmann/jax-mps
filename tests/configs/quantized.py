"""Test configs for weight-only quantization ops (see issue #189).

These exercise `mps.quantize` / `mps.dequantize` / `mps.quantized_matmul` as
compositions so the packed uint32 stays internal to each backend: only the final
float outputs are compared cross-backend (CPU fallback vs fused MPS). The
fallback replicates MLX's affine math exactly, so random (non-boundary) inputs
round-trip identically on both platforms.

Forward-only for now (Phase 1); differentiability is a non-blocking follow-up.
"""

from jax import random

from jax_plugins.mps.ops import dequantize, quantize, quantized_matmul

from .util import OperationTestConfig

# quantized_matmul has no VJP yet (Phase 2, issue #205); differentiating the
# float activation `x` raises this until then. Kept as differentiable so the test
# suite surfaces the gap (xfail) and auto-xpasses once the rule lands.
_QMATMUL_GRAD_XFAIL = "Differentiation rule for 'mps.quantized_matmul' not implemented"


def make_quantized_op_configs():
    with OperationTestConfig.module_name("quantized"):
        for bits in (8, 4):
            gs = 64

            # quantize -> dequantize round-trip (covers quantize + dequantize).
            # Forward-only: quantize rounding has no meaningful gradient.
            yield OperationTestConfig(
                lambda w, bits=bits, gs=gs: dequantize(
                    *quantize(w, group_size=gs, bits=bits), group_size=gs, bits=bits
                ),
                lambda key: random.normal(key, (4, 64)),
                name=f"quant_roundtrip_b{bits}",
                differentiable_argnums=(),
            )

            # quantized_matmul vs dequant+matmul (covers quantize + quantized_matmul).
            # w: [out, in] quantized along in; result x @ wᵀ.
            yield OperationTestConfig(
                lambda x, w, bits=bits, gs=gs: quantized_matmul(
                    x,
                    *quantize(w, group_size=gs, bits=bits),
                    transpose=True,
                    group_size=gs,
                    bits=bits,
                ),
                lambda key: random.normal(key, (3, 64)),  # x: [M, in]
                lambda key: random.normal(key, (8, 64)),  # w: [out, in]
                name=f"quantized_matmul_b{bits}",
                differentiable_argnums=(0,),
                grad_xfail=_QMATMUL_GRAD_XFAIL,
            )

            # quantized_matmul with transpose=False: w is [in, out] quantized
            # along out; result x @ w.
            yield OperationTestConfig(
                lambda x, w, bits=bits, gs=gs: quantized_matmul(
                    x,
                    *quantize(w, group_size=gs, bits=bits),
                    transpose=False,
                    group_size=gs,
                    bits=bits,
                ),
                lambda key: random.normal(key, (3, 8)),  # x: [M, in]
                lambda key: random.normal(key, (8, 64)),  # w: [in, out]
                name=f"quantized_matmul_notranspose_b{bits}",
                differentiable_argnums=(0,),
                grad_xfail=_QMATMUL_GRAD_XFAIL,
            )

        # Non-default group_size (32) to exercise the packing/grouping layout
        # beyond the gs=64 default, across both bit widths.
        yield OperationTestConfig(
            lambda w: dequantize(
                *quantize(w, group_size=32, bits=8), group_size=32, bits=8
            ),
            lambda key: random.normal(key, (4, 64)),
            name="quant_roundtrip_b8_gs32",
            differentiable_argnums=(),
        )
        yield OperationTestConfig(
            lambda x, w: quantized_matmul(
                x,
                *quantize(w, group_size=32, bits=4),
                transpose=True,
                group_size=32,
                bits=4,
            ),
            lambda key: random.normal(key, (3, 64)),  # x: [M, in]
            lambda key: random.normal(key, (8, 64)),  # w: [out, in]
            name="quantized_matmul_b4_gs32",
            differentiable_argnums=(0,),
            grad_xfail=_QMATMUL_GRAD_XFAIL,
        )
