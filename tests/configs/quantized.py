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


def make_quantized_op_configs():
    with OperationTestConfig.module_name("quantized"):
        for bits in (8, 4):
            gs = 64

            # quantize -> dequantize round-trip (covers quantize + dequantize).
            yield OperationTestConfig(
                lambda w, bits=bits, gs=gs: dequantize(
                    *quantize(w, group_size=gs, bits=bits), group_size=gs, bits=bits
                ),
                lambda key: random.normal(key, (4, 64)),
                name=f"quant_roundtrip_b{bits}",
                differentiable_argnums=(),
            )

            # quantized_matmul vs dequant+matmul (covers quantize + quantized_matmul).
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
                differentiable_argnums=(),
            )
