"""Map Parakeet safetensors (NeMo/mlx-audio layout) into the NNX modules."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def conv1d_w(w: np.ndarray) -> np.ndarray:
    # MLX Conv1d [out, k, in/groups] -> lax WIO [k, in/groups, out].
    return np.transpose(w, (1, 2, 0))


def conv2d_w(w: np.ndarray) -> np.ndarray:
    # MLX Conv2d [out, kh, kw, in/groups] -> lax HWIO [kh, kw, in/groups, out].
    return np.transpose(w, (1, 2, 3, 0))


def load_encoder(encoder, t: dict, dtype=jnp.float32):
    """Load Conformer encoder params from the tensor dict ``t``."""

    def w(name):
        return jnp.asarray(t[name]).astype(dtype)

    pre = encoder.pre_encode
    for key, conv in pre.conv.items():
        p = f"encoder.pre_encode.conv.{key}"
        conv.kernel.value = jnp.asarray(conv2d_w(t[f"{p}.weight"])).astype(dtype)
        conv.bias.value = w(f"{p}.bias")
    pre.out.kernel.value = w("encoder.pre_encode.out.weight").T
    pre.out.bias.value = w("encoder.pre_encode.out.bias")

    for i, layer in enumerate(encoder.layers):
        b = f"encoder.layers.{i}"

        def ln(mod, name):
            mod.scale.value = w(f"{b}.{name}.weight")
            mod.bias.value = w(f"{b}.{name}.bias")

        def ff(mod, name):
            mod.linear1.kernel.value = w(f"{b}.{name}.linear1.weight").T
            mod.linear2.kernel.value = w(f"{b}.{name}.linear2.weight").T

        ln(layer.norm_feed_forward1, "norm_feed_forward1")
        ff(layer.feed_forward1, "feed_forward1")

        ln(layer.norm_self_att, "norm_self_att")
        a = layer.self_attn
        for nm in ("linear_q", "linear_k", "linear_v", "linear_pos", "linear_out"):
            getattr(a, nm).kernel.value = w(f"{b}.self_attn.{nm}.weight").T
        a.pos_bias_u.value = w(f"{b}.self_attn.pos_bias_u")
        a.pos_bias_v.value = w(f"{b}.self_attn.pos_bias_v")

        ln(layer.norm_conv, "norm_conv")
        c = layer.conv
        c.pointwise_conv1.kernel.value = jnp.asarray(
            conv1d_w(t[f"{b}.conv.pointwise_conv1.weight"])
        ).astype(dtype)
        c.depthwise_conv.kernel.value = jnp.asarray(
            conv1d_w(t[f"{b}.conv.depthwise_conv.weight"])
        ).astype(dtype)
        c.pointwise_conv2.kernel.value = jnp.asarray(
            conv1d_w(t[f"{b}.conv.pointwise_conv2.weight"])
        ).astype(dtype)
        bn = c.batch_norm
        bn.weight.value = w(f"{b}.conv.batch_norm.weight")
        bn.bias.value = w(f"{b}.conv.batch_norm.bias")
        bn.running_mean.value = w(f"{b}.conv.batch_norm.running_mean")
        bn.running_var.value = w(f"{b}.conv.batch_norm.running_var")

        ln(layer.norm_feed_forward2, "norm_feed_forward2")
        ff(layer.feed_forward2, "feed_forward2")
        ln(layer.norm_out, "norm_out")
