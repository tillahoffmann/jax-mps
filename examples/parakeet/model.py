"""Parakeet Conformer encoder in Flax NNX.

Faithful reimplementation of NVIDIA's FastConformer encoder as used by
mlx-audio's Parakeet-TDT (global ``rel_pos`` attention). This is the
compute-heavy part of the model and is JIT-compiled onto the MPS backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from jax_plugins.mps.ops import sdpa


@dataclass(frozen=True)
class ConformerConfig:
    feat_in: int = 128
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    subsampling_factor: int = 8
    conv_kernel_size: int = 9
    subsampling_conv_channels: int = 256
    pos_emb_max_len: int = 5000
    dtype: jnp.dtype = jnp.float32


class FeedForward(nnx.Module):
    def __init__(self, d_model, d_ff, *, dtype=jnp.float32):
        self.linear1 = nnx.Linear(
            d_model,
            d_ff,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        self.linear2 = nnx.Linear(
            d_ff,
            d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x):
        return self.linear2(nnx.silu(self.linear1(x)))


class RelPositionMultiHeadAttention(nnx.Module):
    def __init__(self, n_head, d_model, *, dtype=jnp.float32):
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim**-0.5

        def lin(out):
            return nnx.Linear(
                d_model,
                out,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                rngs=nnx.Rngs(0),
            )

        # Fused Q/K/V: one (d, 3d) matmul is better-tiled than three (d, d) ones
        # at the short sequence lengths here.
        self.linear_qkv = lin(3 * d_model)
        self.linear_out = lin(d_model)
        self.linear_pos = lin(d_model)
        self.pos_bias_u = nnx.Param(jnp.zeros((n_head, self.head_dim), dtype))
        self.pos_bias_v = nnx.Param(jnp.zeros((n_head, self.head_dim), dtype))

    @staticmethod
    def _rel_shift(x):
        # x: (B, H, T, 2T-1) -> shift so column j holds relative position j-(T-1).
        b, h, t, pos_len = x.shape
        x = jnp.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(b, h, pos_len + 1, t)
        x = x[:, :, 1:, :]
        return x.reshape(b, h, t, pos_len)

    def __call__(self, x, pos_emb):
        b, t, _ = x.shape
        h, d = self.n_head, self.head_dim
        q, k, v = jnp.split(self.linear_qkv(x), 3, axis=-1)
        q = q.reshape(b, t, h, d)
        k = k.reshape(b, t, h, d).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, h, d).transpose(0, 2, 1, 3)
        p = (
            self.linear_pos(pos_emb)
            .reshape(pos_emb.shape[0], -1, h, d)
            .transpose(0, 2, 1, 3)
        )

        q_u = (q + self.pos_bias_u.value).transpose(0, 2, 1, 3)  # (B,H,T,d)
        q_v = (q + self.pos_bias_v.value).transpose(0, 2, 1, 3)

        # Positional bias (matrix_bd), scaled, becomes the additive SDPA mask so
        # the whole content+position attention fuses into one mps.sdpa kernel.
        matrix_bd = jnp.matmul(q_v, p.transpose(0, 1, 3, 2))  # (B,H,T,2T-1)
        bias = (
            self._rel_shift(matrix_bd)[:, :, :, : k.shape[2]] * self.scale
        )  # (B,H,T,T)

        out = sdpa(q_u, k, v, scale=self.scale, mask=bias)  # (B,H,T,d)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, h * d)
        return self.linear_out(out)


class Convolution(nnx.Module):
    def __init__(self, cfg: ConformerConfig):
        d, dt = cfg.d_model, cfg.dtype
        pad = (cfg.conv_kernel_size - 1) // 2
        # The pointwise (kernel-size-1) convs are just channel matmuls on the
        # channels-last activations, so use Linear (they lower to dot_general).
        self.pointwise_conv1 = nnx.Linear(
            d, 2 * d, use_bias=False, dtype=dt, param_dtype=dt, rngs=nnx.Rngs(0)
        )
        self.depthwise_conv = nnx.Conv(
            d,
            d,
            kernel_size=(cfg.conv_kernel_size,),
            padding=pad,
            feature_group_count=d,
            use_bias=False,
            dtype=dt,
            param_dtype=dt,
            rngs=nnx.Rngs(0),
        )
        self.batch_norm = nnx.BatchNorm(
            d, use_running_average=True, dtype=dt, param_dtype=dt, rngs=nnx.Rngs(0)
        )
        self.pointwise_conv2 = nnx.Linear(
            d, d, use_bias=False, dtype=dt, param_dtype=dt, rngs=nnx.Rngs(0)
        )

    def __call__(self, x):  # x: (B, T, d)
        x = nnx.glu(self.pointwise_conv1(x), axis=-1)
        x = self.depthwise_conv(x)
        x = nnx.silu(self.batch_norm(x))
        return self.pointwise_conv2(x)


class ConformerBlock(nnx.Module):
    def __init__(self, cfg: ConformerConfig):
        d, dt = cfg.d_model, cfg.dtype
        d_ff = d * cfg.ff_expansion_factor

        def ln():
            return nnx.LayerNorm(d, dtype=dt, param_dtype=dt, rngs=nnx.Rngs(0))

        self.norm_feed_forward1 = ln()
        self.feed_forward1 = FeedForward(d, d_ff, dtype=dt)
        self.norm_self_att = ln()
        self.self_attn = RelPositionMultiHeadAttention(cfg.n_heads, d, dtype=dt)
        self.norm_conv = ln()
        self.conv = Convolution(cfg)
        self.norm_feed_forward2 = ln()
        self.feed_forward2 = FeedForward(d, d_ff, dtype=dt)
        self.norm_out = ln()

    def __call__(self, x, pos_emb):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class DwStridingSubsampling(nnx.Module):
    """3-stage depthwise-striding subsampling (factor 8) over the mel features."""

    def __init__(self, cfg: ConformerConfig):
        c, dt = cfg.subsampling_conv_channels, cfg.dtype
        self._sampling_num = int(math.log2(cfg.subsampling_factor))
        freq = cfg.feat_in
        for _ in range(self._sampling_num):
            freq = (freq + 2 * 1 - 3) // 2 + 1

        def conv2d(ic, oc, k, stride, groups=1):
            return nnx.Conv(
                ic,
                oc,
                kernel_size=(k, k),
                strides=(stride, stride),
                padding=(k - 1) // 2,
                feature_group_count=groups,
                dtype=dt,
                param_dtype=dt,
                rngs=nnx.Rngs(0),
            )

        # conv dict index matches checkpoint keys (ReLU layers occupy 1/4/7).
        conv = {"0": conv2d(1, c, 3, stride=2)}
        for stage in range(1, self._sampling_num):
            base = 1 + 3 * (stage - 1) + 1  # 2, 5, ...
            conv[str(base)] = conv2d(c, c, 3, stride=2, groups=c)
            conv[str(base + 1)] = conv2d(c, c, 1, stride=1)
        self.conv = nnx.Dict(conv)
        self.out = nnx.Linear(
            c * freq, cfg.d_model, dtype=dt, param_dtype=dt, rngs=nnx.Rngs(0)
        )
        self._order = ["0", "2", "3", "5", "6"][: 1 + 2 * (self._sampling_num - 1)]

    def __call__(self, x):  # x: (B, T, feat)
        x = x[:, :, :, None]  # (B, T, feat, 1) NHWC with 1 input channel
        first = True
        for key in self._order:
            x = self.conv[key](x)
            # ReLU after conv.0 and after each pointwise conv (pattern conv,relu / dw,pw,relu).
            if first or key in ("3", "6"):
                x = nnx.relu(x)
            first = False
        # x: (B, T', F', C) -> (B, T', C*F')
        b, tp, fp, c = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(b, tp, c * fp)
        return self.out(x)


class Conformer(nnx.Module):
    def __init__(self, cfg: ConformerConfig):
        self.cfg = cfg
        self.pre_encode = DwStridingSubsampling(cfg)
        self.layers = nnx.List([ConformerBlock(cfg) for _ in range(cfg.n_layers)])

    def _pos_emb(self, t, dtype):
        d = self.cfg.d_model
        positions = jnp.arange(t - 1, -t, -1, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, d, 2, dtype=jnp.float32) * -(math.log(10000.0) / d)
        )
        ang = positions * div_term
        # Interleave sin (even indices) and cos (odd) via stack+reshape rather
        # than two strided scatters.
        pe = jnp.stack([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(2 * t - 1, d)
        return pe[None].astype(dtype)  # (1, 2T-1, d)

    def __call__(self, mel):  # mel: (B, T, feat)
        x = self.pre_encode(mel)  # (B, T', d)
        pos_emb = self._pos_emb(x.shape[1], x.dtype)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x
