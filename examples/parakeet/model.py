"""Parakeet Conformer encoder in Flax NNX.

Faithful reimplementation of NVIDIA's FastConformer encoder as used by
mlx-audio's Parakeet-TDT (global ``rel_pos`` attention). This is the
compute-heavy part of the model and is JIT-compiled onto the MPS backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


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


def _silu(x):
    return x * jax.nn.sigmoid(x)


def _glu(x, axis=-1):
    a, b = jnp.split(x, 2, axis=axis)
    return a * jax.nn.sigmoid(b)


class Conv1d(nnx.Module):
    """Channels-last (NWC) 1-D conv; kernel stored WIO = (k, in/groups, out)."""

    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        *,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        dtype=jnp.float32,
    ):
        self.stride, self.padding, self.groups = stride, padding, groups
        self.kernel = nnx.Param(jnp.zeros((k, in_ch // groups, out_ch), dtype))
        self.bias = nnx.Param(jnp.zeros((out_ch,), dtype)) if bias else None

    def __call__(self, x):
        y = lax.conv_general_dilated(
            x,
            self.kernel.value,
            (self.stride,),
            [(self.padding, self.padding)],
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias.value
        return y


class Conv2d(nnx.Module):
    """Channels-last (NHWC) 2-D conv; kernel stored HWIO = (kh, kw, in/groups, out)."""

    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        *,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        dtype=jnp.float32,
    ):
        self.stride, self.padding, self.groups = stride, padding, groups
        self.kernel = nnx.Param(jnp.zeros((k, k, in_ch // groups, out_ch), dtype))
        self.bias = nnx.Param(jnp.zeros((out_ch,), dtype)) if bias else None

    def __call__(self, x):
        s, p = self.stride, self.padding
        y = lax.conv_general_dilated(
            x,
            self.kernel.value,
            (s, s),
            [(p, p), (p, p)],
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias.value
        return y


class BatchNorm1d(nnx.Module):
    """Inference-only batch norm over the channel (last) axis."""

    def __init__(self, ch, eps=1e-5, dtype=jnp.float32):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((ch,), dtype))
        self.bias = nnx.Param(jnp.zeros((ch,), dtype))
        self.running_mean = nnx.Param(jnp.zeros((ch,), dtype))
        self.running_var = nnx.Param(jnp.ones((ch,), dtype))

    def __call__(self, x):
        xhat = (x - self.running_mean.value) * lax.rsqrt(
            self.running_var.value + self.eps
        )
        return xhat * self.weight.value + self.bias.value


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
        return self.linear2(_silu(self.linear1(x)))


class RelPositionMultiHeadAttention(nnx.Module):
    def __init__(self, n_head, d_model, *, dtype=jnp.float32):
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim**-0.5

        def lin():
            return nnx.Linear(
                d_model,
                d_model,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                rngs=nnx.Rngs(0),
            )

        self.linear_q, self.linear_k, self.linear_v = lin(), lin(), lin()
        self.linear_out = lin()
        self.linear_pos = lin()
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
        q = self.linear_q(x).reshape(b, t, h, d)
        k = self.linear_k(x).reshape(b, t, h, d).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(b, t, h, d).transpose(0, 2, 1, 3)
        p = (
            self.linear_pos(pos_emb)
            .reshape(pos_emb.shape[0], -1, h, d)
            .transpose(0, 2, 1, 3)
        )

        q_u = (q + self.pos_bias_u.value).transpose(0, 2, 1, 3)  # (B,H,T,d)
        q_v = (q + self.pos_bias_v.value).transpose(0, 2, 1, 3)

        matrix_ac = jnp.matmul(q_u, k.transpose(0, 1, 3, 2))  # (B,H,T,T)
        matrix_bd = jnp.matmul(q_v, p.transpose(0, 1, 3, 2))  # (B,H,T,2T-1)
        matrix_bd = self._rel_shift(matrix_bd)[:, :, :, : k.shape[2]]  # (B,H,T,T)

        scores = (matrix_ac + matrix_bd) * self.scale
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, v)  # (B,H,T,d)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, h * d)
        return self.linear_out(out)


class Convolution(nnx.Module):
    def __init__(self, cfg: ConformerConfig):
        d, dt = cfg.d_model, cfg.dtype
        self.pad = (cfg.conv_kernel_size - 1) // 2
        self.pointwise_conv1 = Conv1d(d, 2 * d, 1, bias=False, dtype=dt)
        self.depthwise_conv = Conv1d(
            d, d, cfg.conv_kernel_size, padding=self.pad, groups=d, bias=False, dtype=dt
        )
        self.batch_norm = BatchNorm1d(d, dtype=dt)
        self.pointwise_conv2 = Conv1d(d, d, 1, bias=False, dtype=dt)

    def __call__(self, x):  # x: (B, T, d)
        x = self.pointwise_conv1(x)
        x = _glu(x, axis=-1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = _silu(x)
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
        # conv dict index matches checkpoint keys (ReLU layers occupy 1/4/7).
        conv = {"0": Conv2d(1, c, 3, stride=2, padding=1, dtype=dt)}
        for stage in range(1, self._sampling_num):
            base = 1 + 3 * (stage - 1) + 1  # 2, 5, ...
            conv[str(base)] = Conv2d(c, c, 3, stride=2, padding=1, groups=c, dtype=dt)
            conv[str(base + 1)] = Conv2d(c, c, 1, stride=1, padding=0, dtype=dt)
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
                x = jax.nn.relu(x)
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
        pe = jnp.zeros((2 * t - 1, d), jnp.float32)
        pe = pe.at[:, 0::2].set(jnp.sin(ang))
        pe = pe.at[:, 1::2].set(jnp.cos(ang))
        return pe[None].astype(dtype)  # (1, 2T-1, d)

    def __call__(self, mel):  # mel: (B, T, feat)
        x = self.pre_encode(mel)  # (B, T', d)
        pos_emb = self._pos_emb(x.shape[1], x.dtype)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x
