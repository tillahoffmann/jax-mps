"""TDT (token-and-duration transducer) greedy decoder.

The prediction network (embedding + 2-layer LSTM) and joint network are tiny and
the greedy loop is inherently sequential (each duration jump depends on the
previous argmax), so this runs host-side in numpy over the encoder features. It
mirrors ``ParakeetTDT.decode`` in mlx-audio exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sentencepiece as spm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DecoderWeights:
    embed: np.ndarray  # (vocab+1, pred_hidden)
    lstm: list[tuple]  # per layer: (Wx, Wh, bias)
    enc_w: np.ndarray  # joint enc (joint_hidden, enc_hidden)
    enc_b: np.ndarray
    pred_w: np.ndarray  # joint pred (joint_hidden, pred_hidden)
    pred_b: np.ndarray
    out_w: np.ndarray  # joint out (num_out, joint_hidden)
    out_b: np.ndarray


def load_decoder(t: dict) -> DecoderWeights:
    def a(name):
        return np.asarray(t[name], dtype=np.float32)

    lstm = []
    i = 0
    while f"decoder.prediction.dec_rnn.lstm.{i}.Wx" in t:
        p = f"decoder.prediction.dec_rnn.lstm.{i}"
        lstm.append((a(f"{p}.Wx"), a(f"{p}.Wh"), a(f"{p}.bias")))
        i += 1
    return DecoderWeights(
        embed=a("decoder.prediction.embed.weight"),
        lstm=lstm,
        enc_w=a("joint.enc.weight"),
        enc_b=a("joint.enc.bias"),
        pred_w=a("joint.pred.weight"),
        pred_b=a("joint.pred.bias"),
        out_w=a("joint.joint_net.2.weight"),
        out_b=a("joint.joint_net.2.bias"),
    )


def _lstm_step(x, h, c, Wx, Wh, b):
    g = Wx @ x + Wh @ h + b
    hh = g.shape[0] // 4
    i, f, gg, o = g[:hh], g[hh : 2 * hh], g[2 * hh : 3 * hh], g[3 * hh :]
    i, f, gg, o = _sigmoid(i), _sigmoid(f), np.tanh(gg), _sigmoid(o)
    c2 = f * c + i * gg
    h2 = o * np.tanh(c2)
    return h2, c2


def load_vocabulary(model_path: str) -> tuple[list[str], spm.SentencePieceProcessor]:
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
    return vocab, sp


def _is_special(piece: str) -> bool:
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in (
        "<unk>",
        "<pad>",
    )


def decode_text(tokens: list[int], vocab: list[str]) -> str:
    parts = []
    for tok in tokens:
        if tok < 0 or tok >= len(vocab):
            continue
        piece = vocab[tok]
        if _is_special(piece):
            continue
        parts.append(piece.replace("▁", " "))
    return "".join(parts)


def greedy_decode(
    features: np.ndarray,
    w: DecoderWeights,
    vocab: list[str],
    durations=(0, 1, 2, 3, 4),
    max_symbols: int = 10,
) -> tuple[str, list[int]]:
    """Greedy TDT decode over encoder features (1, T, enc_hidden)."""
    feats = np.asarray(features[0], dtype=np.float32)  # (T, enc_hidden)
    max_length = feats.shape[0]
    blank_id = len(vocab)

    # The joint's encoder projection is independent of decoder state -> precompute
    # it for every frame once instead of inside the sequential loop.
    enc_proj = feats @ w.enc_w.T + w.enc_b  # (T, joint_hidden)
    n_layers = len(w.lstm)
    hidden = w.lstm[0][0].shape[0] // 4

    h = [np.zeros(hidden, np.float32) for _ in range(n_layers)]
    c = [np.zeros(hidden, np.float32) for _ in range(n_layers)]
    last_token = blank_id
    tokens: list[int] = []

    time = 0
    new_symbols = 0
    while time < max_length:
        # Prediction network step (blank -> zero embedding).
        x = (
            np.zeros(w.embed.shape[1], np.float32)
            if last_token == blank_id
            else w.embed[last_token]
        )
        nh, nc = [], []
        for li in range(n_layers):
            hi, ci = _lstm_step(x, h[li], c[li], *w.lstm[li])
            nh.append(hi)
            nc.append(ci)
            x = hi
        dec_out = x  # last-layer hidden

        # Joint network + TDT split.
        j = np.maximum(0.0, enc_proj[time] + w.pred_w @ dec_out + w.pred_b)
        logits = w.out_w @ j + w.out_b
        pred_token = int(np.argmax(logits[: blank_id + 1]))
        decision = int(np.argmax(logits[blank_id + 1 :]))
        duration = durations[decision]

        if pred_token != blank_id:
            last_token = pred_token
            h, c = nh, nc
            if not _is_special(vocab[pred_token]):
                tokens.append(pred_token)

        time += duration
        new_symbols += 1
        if duration != 0:
            new_symbols = 0
        elif max_symbols is not None and max_symbols <= new_symbols:
            time += 1
            new_symbols = 0

    return decode_text(tokens, vocab), tokens
