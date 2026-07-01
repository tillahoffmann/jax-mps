"""Log-mel front-end for Parakeet, matching NeMo / mlx-audio preprocessing.

The mel filterbank and analysis window are host-side constants (built once with
numpy); the STFT + power + mel-projection + per-feature normalization run in JAX
so the whole front-end is jittable and can execute on the MPS backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16000
    features: int = 128
    n_fft: int = 512
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    normalize: str = "per_feature"
    preemph: float | None = 0.97
    log_zero_guard: float = 2.0**-24

    @property
    def window_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


# --- host-side constants (numpy), replicating mlx_audio.utils exactly ---


def hann_window(size: int) -> np.ndarray:
    # Symmetric (periodic=False) Hann window, matching mlx_audio.utils.hanning.
    denom = size - 1
    n = np.arange(size, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * n / denom))


def _hz_to_mel_slaney(freq: float) -> float:
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mels >= min_log_mel,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        freqs,
    )


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Slaney-normalized triangular mel filterbank, shape (n_mels, n_fft//2+1).

    Mirrors ``mlx_audio.utils.mel_filters(..., norm="slaney", mel_scale="slaney")``
    (float32 path) so the resulting mel spectrogram matches token-for-token.
    """
    f_max = sample_rate / 2
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sample_rate // 2, n_freqs, dtype=np.float32)

    m_min = _hz_to_mel_slaney(0.0)
    m_max = _hz_to_mel_slaney(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = _mel_to_hz_slaney(m_pts).astype(np.float32)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = (-slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(np.zeros_like(down), np.minimum(down, up))

    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    fb = fb * enorm[None, :]
    return np.moveaxis(fb, 0, 1)  # (n_mels, n_freqs)


def build_frontend(cfg: PreprocessConfig):
    """Return the analysis (window, filterbank) as float32 host constants.

    They are float32 regardless of the model dtype: the STFT runs in float32 for
    numerical stability and to match mlx-audio's (non-precise) mel path, and
    log_mel casts the waveform to float32 too.
    """
    window = hann_window(cfg.window_length)
    # Center-pad the window to n_fft (matches torch.stft / NeMo / mlx-audio).
    if window.shape[0] < cfg.n_fft:
        left = (cfg.n_fft - window.shape[0]) // 2
        right = cfg.n_fft - window.shape[0] - left
        window = np.concatenate([np.zeros(left), window, np.zeros(right)])
    filterbank = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.features)
    return jnp.asarray(window, jnp.float32), jnp.asarray(filterbank, jnp.float32)


def log_mel(
    x: jnp.ndarray, cfg: PreprocessConfig, window: jnp.ndarray, filterbank: jnp.ndarray
):
    """Compute a normalized log-mel spectrogram, shape (1, num_frames, features).

    ``x`` is a 1-D waveform (float32). ``window`` is the centered analysis window
    (length n_fft) and ``filterbank`` is (features, n_fft//2+1) from build_frontend.
    """
    n_fft = cfg.n_fft
    hop = cfg.hop_length
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)

    # Pre-emphasis high-pass filter: y[n] = x[n] - a*x[n-1].
    if cfg.preemph:
        x = jnp.concatenate([x[:1], x[1:] - cfg.preemph * x[:-1]])

    # center=True: reflect-free constant (zero) padding of n_fft//2 each side.
    pad = n_fft // 2
    x = jnp.pad(x, (pad, pad))

    num_frames = 1 + (x.shape[0] - n_fft) // hop
    idx = jnp.arange(num_frames)[:, None] * hop + jnp.arange(n_fft)[None, :]
    frames = x[idx]  # (num_frames, n_fft)

    spec = jnp.fft.rfft(frames * window)  # (num_frames, n_fft//2+1)
    power = spec.real**2 + spec.imag**2  # |spec|^2 without the abs()'s sqrt

    mel = filterbank @ power.T  # (features, num_frames)
    mel = jnp.log(mel + cfg.log_zero_guard)

    if cfg.normalize == "per_feature":
        mean = jnp.mean(mel, axis=1, keepdims=True)
        n = max(mel.shape[1] - 1, 1)
        var = jnp.sum((mel - mean) ** 2, axis=1, keepdims=True) / n
        std = jnp.sqrt(var)
        mel = (mel - mean) / (std + 1e-5)
    else:
        mel = (mel - jnp.mean(mel)) / (jnp.std(mel) + 1e-5)

    mel = mel.T[None]  # (1, num_frames, features)
    return mel.astype(orig_dtype)


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file as a 1-D float32 waveform at ``sample_rate`` (mono)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return np.ascontiguousarray(wav, dtype=np.float32)
