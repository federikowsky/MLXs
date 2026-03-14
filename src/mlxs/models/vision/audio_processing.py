"""Audio preprocessing: WAV/raw audio -> mel spectrogram features (§7.4, FR12).

Converts raw waveform data into log-mel spectrogram features suitable for
the Gemma3n Conformer audio encoder. Uses numpy for FFT/mel filterbank
computation (no heavy audio library dependencies beyond soundfile for I/O).
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a mel filterbank matrix ``(n_mels, n_fft // 2 + 1)``."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    bin_idx = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bin_idx[i], bin_idx[i + 1], bin_idx[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


# ---------------------------------------------------------------------------
# STFT + log-mel
# ---------------------------------------------------------------------------


def _stft(
    waveform: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Short-time Fourier transform (magnitude spectrogram).

    Returns shape ``(n_frames, n_fft // 2 + 1)``.
    """
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)

    # Pad so that the last frame is centred
    pad_len = n_fft // 2
    waveform = np.pad(waveform, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(n_frames, n_fft),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0]),
    ).copy()

    frames *= window
    spec = np.fft.rfft(frames, n=n_fft)
    return np.abs(spec).astype(np.float32)


def compute_log_mel(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    f_min: float = 0.0,
    f_max: float | None = None,
    log_floor: float = 1e-10,
) -> mx.array:
    """Convert a 1-D waveform to a log-mel spectrogram ``(1, T, n_mels)``.

    This is a lightweight alternative to ``librosa.feature.melspectrogram``
    that uses only numpy for FFT computation.
    """
    waveform = waveform.astype(np.float32)
    mag = _stft(waveform, n_fft=n_fft, hop_length=hop_length)  # (T, n_fft//2+1)
    fb = _mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)  # (n_mels, n_fft//2+1)
    mel = mag @ fb.T  # (T, n_mels)
    log_mel = np.log(np.maximum(mel, log_floor))
    return mx.array(log_mel[None])  # (1, T, n_mels)


# ---------------------------------------------------------------------------
# High-level preprocessing
# ---------------------------------------------------------------------------


def load_audio(
    source: str | bytes | io.IOBase,
    *,
    target_sr: int = 16000,
) -> np.ndarray:
    """Load audio from file path, bytes, or file-like object.

    Returns a 1-D float32 numpy array at ``target_sr`` Hz.
    Requires the ``soundfile`` package (optional dependency).
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for audio processing. "
            "Install with: pip install 'mlxs[audio]'"
        ) from e

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    data, sr = sf.read(source, dtype="float32", always_2d=False)
    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Simple resample via linear interpolation if needed
    if sr != target_sr:
        ratio = target_sr / sr
        n_out = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n_out)
        idx_lo = np.floor(indices).astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, len(data) - 1)
        frac = (indices - idx_lo).astype(np.float32)
        data = data[idx_lo] * (1 - frac) + data[idx_hi] * frac

    return data.astype(np.float32)


def preprocess_audio(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_mels: int = 80,
    max_length: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Preprocess raw audio waveform into mel features + mask.

    Args:
        waveform: 1-D float32 numpy array.
        sample_rate: Sample rate of the waveform.
        n_mels: Number of mel filterbank channels.
        max_length: Optional max number of mel frames. Truncates if exceeded.

    Returns:
        Tuple of:
        - ``audio_mel``: ``(1, T, n_mels)`` mel spectrogram.
        - ``audio_mel_mask``: ``(1, T)`` boolean mask (True = padded).
    """
    mel = compute_log_mel(waveform, sample_rate=sample_rate, n_mels=n_mels)
    T = mel.shape[1]

    if max_length is not None and T > max_length:
        mel = mel[:, :max_length]
        T = max_length

    mask = mx.zeros((1, T), dtype=mx.bool_)  # no padding
    return mel, mask


def preprocess_audio_batch(
    waveforms: list[np.ndarray],
    *,
    sample_rate: int = 16000,
    n_mels: int = 80,
) -> tuple[mx.array, mx.array]:
    """Batch-preprocess multiple audio waveforms with padding.

    Returns:
        Tuple of:
        - ``audio_mel``: ``(B, T_max, n_mels)`` padded mel spectrograms.
        - ``audio_mel_mask``: ``(B, T_max)`` boolean mask (True = padded).
    """
    mels = [
        compute_log_mel(w, sample_rate=sample_rate, n_mels=n_mels)
        for w in waveforms
    ]
    lengths = [m.shape[1] for m in mels]
    max_t = max(lengths)

    padded = []
    masks = []
    for mel, length in zip(mels, lengths):
        if length < max_t:
            pad_n = max_t - length
            mel = mx.pad(mel, [(0, 0), (0, pad_n), (0, 0)])
        padded.append(mel)
        mask = mx.concatenate([
            mx.zeros((1, length), dtype=mx.bool_),
            mx.ones((1, max_t - length), dtype=mx.bool_),
        ], axis=1)
        masks.append(mask)

    return mx.concatenate(padded, axis=0), mx.concatenate(masks, axis=0)
