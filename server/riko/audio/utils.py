"""Small, dependency-free audio helpers (numpy + stdlib only)."""
from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler. Adequate for speech playback/ASR input."""
    if src_rate == dst_rate or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(x.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_out = np.linspace(0, x.size - 1, n_out)
    return np.interp(t_out, np.arange(x.size), x).astype(np.float32)


def encode_wav(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


@dataclass
class WavFormat:
    sample_rate: int
    channels: int
    bits: int
    float_data: bool
    data_offset: int


def parse_wav_header(data: bytes) -> Optional[WavFormat]:
    """Parse a RIFF/WAVE header, tolerating streaming headers (data size 0 / 0xFFFFFFFF).

    Returns None if more bytes are needed.
    """
    if len(data) < 12:
        return None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("response is not a WAV stream")
    pos = 12
    fmt: Optional[tuple[int, int, int, int]] = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if cid == b"fmt ":
            if body + 16 > len(data):
                return None
            audio_fmt, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if audio_fmt == 0xFFFE and body + 26 <= len(data):  # WAVE_FORMAT_EXTENSIBLE
                audio_fmt = struct.unpack_from("<H", data, body + 24)[0]
            fmt = (audio_fmt, channels, rate, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk before fmt chunk")
            audio_fmt, channels, rate, bits = fmt
            return WavFormat(rate, channels, bits, audio_fmt == 3, body)
        pos = body + size + (size & 1)
    return None


def to_mono_pcm16(raw: bytes, fmt: WavFormat) -> bytes:
    """Convert a block of WAV sample data to mono int16 (caller keeps frame alignment)."""
    if fmt.bits == 16 and not fmt.float_data:
        arr = np.frombuffer(raw, dtype="<i2")
        if fmt.channels > 1:
            arr = arr.reshape(-1, fmt.channels).mean(axis=1).astype("<i2")
        return arr.tobytes()
    if fmt.bits == 32 and fmt.float_data:
        arr32 = np.frombuffer(raw, dtype="<f4")
        if fmt.channels > 1:
            arr32 = arr32.reshape(-1, fmt.channels).mean(axis=1)
        return float32_to_pcm16(arr32)
    raise ValueError(f"unsupported WAV sample format: {fmt.bits}-bit float={fmt.float_data}")


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))
