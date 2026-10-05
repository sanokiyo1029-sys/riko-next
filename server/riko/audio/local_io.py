"""Local microphone/speaker I/O for terminal mode (``riko chat``), via sounddevice.

The browser client gets echo cancellation for free; a terminal does not. In
auto (VAD) mode with speakers, Riko can hear herself, so terminal mode turns
barge-in off unless you pass ``--headphones``.
"""
from __future__ import annotations

import asyncio
import collections
import threading
from typing import Any, Callable, Deque, Optional

import numpy as np

from .utils import pcm16_to_float32, resample_linear


def _sd() -> Any:
    try:
        import sounddevice as sd  # type: ignore
    except (ImportError, OSError) as e:
        raise RuntimeError("sounddevice/PortAudio is not available. Run: pip install -e \".[local-audio]\" "
                           "(Linux also needs: sudo apt install libportaudio2)") from e
    return sd


def _device(name: Optional[str]) -> Any:
    if name is None:
        return None
    return int(name) if name.isdigit() else name


class MicStream:
    """Pushes 16 kHz int16 mono frames into an asyncio callback."""

    def __init__(self, loop: asyncio.AbstractEventLoop, on_frame: Callable[[bytes], Any],
                 device: Optional[str] = None) -> None:
        sd = _sd()
        self._loop = loop
        self._on_frame = on_frame

        def cb(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            loop.call_soon_threadsafe(self._on_frame, bytes(indata))

        self._stream = sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=512,
                                         device=_device(device), callback=cb)

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class Player:
    """Streams PCM chunks of any sample rate to the default output device.

    Supports instant flush (barge-in) and ducking; reports playing/idle changes.
    """

    def __init__(self, on_state: Callable[[bool], None], device: Optional[str] = None) -> None:
        sd = _sd()
        info = sd.query_devices(_device(device), "output")
        self.rate = int(info["default_samplerate"])
        self._buf: Deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._current: Optional[np.ndarray] = None
        self._pos = 0
        self.volume = 1.0
        self._playing = False
        self._on_state = on_state
        self._stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="float32",
                                       device=_device(device), callback=self._cb, blocksize=0)
        self._stream.start()

    def _cb(self, out: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        out.fill(0)
        written = 0
        with self._lock:
            while written < frames:
                if self._current is None or self._pos >= self._current.size:
                    if not self._buf:
                        break
                    self._current, self._pos = self._buf.popleft(), 0
                n = min(frames - written, self._current.size - self._pos)
                out[written:written + n, 0] = self._current[self._pos:self._pos + n] * self.volume
                self._pos += n
                written += n
            playing = written > 0 or bool(self._buf)
        if playing != self._playing:
            self._playing = playing
            self._on_state(playing)

    def play(self, pcm: bytes, sample_rate: int) -> None:
        data = resample_linear(pcm16_to_float32(pcm), sample_rate, self.rate)
        with self._lock:
            self._buf.append(data)

    def flush(self) -> None:
        with self._lock:
            self._buf.clear()
            self._current = None

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
