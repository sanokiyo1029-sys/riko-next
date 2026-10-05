"""Voice activity detection and utterance segmentation.

``SileroVAD`` streams the Silero v6 ONNX model that ships *inside*
faster-whisper (no torch, no extra download). ``EnergyVAD`` is a pure-numpy
fallback with an adaptive noise floor. ``VADSegmenter`` turns per-frame speech
probabilities into utterances with pre-roll, hysteresis and endpointing.
"""
from __future__ import annotations

import collections
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Protocol, Union

import numpy as np

from ..core.config import VADConfig
from ..core.logging import get_logger
from .utils import rms_dbfs

log = get_logger("vad")

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms at 16 kHz (Silero's native frame)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE


class VADEngine(Protocol):
    def speech_prob(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


def find_silero_model(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    try:
        from faster_whisper.vad import get_assets_path  # type: ignore
    except Exception:
        return None
    for name in ("silero_vad_v6.onnx", "silero_vad_v5.onnx", "silero_vad.onnx"):
        p = Path(get_assets_path()) / name
        if p.is_file():
            return p
    return None


class SileroVAD:
    """Stateful, frame-by-frame Silero VAD (v5/v6 ONNX, 16 kHz, 512-sample frames)."""

    CONTEXT = 64

    def __init__(self, model_path: Union[str, os.PathLike[str]]) -> None:
        import onnxruntime as ort  # bundled with faster-whisper

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=opts)
        names = {i.name for i in self._sess.get_inputs()}
        if not {"input", "h", "c"} <= names:
            raise RuntimeError(f"unsupported Silero model inputs: {sorted(names)}")
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(self.CONTEXT, dtype=np.float32)

    def speech_prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._ctx, frame.astype(np.float32, copy=False)])[None, :]
        out, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = frame[-self.CONTEXT:].astype(np.float32, copy=True)
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Adaptive-noise-floor energy detector. Works anywhere; less robust to noise."""

    def __init__(self, margin_db: float = 12.0) -> None:
        self.margin_db = margin_db
        self.reset()

    def reset(self) -> None:
        self._floor = -60.0

    def speech_prob(self, frame: np.ndarray) -> float:
        db = rms_dbfs(frame)
        # The floor tracks quiet frames quickly and loud frames slowly.
        rate = 0.05 if db < self._floor + self.margin_db else 0.002
        self._floor = max(-90.0, (1 - rate) * self._floor + rate * db)
        x = (db - (self._floor + self.margin_db)) / 3.0
        return 1.0 / (1.0 + math.exp(-x))


def create_vad_engine(cfg: VADConfig) -> VADEngine:
    if cfg.engine == "silero":
        path = find_silero_model()
        if path:
            try:
                return SileroVAD(path)
            except Exception as e:  # noqa: BLE001
                log.warning("Silero VAD unavailable, falling back to energy VAD", extra={"error": str(e)})
        else:
            log.warning("Silero model not found (install faster-whisper); falling back to energy VAD")
    return EnergyVAD(cfg.energy_margin_db)


@dataclass
class SpeechStart:
    pass


@dataclass
class SpeechEnd:
    audio: np.ndarray
    duration_s: float
    forced: bool = False


VADEvent = Union[SpeechStart, SpeechEnd]


@dataclass
class VADSegmenter:
    """Frame-level probabilities -> utterances.

    * speech starts after ``min_speech_ms`` of consecutive voiced frames;
    * speech ends after ``silence_ms`` below ``threshold - 0.15`` (hysteresis);
    * ``preroll_ms`` of audio before the start is kept so first syllables
      aren't clipped (a common cause of bad transcriptions).
    """

    engine: VADEngine
    threshold: float = 0.5
    min_speech_ms: int = 200
    silence_ms: int = 650
    preroll_ms: int = 300
    max_utterance_s: float = 30.0
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _preroll: Deque[np.ndarray] = field(default_factory=collections.deque)
    _speech: list[np.ndarray] = field(default_factory=list)
    _voiced_run: int = 0
    _silent_run: int = 0
    in_speech: bool = False
    last_prob: float = 0.0

    @classmethod
    def from_config(cls, cfg: VADConfig, engine: Optional[VADEngine] = None) -> "VADSegmenter":
        return cls(engine=engine or create_vad_engine(cfg), threshold=cfg.threshold,
                   min_speech_ms=cfg.min_speech_ms, silence_ms=cfg.silence_ms,
                   preroll_ms=cfg.preroll_ms, max_utterance_s=cfg.max_utterance_s)

    _frames_since_start: int = 0

    @property
    def speech_ms(self) -> float:
        """Voiced time in the current utterance (silent gaps excluded). Used for barge-in,
        so a cough followed by silence never counts as a sustained interruption."""
        return self._frames_since_start * FRAME_MS if self.in_speech else 0.0

    def reset(self) -> None:
        self.engine.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._speech.clear()
        self._voiced_run = self._silent_run = 0
        self.in_speech = False

    def process(self, samples: np.ndarray) -> list[VADEvent]:
        events: list[VADEvent] = []
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = buf.size // FRAME
        for i in range(n):
            ev = self._process_frame(buf[i * FRAME:(i + 1) * FRAME])
            if ev:
                events.append(ev)
        self._pending = buf[n * FRAME:].copy()
        return events

    def _process_frame(self, frame: np.ndarray) -> Optional[VADEvent]:
        prob = self.engine.speech_prob(frame)
        self.last_prob = prob
        min_frames = max(1, int(self.min_speech_ms / FRAME_MS))
        if not self.in_speech:
            self._preroll.append(frame)
            max_pre = max(min_frames, int(self.preroll_ms / FRAME_MS)) + min_frames
            while len(self._preroll) > max_pre:
                self._preroll.popleft()
            self._voiced_run = self._voiced_run + 1 if prob >= self.threshold else 0
            if self._voiced_run >= min_frames:
                self.in_speech = True
                self._silent_run = 0
                self._speech = list(self._preroll)
                self._preroll.clear()
                self._frames_since_start = min_frames
                return SpeechStart()
            return None

        self._speech.append(frame)
        if prob < max(0.05, self.threshold - 0.15):
            self._silent_run += 1
        else:
            self._silent_run = 0
            self._frames_since_start += 1
        if self._silent_run * FRAME_MS >= self.silence_ms:
            return self._finish(forced=False)
        if len(self._speech) * FRAME_MS >= self.max_utterance_s * 1000:
            return self._finish(forced=True)
        return None

    def _finish(self, forced: bool) -> SpeechEnd:
        # trim most of the trailing silence, keep ~150 ms
        keep_tail = int(150 / FRAME_MS)
        frames = self._speech if forced else self._speech[: max(1, len(self._speech) - self._silent_run + keep_tail)]
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        self._speech = []
        self._voiced_run = self._silent_run = 0
        self.in_speech = False
        if forced:
            self.engine.reset()
        return SpeechEnd(audio=audio, duration_s=audio.size / SAMPLE_RATE, forced=forced)
