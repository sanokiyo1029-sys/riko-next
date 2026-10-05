"""Speech-to-text providers.

* ``faster_whisper`` - local, the original Riko's choice. Improvements over the
  original: model loaded once and warmed up at startup, audio passed in-memory
  (no temp WAV files), GPU auto-detect with automatic CPU fallback when
  CUDA/cuDNN libraries are missing, beam size 1 by default for latency, and
  segment-level hallucination filtering.
* ``openai_compatible`` / ``groq`` - remote Whisper (Groq is what Rayen moved
  to in his Patreon v1.1 build; also works with OpenAI and local servers such
  as speaches that expose ``/v1/audio/transcriptions``).
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import numpy as np

from ..audio.utils import encode_wav, float32_to_pcm16
from ..core.config import ASRConfig, get_secret
from ..core.logging import get_logger

log = get_logger("asr")


class ASRError(Exception):
    pass


@dataclass
class Transcript:
    text: str
    language: Optional[str]
    audio_s: float
    latency_s: float


class ASRProvider(ABC):
    name = "base"

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> Transcript:
        """``audio``: mono float32 at 16 kHz."""

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class FasterWhisperASR(ASRProvider):
    name = "faster_whisper"

    def __init__(self, cfg: ASRConfig, model: Any = None) -> None:
        self.cfg = cfg
        self._model = model  # injectable for tests
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self.device: Optional[str] = None

    def _load_sync(self) -> Any:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise ASRError("faster-whisper is not installed. Run: pip install -e \".[asr]\"") from e
        device = self.cfg.device
        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2  # type: ignore
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:  # noqa: BLE001
                pass
        attempts = [device] + (["cpu"] if device == "cuda" else [])
        last: Optional[Exception] = None
        for dev in attempts:
            compute = self.cfg.compute_type
            if compute == "auto" or (dev == "cpu" and compute in ("float16", "int8_float16")):
                compute = "float16" if dev == "cuda" else "int8"
            try:
                t0 = time.perf_counter()
                model = WhisperModel(self.cfg.model, device=dev, compute_type=compute,
                                     cpu_threads=self.cfg.cpu_threads)
                # Run once so CUDA kernels / cuDNN load now, not mid-conversation.
                segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language=self.cfg.language)
                list(segs)
                self.device = dev
                log.info("faster-whisper ready", extra={"model": self.cfg.model, "device": dev,
                                                        "compute_type": compute,
                                                        "load_s": round(time.perf_counter() - t0, 2)})
                return model
            except Exception as e:  # noqa: BLE001
                last = e
                if dev == "cuda":
                    log.warning("CUDA load failed (missing cuBLAS/cuDNN?) - falling back to CPU",
                                extra={"error": str(e)[:200]})
        raise ASRError(f"could not load faster-whisper model '{self.cfg.model}': {last}")

    async def warmup(self) -> None:
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_sync)

    def _transcribe_sync(self, audio: np.ndarray) -> tuple[str, Optional[str]]:
        segments, info = self._model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,  # already segmented by our VAD
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.cfg.initial_prompt,
        )
        kept = []
        for seg in segments:
            # Classic Whisper hallucination guard ("Thanks for watching!" on near-silence)
            if getattr(seg, "no_speech_prob", 0.0) > 0.6 and getattr(seg, "avg_logprob", 0.0) < -1.0:
                continue
            kept.append(seg.text.strip())
        return " ".join(t for t in kept if t).strip(), getattr(info, "language", None)

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        await self.warmup()
        t0 = time.perf_counter()
        async with self._infer_lock:
            try:
                text, lang = await asyncio.to_thread(self._transcribe_sync, audio.astype(np.float32, copy=False))
            except Exception as e:  # noqa: BLE001
                raise ASRError(f"transcription failed: {e}") from e
        return Transcript(text, lang, audio.size / 16000, time.perf_counter() - t0)


class OpenAICompatibleASR(ASRProvider):
    name = "openai_compatible"

    def __init__(self, cfg: ASRConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.cfg = cfg
        key = get_secret(cfg.api_key_env)
        if cfg.provider == "groq" and not key:
            raise ASRError(f"asr.provider 'groq' needs an API key: set {cfg.api_key_env} in .env")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(base_url=(cfg.base_url or "").rstrip("/"), headers=headers,
                                         timeout=httpx.Timeout(cfg.timeout_s, connect=5.0), transport=transport)

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        wav = encode_wav(float32_to_pcm16(audio), 16000)
        data = {"model": self.cfg.remote_model, "response_format": "json", "temperature": "0"}
        if self.cfg.language:
            data["language"] = self.cfg.language
        if self.cfg.initial_prompt:
            data["prompt"] = self.cfg.initial_prompt
        try:
            r = await self._client.post("/audio/transcriptions", data=data,
                                        files={"file": ("speech.wav", wav, "audio/wav")})
        except httpx.HTTPError as e:
            raise ASRError(f"cannot reach ASR server {self._client.base_url}: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise ASRError(f"ASR HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        return Transcript(str(body.get("text", "")).strip(), body.get("language"),
                          audio.size / 16000, time.perf_counter() - t0)

    async def aclose(self) -> None:
        await self._client.aclose()


def create_asr(cfg: ASRConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> ASRProvider:
    if cfg.provider == "faster_whisper":
        return FasterWhisperASR(cfg)
    return OpenAICompatibleASR(cfg, transport=transport)
