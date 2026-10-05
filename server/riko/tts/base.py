"""Provider-agnostic text-to-speech interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from ..core.config import VoiceConfig


class TTSError(Exception):
    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.unavailable = unavailable  # server down: don't retry every sentence


@dataclass
class SpeechRequest:
    text: str
    voice: VoiceConfig
    emotion: Optional[str] = None
    intensity: float = 0.5
    speed_factor: float = 1.0


@dataclass
class AudioChunk:
    pcm: bytes  # int16 little-endian mono
    sample_rate: int


class TTSProvider(ABC):
    name = "base"
    #: False for providers whose audio is produced client-side (browser speechSynthesis)
    produces_audio = True

    @abstractmethod
    def synthesize(self, req: SpeechRequest) -> AsyncIterator[AudioChunk]: ...

    async def health(self) -> tuple[bool, str]:
        return True, "ok"

    async def aclose(self) -> None:
        return None


class ClientSideTTS(TTSProvider):
    """No server audio. ``browser`` = the web client speaks with Web Speech API;
    ``none`` = text only. Both are real, working fallbacks when GPT-SoVITS isn't set up."""

    produces_audio = False

    def __init__(self, name: str) -> None:
        self.name = name

    async def synthesize(self, req: SpeechRequest) -> AsyncIterator[AudioChunk]:
        return
        yield  # pragma: no cover - makes this an async generator
