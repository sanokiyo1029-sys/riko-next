"""Text-to-speech providers. Select with ``tts.provider`` in config.yaml.

Adding an engine: subclass ``TTSProvider``, implement ``synthesize`` as an
async generator of ``AudioChunk`` (int16 mono PCM + sample rate), and add a
branch to ``create_tts``.
"""
from __future__ import annotations

from typing import Optional

import httpx

from ..core.config import RikoConfig
from .base import AudioChunk, ClientSideTTS, SpeechRequest, TTSError, TTSProvider
from .gpt_sovits import GPTSoVITSTTS
from .openai_compat import OpenAICompatibleTTS

__all__ = ["AudioChunk", "ClientSideTTS", "SpeechRequest", "TTSError", "TTSProvider", "GPTSoVITSTTS",
           "OpenAICompatibleTTS", "create_tts"]


def create_tts(cfg: RikoConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> TTSProvider:
    t = cfg.tts
    if t.provider == "gpt_sovits":
        return GPTSoVITSTTS(t.gpt_sovits, cfg.resolve_path, transport=transport)
    if t.provider == "openai_compatible":
        return OpenAICompatibleTTS(t.openai_compatible, transport=transport)
    return ClientSideTTS(t.provider)
