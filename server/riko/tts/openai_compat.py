"""OpenAI-compatible ``/v1/audio/speech`` TTS.

Works with OpenAI and with local servers such as Kokoro-FastAPI, which expose
the same endpoint. Requests ``response_format: pcm`` (raw int16, 24 kHz by
OpenAI's spec) so audio can be streamed without container parsing.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

import httpx

from ..core.config import OpenAITTSConfig, get_secret
from .base import AudioChunk, SpeechRequest, TTSError, TTSProvider


class OpenAICompatibleTTS(TTSProvider):
    name = "openai_compatible"

    def __init__(self, cfg: OpenAITTSConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.cfg = cfg
        key = get_secret(cfg.api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), headers=headers,
                                         timeout=httpx.Timeout(cfg.timeout_s, connect=3.0), transport=transport)

    async def synthesize(self, req: SpeechRequest) -> AsyncIterator[AudioChunk]:
        voice_id = req.voice.voice_id or "alloy"
        speed = req.voice.speed_factor * req.speed_factor
        override = req.voice.emotions.get(req.emotion or "")
        if override and override.speed_factor:
            speed = override.speed_factor * req.speed_factor
        body = {"model": self.cfg.model, "input": req.text, "voice": voice_id,
                "response_format": "pcm", "speed": round(max(0.25, min(4.0, speed)), 3)}
        carry = b""
        try:
            async with self._client.stream("POST", "/audio/speech", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise TTSError(f"TTS HTTP {resp.status_code}: {text[:300]}")
                async for data in resp.aiter_bytes():
                    data = carry + data
                    usable = len(data) - (len(data) % 2)
                    carry = data[usable:]
                    if usable:
                        yield AudioChunk(data[:usable], self.cfg.sample_rate)
        except httpx.ConnectError as e:
            raise TTSError(f"cannot connect to TTS server at {self.cfg.base_url}", unavailable=True) from e
        except httpx.TimeoutException as e:
            raise TTSError("TTS request timed out") from e

    async def health(self) -> tuple[bool, str]:
        try:
            r = await self._client.get("/models", timeout=3.0)
            return True, f"reachable (HTTP {r.status_code})"
        except httpx.HTTPError as e:
            return False, f"unreachable at {self.cfg.base_url} ({e.__class__.__name__})"

    async def aclose(self) -> None:
        await self._client.aclose()
