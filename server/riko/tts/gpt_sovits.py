"""GPT-SoVITS ``api_v2.py`` client (``POST /tts``).

Wire format (verified against RVC-Boss/GPT-SoVITS api_v2.py):
  * non-streaming: a complete WAV file;
  * streaming (``streaming_mode`` 1-3 with ``media_type=wav``): first a WAV
    header with 0 frames, then raw int16 PCM chunks. HTTP chunk boundaries
    may fall mid-sample, so an odd trailing byte is carried over.
  * failure: HTTP 400 with JSON ``{"message": ..., "Exception": ...}``.

Emotion support: GPT-SoVITS has no emotion parameter. The established
technique is a different *reference clip* per emotion (a happy clip makes a
happy voice) plus speed. ``VoiceConfig.emotions`` maps emotion -> overrides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import httpx

from ..audio.utils import WavFormat, parse_wav_header, to_mono_pcm16
from ..core.config import GPTSoVITSConfig
from ..core.logging import get_logger
from .base import AudioChunk, SpeechRequest, TTSError, TTSProvider

log = get_logger("tts")


class GPTSoVITSTTS(TTSProvider):
    name = "gpt_sovits"

    def __init__(self, cfg: GPTSoVITSConfig, resolve_path: Callable[[str], Path],
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.cfg = cfg
        self._resolve = resolve_path
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"),
                                         timeout=httpx.Timeout(cfg.timeout_s, connect=3.0), transport=transport)

    def _server_path(self, p: str) -> str:
        """GPT-SoVITS opens ref audio from *its* filesystem/cwd: send absolute paths."""
        local = self._resolve(p)
        if self.cfg.remote_path_prefix:
            return f"{self.cfg.remote_path_prefix.rstrip('/')}/{local.name}"
        if not local.is_file():
            raise TTSError(f"reference audio not found: {local}")
        return str(local)

    def build_payload(self, req: SpeechRequest) -> dict:
        voice = req.voice
        ref, prompt, speed = voice.ref_audio, voice.prompt_text, voice.speed_factor
        override = voice.emotions.get(req.emotion or "")
        if override:
            ref = override.ref_audio or ref
            if override.ref_audio:
                prompt = override.prompt_text or ""
            speed = override.speed_factor or speed
        if not ref:
            raise TTSError("tts voice has no ref_audio configured")
        payload = {
            "text": req.text,
            "text_lang": self.cfg.text_lang,
            "ref_audio_path": self._server_path(ref),
            "prompt_text": prompt,
            "prompt_lang": voice.prompt_lang,
            "media_type": "wav",
            "streaming_mode": self.cfg.streaming_mode,
            "text_split_method": self.cfg.text_split_method,
            "batch_size": 1,
            "speed_factor": round(speed * req.speed_factor, 3),
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "temperature": self.cfg.temperature,
        }
        if voice.aux_ref_audio:
            payload["aux_ref_audio_paths"] = [self._server_path(p) for p in voice.aux_ref_audio]
        return payload

    async def synthesize(self, req: SpeechRequest) -> AsyncIterator[AudioChunk]:
        payload = self.build_payload(req)
        try:
            async with self._client.stream("POST", "/tts", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    try:
                        detail = json.loads(body)
                        body = f"{detail.get('message', '')} {detail.get('Exception', '')}".strip()
                    except ValueError:
                        pass
                    raise TTSError(f"GPT-SoVITS HTTP {resp.status_code}: {body[:300]}")
                header = b""
                fmt: Optional[WavFormat] = None
                carry = b""
                async for data in resp.aiter_bytes():
                    if fmt is None:
                        header += data
                        fmt = parse_wav_header(header)
                        if fmt is None:
                            continue
                        data = header[fmt.data_offset:]
                    block = fmt.channels * fmt.bits // 8
                    data = carry + data
                    usable = len(data) - (len(data) % block)
                    carry = data[usable:]
                    if usable:
                        yield AudioChunk(to_mono_pcm16(data[:usable], fmt), fmt.sample_rate)
        except httpx.ConnectError as e:
            raise TTSError(f"cannot connect to GPT-SoVITS at {self.cfg.base_url} - start api_v2.py first",
                           unavailable=True) from e
        except httpx.TimeoutException as e:
            raise TTSError("GPT-SoVITS request timed out") from e

    async def health(self) -> tuple[bool, str]:
        try:
            # api_v2 has no health route; any HTTP answer means the server is up.
            r = await self._client.get("/tts", timeout=3.0)
            return True, f"reachable (HTTP {r.status_code})"
        except httpx.HTTPError as e:
            return False, f"unreachable at {self.cfg.base_url} ({e.__class__.__name__})"

    async def aclose(self) -> None:
        await self._client.aclose()
