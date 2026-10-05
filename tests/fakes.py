"""Test doubles. They implement the real provider interfaces; only the model is fake."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from riko.asr import ASRProvider, Transcript
from riko.audio.vad import EnergyVAD, VADSegmenter
from riko.core.config import RikoConfig
from riko.core.session import Services
from riko.llm import LLMProvider
from riko.memory import HashEmbedder, MemoryManager, MemoryStore
from riko.personality import PersonalityEngine
from riko.tts import AudioChunk, SpeechRequest, TTSProvider

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply: str = "[happy] Hello senpai! It's nice to see you again. How was your day?",
                 delay: float = 0.0) -> None:
        super().__init__("fake-model")
        self.reply, self.delay, self.calls = reply, delay, []

    async def stream_chat(self, messages, *, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        self.calls.append(messages)
        last = messages[-1]["content"]
        if "long-term memory" in last:
            if "Alex" in last.split("Exchange:")[-1]:
                yield '```json\n[{"layer": "semantic", "content": "User\'s name is Alex", "importance": 8, "category": "identity"}]\n```'
            else:
                yield "[]"
            return
        if "running summary" in last or "Merge these" in last:
            yield "Summary of earlier things."
            return
        for tok in re.findall(r"\S+\s*", self.reply):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield tok

    async def list_models(self):
        return ["fake-model", "other-model"]


class FakeASR(ASRProvider):
    name = "fake"

    def __init__(self, text: str = "Hello Riko") -> None:
        self.text, self.calls = text, 0

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls += 1
        return Transcript(self.text, "en", audio.size / 16000, 0.01)


class FakeTTS(TTSProvider):
    name = "fake"

    def __init__(self, chunk_delay: float = 0.0, chunks: int = 3) -> None:
        self.texts: list[str] = []
        self.requests: list[SpeechRequest] = []
        self.chunk_delay, self.chunks = chunk_delay, chunks

    async def synthesize(self, req: SpeechRequest) -> AsyncIterator[AudioChunk]:
        self.texts.append(req.text)
        self.requests.append(req)
        t = np.arange(2400) / 24000
        pcm = (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()
        for _ in range(self.chunks):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield AudioChunk(pcm, 24000)


def test_config(tmp_path: Path, **memory_kw) -> RikoConfig:
    cfg = RikoConfig.model_validate({
        "memory": {"db_path": str(tmp_path / "riko.db"), **memory_kw},
        "audio": {"vad": {"engine": "energy"}},
    })
    cfg.root = ROOT
    return cfg


def make_services(tmp_path: Path, llm=None, asr=None, tts=None, memory: bool = True,
                  cfg: Optional[RikoConfig] = None) -> Services:
    cfg = cfg or test_config(tmp_path)
    llm = llm or FakeLLM()
    personality = PersonalityEngine(ROOT / "characters", "riko", "default")
    mem = None
    if memory:
        async def complete(msgs):
            return await llm.complete(msgs)
        mem = MemoryManager(MemoryStore(tmp_path / "riko.db"), HashEmbedder(256), cfg.memory, "riko",
                            complete=complete, character_name="Riko")
    return Services(config=cfg, llm=llm, asr=asr or FakeASR(), tts=tts or FakeTTS(), personality=personality,
                    memory=mem, vad_factory=lambda: VADSegmenter.from_config(cfg.audio.vad, EnergyVAD(12.0)))


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.audio: list[tuple[int, int, int, bytes]] = []

    async def send_event(self, event: dict) -> None:
        self.events.append(event)

    async def send_audio(self, turn_id, sentence_index, sample_rate, pcm) -> None:
        self.audio.append((turn_id, sentence_index, sample_rate, pcm))

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of(self, t: str) -> list[dict]:
        return [e for e in self.events if e["type"] == t]


def speech_audio(speech_s: float = 1.0, lead_s: float = 0.6, tail_s: float = 1.0, amp: float = 0.3) -> np.ndarray:
    """Quiet noise, a loud voiced-like tone, quiet noise (float32, 16 kHz)."""
    rng = np.random.default_rng(1)
    t = np.arange(int(speech_s * 16000)) / 16000
    voiced = amp * np.sin(2 * np.pi * 160 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    noise = lambda s: 0.001 * rng.standard_normal(int(s * 16000))
    return np.concatenate([noise(lead_s), voiced, noise(tail_s)]).astype(np.float32)


def to_frames(x: np.ndarray, n: int = 512) -> list[bytes]:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    return [pcm[i:i + n].tobytes() for i in range(0, pcm.size, n)]
