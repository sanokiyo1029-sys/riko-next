"""The real-time conversation pipeline.

    audio frames -> VAD -> ASR -> memory retrieval -> LLM token stream
      -> sentence chunker -> emotion -> TTS (per sentence, while the LLM keeps
      writing) -> audio frames to the client

``ConversationSession`` is transport-agnostic: the WebSocket server and the
terminal client both drive it through ``feed_audio`` / ``ptt_*`` /
``submit_text`` / ``interrupt`` and receive output through a ``SessionSink``.

Cancellation model: each user turn is one asyncio task. Interrupting cancels
that task, which cancels the TTS worker and closes the LLM HTTP stream (so
local servers stop generating). Only the sentences that actually started
playing are stored in history, so Riko knows what the user really heard.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from ..asr import ASRError, ASRProvider, create_asr
from ..audio.utils import pcm16_to_float32
from ..audio.vad import SpeechEnd, SpeechStart, VADSegmenter
from ..emotion import EmotionAnalyzer, EmotionState, strip_tags
from ..llm import LLMError, LLMProvider, create_llm
from ..memory import MemoryManager, MemoryStore, create_embedder
from ..personality import PersonalityEngine, PersonalityError
from ..tts import SpeechRequest, TTSError, TTSProvider, create_tts
from .chunker import SentenceChunker, ThinkFilter, clean_for_tts
from .config import RikoConfig, is_safe_name
from .events import Ev, make_event
from .logging import get_logger

log = get_logger("session")
_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.S)


class SessionSink(Protocol):
    async def send_event(self, event: dict[str, Any]) -> None: ...
    async def send_audio(self, turn_id: int, sentence_index: int, sample_rate: int, pcm: bytes) -> None: ...


@dataclass
class Services:
    """Long-lived components shared by all sessions."""

    config: RikoConfig
    llm: LLMProvider
    asr: ASRProvider
    tts: TTSProvider
    personality: PersonalityEngine
    memory: Optional[MemoryManager]
    vad_factory: Callable[[], VADSegmenter]

    @classmethod
    def from_config(cls, cfg: RikoConfig) -> "Services":
        llm = create_llm(cfg.llm)
        personality = PersonalityEngine(cfg.resolve_path(cfg.personality.characters_dir),
                                        cfg.personality.character, cfg.personality.preset)
        memory = None
        if cfg.memory.enabled:
            async def complete(msgs: list[dict]) -> str:
                return await llm.complete(msgs, temperature=0.2, max_tokens=350)
            memory = MemoryManager(MemoryStore(cfg.resolve_path(cfg.memory.db_path)),
                                   create_embedder(cfg.memory.embeddings), cfg.memory,
                                   personality.character_id, complete=complete,
                                   character_name=personality.current.name)
        return cls(config=cfg, llm=llm, asr=create_asr(cfg.asr), tts=create_tts(cfg), personality=personality,
                   memory=memory, vad_factory=lambda: VADSegmenter.from_config(cfg.audio.vad))

    async def aclose(self) -> None:
        for c in (self.llm, self.asr, self.tts):
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                pass
        if self.memory:
            try:
                await self.memory.embedder.aclose()
            except Exception:  # noqa: BLE001
                pass
            self.memory.store.close()


@dataclass
class _Sentence:
    index: int
    raw: str
    display: str
    speech: str
    emotion: EmotionState


@dataclass
class _TurnState:
    turn_id: int
    t0: float
    spoken: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    speaking_announced: bool = False


class ConversationSession:
    def __init__(self, services: Services, sink: SessionSink, session_id: Optional[str] = None) -> None:
        cfg = services.config
        self.s = services
        self.sink = sink
        self.id = session_id or uuid.uuid4().hex[:8]
        self.input_mode = cfg.audio.input_mode
        self.muted = False
        self.mic_enabled = True
        self.emotion = EmotionAnalyzer(cfg.emotion.enabled, cfg.emotion.mode, cfg.emotion.min_change)
        self.segmenter = services.vad_factory()
        char_voice = services.personality.current.voice
        self.voice_name = char_voice if char_voice in cfg.tts.voices else cfg.tts.voice
        self.turn_id = 0
        self.state = "listening"
        self.client_playing = False
        self._turn_task: Optional[asyncio.Task[None]] = None
        self._cancel_reason: Optional[str] = None
        self._bg: set[asyncio.Task[Any]] = set()
        self._asr_lock = asyncio.Lock()
        self._ptt_active = False
        self._ptt_frames: list[np.ndarray] = []
        self._barge_pending = False
        self._ignore_current_utterance = False
        self._history: list[dict[str, str]] = []  # used only when memory is disabled
        self._closed = False

    # ================================================================ plumbing
    async def _emit(self, type_: str, **fields: Any) -> None:
        if self._closed:
            return
        try:
            await self.sink.send_event(make_event(type_, **fields))
        except Exception as e:  # noqa: BLE001 - a dead socket must not kill the pipeline
            log.debug("emit failed", extra={"event": type_, "error": str(e)})

    async def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self._emit(Ev.STATE, state=state)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    @property
    def turn_running(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    @property
    def assistant_active(self) -> bool:
        return self.turn_running or self.client_playing

    def settings(self) -> dict[str, Any]:
        cfg = self.s.config
        p = self.s.personality
        return {
            "session_id": self.id, "input_mode": self.input_mode, "muted": self.muted, "mic": self.mic_enabled,
            "character": p.character_id, "character_name": p.current.name, "characters": p.available_characters(),
            "preset": p.preset, "presets": p.presets(), "model": self.s.llm.model, "llm_provider": cfg.llm.provider,
            "voice": self.voice_name, "voices": list(cfg.tts.voices), "tts_provider": cfg.tts.provider,
            "asr_provider": cfg.asr.provider, "memory": bool(self.s.memory), "emotion": cfg.emotion.enabled,
            "barge_in": cfg.audio.barge_in.enabled, "duck_volume": cfg.audio.barge_in.duck_volume,
            "avatar": cfg.frontend.avatar.model_dump(), "show_tokens": cfg.frontend.show_token_stream,
        }

    async def start(self) -> None:
        await self._emit(Ev.HELLO, settings=self.settings(), emotion=self.emotion.current.to_dict())
        await self._emit(Ev.STATE, state=self.state)

    async def close(self) -> None:
        await self._cancel_turn("closed")
        self._closed = True  # background memory tasks keep running; they just stop emitting

    # ================================================================ audio input
    async def feed_audio(self, pcm: bytes) -> None:
        """Mono int16 PCM at 16 kHz, any frame size."""
        if self._closed or not self.mic_enabled or not pcm:
            return
        samples = pcm16_to_float32(pcm)
        if self._ptt_active:
            self._ptt_frames.append(samples)
            if sum(f.size for f in self._ptt_frames) > self.s.config.audio.vad.max_utterance_s * 16000:
                await self.ptt_stop()
            return
        if self.input_mode != "auto":
            return
        for ev in self.segmenter.process(samples):
            if isinstance(ev, SpeechStart):
                await self._on_speech_start()
            elif isinstance(ev, SpeechEnd):
                await self._on_speech_end(ev)
        barge = self.s.config.audio.barge_in
        if self._barge_pending and self.segmenter.speech_ms >= barge.min_speech_ms:
            self._barge_pending = False
            log.info("barge-in", extra={"session": self.id, "turn": self.turn_id})
            await self.interrupt("barge_in")

    async def _on_speech_start(self) -> None:
        await self._emit(Ev.USER_SPEECH_STARTED, mode="auto")
        self._ignore_current_utterance = False
        if self.assistant_active:
            barge = self.s.config.audio.barge_in
            if barge.enabled:
                self._barge_pending = True
                await self._emit(Ev.ASSISTANT_DUCK, volume=barge.duck_volume)
            else:
                self._ignore_current_utterance = True  # probably our own voice leaking into the mic
        else:
            await self._set_state("listening")

    async def _on_speech_end(self, ev: SpeechEnd) -> None:
        await self._emit(Ev.USER_SPEECH_STOPPED, duration_s=round(ev.duration_s, 2))
        if self._barge_pending:  # too short to count as an interruption (cough, echo, "mm-hm")
            self._barge_pending = False
            await self._emit(Ev.ASSISTANT_UNDUCK)
            return
        if self._ignore_current_utterance or ev.duration_s < 0.3:
            return
        self._spawn(self._handle_utterance(ev.audio))

    async def ptt_start(self) -> None:
        if self.assistant_active:
            await self.interrupt("push_to_talk")
        self.segmenter.reset()
        self._ptt_frames = []
        self._ptt_active = True
        await self._emit(Ev.USER_SPEECH_STARTED, mode="ptt")
        await self._set_state("listening")

    async def ptt_stop(self) -> None:
        if not self._ptt_active:
            return
        self._ptt_active = False
        audio = np.concatenate(self._ptt_frames) if self._ptt_frames else np.zeros(0, dtype=np.float32)
        self._ptt_frames = []
        await self._emit(Ev.USER_SPEECH_STOPPED, duration_s=round(audio.size / 16000, 2))
        if audio.size < 0.25 * 16000:
            await self._emit(Ev.TRANSCRIPTION, text="", final=True, empty=True, reason="too_short")
            return
        self._spawn(self._handle_utterance(audio))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        async with self._asr_lock:
            await self._set_state("transcribing")
            try:
                tr = await self.s.asr.transcribe(audio)
            except ASRError as e:
                log.error("ASR failed", extra={"error": str(e)})
                await self._emit(Ev.ERROR, component="asr", message=str(e), recoverable=True)
                await self._set_state("listening")
                return
            log.info("transcribed", extra={"chars": len(tr.text), "audio_s": round(tr.audio_s, 2),
                                           "asr_ms": round(tr.latency_s * 1000)})
            if not tr.text:
                await self._emit(Ev.TRANSCRIPTION, text="", final=True, empty=True, reason="no_speech")
                await self._set_state("listening")
                return
            await self.submit_text(tr.text, source="voice", asr_ms=round(tr.latency_s * 1000))

    # ================================================================ turns
    async def submit_text(self, text: str, source: str = "text", asr_ms: Optional[int] = None) -> None:
        text = " ".join(text.split())[: self.s.config.server.max_text_chars]
        if not text:
            return
        cancelled = await self._cancel_turn("new_turn")
        if not cancelled and self.client_playing:  # previous reply still playing on the client
            await self._emit(Ev.ASSISTANT_STOPPED, turn_id=self.turn_id, reason="new_turn")
        self.client_playing = False
        self.turn_id += 1
        self._cancel_reason = None
        self._turn_task = asyncio.create_task(self._run_turn(self.turn_id, text, source, asr_ms))

    async def interrupt(self, reason: str = "user") -> None:
        cancelled = await self._cancel_turn(reason)
        if not cancelled and self.client_playing:
            await self._emit(Ev.ASSISTANT_STOPPED, turn_id=self.turn_id, reason=reason)
        self.client_playing = False
        await self._set_state("listening")

    async def _cancel_turn(self, reason: str) -> bool:
        task = self._turn_task
        if task is None or task.done():
            return False
        self._cancel_reason = reason
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("turn task failed during cancellation")
        return True

    async def _build_messages(self, text: str) -> list[dict[str, str]]:
        mem = self.s.memory
        memories_txt = working_txt = summary = ""
        history: list[dict[str, str]]
        if mem:
            try:
                found = await asyncio.wait_for(mem.retrieve(text), self.s.config.memory.retrieval_timeout_s)
            except Exception as e:  # noqa: BLE001 - includes TimeoutError; memory is best-effort
                log.warning("memory retrieval skipped", extra={"error": str(e)[:200] or e.__class__.__name__})
                found = []
            if found:
                await self._emit(Ev.MEMORY_RETRIEVED, turn_id=self.turn_id,
                                 items=[{**s.memory.public(), "score": s.score} for s in found])
            memories_txt = mem.format_memories(found)
            working_txt = mem.format_working(mem.working_memories())
            summary = mem.summary()
            history = mem.context_messages()
        else:
            history = self._history[-self.s.config.memory.short_term_messages:]
        system = self.s.personality.build_system_prompt(memories=memories_txt, working=working_txt, summary=summary,
                                                       emotion_instructions=self.emotion.instructions())
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]

    def _remember_message(self, role: str, content: str, turn_id: int, interrupted: bool = False) -> None:
        if self.s.memory:
            self.s.memory.add_message(role, content, turn_id, interrupted)
        else:
            self._history.append({"role": role, "content": content + (" -- (interrupted)" if interrupted else "")})

    async def _run_turn(self, turn_id: int, text: str, source: str, asr_ms: Optional[int]) -> None:
        st = _TurnState(turn_id, time.perf_counter())
        if asr_ms is not None:
            st.metrics["asr_ms"] = asr_ms
        reason = "completed"
        raw: list[str] = []
        queue: asyncio.Queue[Optional[_Sentence]] = asyncio.Queue()
        tts_task: Optional[asyncio.Task[None]] = None
        try:
            await self._emit(Ev.TRANSCRIPTION, text=text, final=True, source=source, turn_id=turn_id)
            await self._set_state("thinking")
            messages = await self._build_messages(text)
            self._remember_message("user", text, turn_id)
            st.metrics["context_ms"] = round((time.perf_counter() - st.t0) * 1000)
            await self._emit(Ev.LLM_STARTED, turn_id=turn_id, model=self.s.llm.model)
            tts_task = asyncio.create_task(self._tts_worker(st, queue))
            think, chunker, idx = ThinkFilter(), SentenceChunker(), 0
            async for delta in self.s.llm.stream_chat(messages):
                if not raw:
                    st.metrics["llm_first_token_ms"] = round((time.perf_counter() - st.t0) * 1000)
                raw.append(delta)
                visible = think.feed(delta)
                if not visible:
                    continue
                if self.s.config.frontend.show_token_stream:
                    await self._emit(Ev.LLM_TOKEN, turn_id=turn_id, text=visible)
                for chunk in chunker.feed(visible):
                    idx = await self._dispatch(turn_id, idx, chunk, queue)
            for chunk in chunker.feed(think.flush()) + chunker.flush():
                idx = await self._dispatch(turn_id, idx, chunk, queue)
            await self._emit(Ev.LLM_FINISHED, turn_id=turn_id, text=strip_tags(self._strip_think("".join(raw))))
            await queue.put(None)
            await tts_task
        except asyncio.CancelledError:
            reason = self._cancel_reason or "interrupted"
            raise
        except LLMError as e:
            reason = "error"
            log.error("LLM failed", extra={"error": str(e), "turn": turn_id})
            await self._emit(Ev.ERROR, component="llm", message=str(e), recoverable=True, turn_id=turn_id)
        except Exception as e:  # noqa: BLE001
            reason = "error"
            log.exception("turn failed")
            await self._emit(Ev.ERROR, component="pipeline", message=f"{e.__class__.__name__}: {e}", turn_id=turn_id)
        finally:
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            full = self._strip_think("".join(raw)).strip()
            stored = full if reason == "completed" else " ".join(st.spoken).strip()
            if stored:
                self._remember_message("assistant", stored, turn_id, interrupted=reason != "completed")
            st.metrics["total_ms"] = round((time.perf_counter() - st.t0) * 1000)
            log.info("turn finished", extra={"turn": turn_id, "reason": reason, **st.metrics})
            await self._emit(Ev.ASSISTANT_STOPPED, turn_id=turn_id, reason=reason)
            await self._emit(Ev.METRICS, turn_id=turn_id, **st.metrics)
            if not self._closed:
                self.state = "speaking" if self.client_playing else "listening"
                await self._emit(Ev.STATE, state=self.state)
            if reason == "completed" and self.s.memory and full:
                self._spawn(self._post_turn_memory(text, full))

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_BLOCK.sub("", text)

    async def _dispatch(self, turn_id: int, idx: int, chunk: str, queue: "asyncio.Queue[Optional[_Sentence]]") -> int:
        display, state = self.emotion.analyze(chunk)
        changed = self.emotion.update(state)
        if changed:
            await self._emit(Ev.EMOTION_CHANGED, turn_id=turn_id, sentence_index=idx, **changed.to_dict())
        if not display:
            return idx
        await self._emit(Ev.LLM_SENTENCE, turn_id=turn_id, index=idx, text=display)
        speech = clean_for_tts(display, self.s.config.tts.strip_actions)
        await queue.put(_Sentence(idx, chunk, display, speech, self.emotion.current))
        return idx + 1

    async def _announce_speaking(self, st: _TurnState) -> None:
        if not st.speaking_announced:
            st.speaking_announced = True
            st.metrics["first_audio_ms"] = round((time.perf_counter() - st.t0) * 1000)
            await self._emit(Ev.ASSISTANT_SPEAKING, turn_id=st.turn_id)
            await self._set_state("speaking")

    async def _tts_worker(self, st: _TurnState, queue: "asyncio.Queue[Optional[_Sentence]]") -> None:
        tts = self.s.tts
        tts_down = False
        while True:
            job = await queue.get()
            if job is None:
                return
            if not job.speech or self.muted or tts.name == "none" or tts_down:
                st.spoken.append(job.raw)
                continue
            if not tts.produces_audio:  # browser speechSynthesis fallback
                st.spoken.append(job.raw)
                await self._announce_speaking(st)
                await self._emit(Ev.TTS_SPEAK, turn_id=st.turn_id, index=job.index, text=job.speech,
                                 emotion=job.emotion.emotion, speed=job.emotion.speed_factor)
                continue
            await self._emit(Ev.TTS_STARTED, turn_id=st.turn_id, index=job.index)
            req = SpeechRequest(job.speech, self.s.config.tts.voices[self.voice_name], job.emotion.emotion,
                                job.emotion.intensity, job.emotion.speed_factor)
            first = True
            try:
                async for chunk in tts.synthesize(req):
                    if first:
                        first = False
                        st.spoken.append(job.raw)
                        await self._announce_speaking(st)
                    await self.sink.send_audio(st.turn_id, job.index, chunk.sample_rate, chunk.pcm)
            except TTSError as e:
                log.error("TTS failed", extra={"error": str(e), "sentence": job.index})
                await self._emit(Ev.ERROR, component="tts", message=str(e), recoverable=True, turn_id=st.turn_id)
                tts_down = e.unavailable  # don't hammer a dead server for every sentence
                if first:
                    st.spoken.append(job.raw)  # the text was still shown to the user
            await self._emit(Ev.TTS_FINISHED, turn_id=st.turn_id, index=job.index)

    async def _post_turn_memory(self, user_text: str, assistant_text: str) -> None:
        """Extract memories once the conversation is idle so a local LLM isn't busy
        with bookkeeping while the user is waiting for a reply."""
        mem = self.s.memory
        if mem is None:
            return
        try:
            deadline = time.monotonic() + 60
            idle_since: Optional[float] = None
            while time.monotonic() < deadline:
                if self.assistant_active or self._ptt_active or self.segmenter.in_speech:
                    idle_since = None
                else:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= 0.8:
                        break
                await asyncio.sleep(0.2)
            created = await mem.extract_from_turn(user_text, assistant_text)
            for m in created:
                await self._emit(Ev.MEMORY_CREATED, **m.public())
            await mem.maybe_summarize()
        except Exception:  # noqa: BLE001
            log.exception("post-turn memory processing failed")

    # ================================================================ client state / settings
    async def set_playback(self, playing: bool) -> None:
        self.client_playing = playing
        if not playing and not self.turn_running:
            await self._set_state("listening")

    async def apply_settings(self, **kw: Any) -> None:
        cfg = self.s.config
        try:
            if kw.get("input_mode") in ("auto", "ptt"):
                self.input_mode = kw["input_mode"]
                self.segmenter.reset()
            if kw.get("muted") is not None:
                self.muted = bool(kw["muted"])
            if kw.get("mic") is not None:
                self.mic_enabled = bool(kw["mic"])
                if not self.mic_enabled:
                    self.segmenter.reset()
            if kw.get("character"):
                self.s.personality.load(kw["character"])
                if self.s.memory:
                    self.s.memory.set_character(self.s.personality.character_id, self.s.personality.current.name)
            if kw.get("preset"):
                self.s.personality.set_preset(kw["preset"])
            if kw.get("voice"):
                if kw["voice"] not in cfg.tts.voices:
                    raise PersonalityError(f"unknown voice '{kw['voice']}'")
                self.voice_name = kw["voice"]
            if kw.get("model"):
                if not is_safe_name(kw["model"]):
                    raise PersonalityError("invalid model name")
                self.s.llm.model = kw["model"]
        except PersonalityError as e:
            await self._emit(Ev.ERROR, component="settings", message=str(e), recoverable=True)
        await self._emit(Ev.SETTINGS, settings=self.settings())
