"""The real-time conversation pipeline.

    audio frames -> VAD -> ASR -> context (memory, mood, familiarity, image) -> LLM event stream
      -> [tool calls -> validated execution -> results -> LLM]* -> sentence chunker -> emotion
      -> TTS (per sentence, while the LLM keeps writing) -> audio frames to the client

``ConversationSession`` is transport-agnostic: the WebSocket server and the terminal client both drive it
through ``feed_audio`` / ``ptt_*`` / ``submit_text`` / ``interrupt`` and receive output through a
``SessionSink``.

Cancellation: each turn is one asyncio task. Interrupting cancels it, which cancels the TTS worker and
closes the LLM HTTP stream (local servers stop generating). Only sentences that actually started playing
are stored for an interrupted reply, so Riko knows what the user really heard (I03). Reasoning text never
leaves the provider layer (I01); stale turns can never emit audio (I02).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Protocol

import httpx
import numpy as np

from ..asr import ASRError, ASRProvider, create_asr
from ..audio.utils import pcm16_to_float32
from ..audio.vad import SpeechEnd, SpeechStart, VADSegmenter
from ..emotion import (EMOTIONS, EmotionAnalyzer, EmotionState, Mood, MoodModel, classify_user_affect,
                       strip_tags)
from ..llm import LLMError, LLMEvent, LLMProvider, ModelRouter
from ..memory import MemoryManager, MemoryStore, Status, create_embedder
from ..personality import STAGE_LABELS, PersonalityEngine, PersonalityError, describe_model
from ..tools import ToolContext, ToolRegistry, default_registry
from ..tts import SpeechRequest, TTSError, TTSProvider, create_tts
from .chunker import SentenceChunker, ThinkFilter, clean_for_tts
from .config import RikoConfig, is_safe_name
from .context import (ContextOverflow, ImageStore, context_message, familiarity_stage, fit_history,
                      humanize_gap, local_now)
from .events import PROTOCOL_VERSION, Ev, make_event
from .logging import get_logger
from .scheduler import Scheduler

log = get_logger("session")
_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.S)
_ACT = re.compile(r"\[\s*act\s*:\s*([a-z_]+)\s*\]\s*", re.I)
_ACTIONS = {"wander", "hop", "nap", "follow", "inspect", "edge", "come"}
USER_SOURCES = ("text", "voice")
FALLBACKS = {
    "empty": "Sorry, I lost my train of thought there. Could you say that again?",
    "timeout": "Sorry, that's taking me too long. Could you try again in a moment?",
}

TOOLS_NOTE = ("You have tools. Use one only when it is actually needed for the answer: exact arithmetic, the "
              "current time in another timezone, the user's saved memories, notes, tasks and reminders, and "
              "anything else listed. Most replies need no tool. You may chain several tool calls. Before a slow tool, a few words like \"Let me check.\" are fine. Never claim "
              "something was saved, scheduled, found or deleted unless a tool result says so.")

VISION_PROMPT = ("Describe this image for an assistant that cannot see it. Be factual and specific: main "
                 "subjects, layout, notable details, and transcribe any visible text or code exactly. "
                 "If the user asked something about it, focus on that. User's message: {q}")


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
    router: Optional[ModelRouter] = None
    tools: Optional[ToolRegistry] = None
    scheduler: Optional[Scheduler] = None
    images: ImageStore = field(default_factory=ImageStore)
    http: Optional[httpx.AsyncClient] = None
    mood_model: Optional[MoodModel] = None
    status: dict[str, Any] = field(default_factory=dict)  # readiness facts for /api/status

    def __post_init__(self) -> None:
        cfg = self.config
        if self.router is None:
            self.router = ModelRouter(cfg, primary=self.llm)
        if self.tools is None:
            self.tools = default_registry(cfg)
        if self.scheduler is None:
            self.scheduler = Scheduler(cfg.memory.max_pending_jobs, cfg.memory.job_expiry_minutes * 60,
                                       cfg.memory.job_max_attempts, idle_s=cfg.memory.background_idle_s)
        if self.http is None and "network" in cfg.tools.permissions:
            self.http = httpx.AsyncClient(timeout=10.0, follow_redirects=True,
                                          headers={"User-Agent": "riko-next (local assistant)"})
        self.reset_mood_model()

    def reset_mood_model(self) -> None:
        bm = self.personality.current.baseline_mood
        e = self.config.emotion
        self.mood_model = MoodModel(bm.valence, bm.arousal, e.mood_half_life_minutes, e.mood_max_step)

    @classmethod
    def from_config(cls, cfg: RikoConfig) -> "Services":
        router = ModelRouter(cfg)
        llm = router.get("fast")
        assert llm is not None
        personality = PersonalityEngine(cfg.resolve_path(cfg.personality.characters_dir),
                                        cfg.personality.character, cfg.personality.preset)
        memory = None
        if cfg.memory.enabled:
            async def complete(msgs: list[dict], json_schema: Optional[dict] = None) -> str:
                worker = router.get("fast") or llm
                return await worker.complete(msgs, temperature=0.2, json_schema=json_schema)
            memory = MemoryManager(MemoryStore(cfg.resolve_path(cfg.memory.db_path)),
                                   create_embedder(cfg.memory.embeddings), cfg.memory,
                                   personality.character_id, complete=complete,
                                   character_name=personality.current.name)
        return cls(config=cfg, llm=llm, asr=create_asr(cfg.asr), tts=create_tts(cfg), personality=personality,
                   memory=memory, vad_factory=lambda: VADSegmenter.from_config(cfg.audio.vad), router=router)

    async def aclose(self) -> None:
        if self.scheduler:
            await self.scheduler.aclose()
        for c in (self.asr, self.tts):
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                pass
        if self.router:
            await self.router.aclose()
        else:
            await self.llm.aclose()
        if self.http:
            await self.http.aclose()
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
    source: str = "text"
    spoken: list[str] = field(default_factory=list)       # sentences whose audio started (or were voiced)
    displayed: list[str] = field(default_factory=list)    # sentences shown to the user
    emotions: list[EmotionState] = field(default_factory=list)
    seen_sentences: set[str] = field(default_factory=set)
    metrics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    speaking_announced: bool = False
    thinking_announced: bool = False
    progressed: bool = False        # any content or tool call yet (first-content deadline)
    spoken_words: int = 0
    spoken_sentences: int = 0
    speech_capped: bool = False
    finish_reason: Optional[str] = None

    def ms(self) -> int:
        return round((time.perf_counter() - self.t0) * 1000)


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
        self.client_ready = False
        self.proactive_level = cfg.proactive.notification_level if cfg.proactive.enabled else "off"
        self._turn_task: Optional[asyncio.Task[None]] = None
        self._turn_source = "text"
        self._cancel_reason: Optional[str] = None
        self._bg: set[asyncio.Task[Any]] = set()
        self._asr_lock = asyncio.Lock()
        self._ptt_active = False
        self._ptt_frames: list[np.ndarray] = []
        self._barge_pending = False
        self._ignore_current_utterance = False
        self._history: list[dict[str, str]] = []  # used only when memory is disabled
        self._closed = False
        self._static_cache: tuple[Any, str] = (None, "")
        self.last_image_id: Optional[str] = None
        self._tts_notice_sent = False
        self.last_activity = time.monotonic()
        self.user_turns = 0
        self.logical_key = uuid.uuid4().hex[:12]
        rel = self._relationship()
        now = time.time()
        # Stage and "time since last talk" are computed once, before this session updates them.
        self.stage = familiarity_stage(rel, cfg.memory.familiarity, now) if rel else 0
        self.last_seen_gap_s = (now - rel["last_seen"]) if rel and rel.get("last_seen") else None
        self.new_logical_session = (self.last_seen_gap_s is None or
                                    self.last_seen_gap_s > cfg.memory.familiarity.session_gap_minutes * 60)
        services.scheduler.busy_checks[self.id] = self._busy_for_background
        from .proactive import ProactiveController
        self.proactive = ProactiveController(self)

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

    def _busy_for_background(self) -> bool:
        # also busy while speech is being transcribed and shortly after anyone spoke: a reply is likely next
        return (self.assistant_active or self._ptt_active or self.segmenter.in_speech or self._asr_lock.locked()
                or time.monotonic() - self.last_activity < self.s.scheduler.idle_s)

    @property
    def turn_running(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    @property
    def assistant_active(self) -> bool:
        return self.turn_running or self.client_playing

    @property
    def memory(self) -> Optional[MemoryManager]:
        m = self.s.memory
        return m if m is not None and m.enabled else None

    def _vrm_found(self) -> bool:
        from ..api.server import WEB_DIR  # local import: the api package imports this module
        url = self.s.config.frontend.avatar.vrm_url
        if not url or "://" in url:
            return bool(url)
        p = (WEB_DIR / url).resolve()
        return p.is_relative_to(WEB_DIR.resolve()) and p.is_file()

    def _relationship(self) -> Optional[dict[str, Any]]:
        m = self.s.memory
        return m.store.get_relationship(m.character, profile=m.profile) if m else None

    def settings(self) -> dict[str, Any]:
        cfg = self.s.config
        p = self.s.personality
        vision = self.s.router.config_for("vision")
        return {
            "session_id": self.id, "protocol": PROTOCOL_VERSION, "input_mode": self.input_mode, "muted": self.muted,
            "mic": self.mic_enabled, "character": p.character_id, "character_name": p.current.name,
            "characters": p.available_characters(), "preset": p.preset, "presets": p.presets(),
            "model": self.s.llm.model, "llm_provider": cfg.llm.provider,
            "reasoning_effort": cfg.llm.reasoning_effort, "vision_model": vision.model if vision else None,
            "voice": self.voice_name, "voices": list(cfg.tts.voices), "tts_provider": cfg.tts.provider,
            "asr_provider": cfg.asr.provider, "memory": self.s.memory is not None,
            "memory_enabled": bool(self.memory), "emotion": cfg.emotion.enabled, "emotions": list(EMOTIONS),
            "tools": cfg.tools.enabled, "proactive_level": self.proactive_level,
            "barge_in": cfg.audio.barge_in.enabled, "duck_volume": cfg.audio.barge_in.duck_volume,
            "avatar": {**cfg.frontend.avatar.model_dump(), "vrm_found": self._vrm_found()},
            "browser_voice": cfg.tts.browser_voice, "browser_pitch": cfg.tts.browser_pitch,
            "show_tokens": cfg.frontend.show_token_stream,
            "familiarity": STAGE_LABELS.get(self.stage, "unknown"),
        }

    async def start(self) -> None:
        await self._emit(Ev.HELLO, settings=self.settings(), emotion=self.emotion.current.to_dict(),
                         mood=self._mood_public())
        await self._emit(Ev.STATE, state=self.state)
        self.proactive.start()

    async def close(self) -> None:
        self.proactive.stop()
        await self._cancel_turn("closed")
        self._closed = True  # background memory jobs keep running; they just stop emitting
        self.s.scheduler.busy_checks.pop(self.id, None)

    def note_activity(self) -> None:
        self.last_activity = time.monotonic()

    async def set_client_ready(self) -> None:
        self.client_ready = True
        self.note_activity()

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
        self.note_activity()
        self.proactive.user_spoke()
        await self._emit(Ev.USER_SPEECH_STARTED, mode="auto")
        self._ignore_current_utterance = False
        self.s.scheduler.preempt()  # user speech outranks background work
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
        self.note_activity()
        await self._emit(Ev.USER_SPEECH_STOPPED, duration_s=round(ev.duration_s, 2))
        if self._barge_pending:  # too short to count as an interruption (cough, echo, "mm-hm")
            self._barge_pending = False
            await self._emit(Ev.ASSISTANT_UNDUCK)
            return
        if self._ignore_current_utterance or ev.duration_s < 0.3:
            return
        self._spawn(self._handle_utterance(ev.audio))

    async def ptt_start(self) -> None:
        self.proactive.user_spoke()
        if self.assistant_active:
            await self.interrupt("push_to_talk")
        self.segmenter.reset()
        self._ptt_frames = []
        self._ptt_active = True
        self.s.scheduler.preempt()
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
    async def submit_text(self, text: str, source: str = "text", asr_ms: Optional[int] = None,
                          image_id: Optional[str] = None) -> None:
        text = " ".join(text.split())[: self.s.config.server.max_text_chars]
        if not text and not image_id:
            return
        if source in USER_SOURCES:
            self.note_activity()
            self.proactive.user_spoke()
        await self._start_turn(text or "(sent an image)", source, asr_ms, image_id)

    async def _start_turn(self, text: str, source: str, asr_ms: Optional[int] = None,
                          image_id: Optional[str] = None, extra: Optional[dict[str, Any]] = None) -> None:
        cancelled = await self._cancel_turn("new_turn")
        if not cancelled and self.client_playing:  # previous reply still playing on the client
            await self._emit(Ev.ASSISTANT_STOPPED, turn_id=self.turn_id, reason="new_turn")
        self.client_playing = False
        self.turn_id += 1
        self._cancel_reason = None
        self._turn_source = source
        self._turn_task = asyncio.create_task(self._run_turn(self.turn_id, text, source, asr_ms, image_id, extra))

    async def proactive_turn(self, kind: str, instruction: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Assistant-initiated turn (greeting, nudge, reminder). Typed source; never stored as a user turn."""
        await self._start_turn(instruction, kind, extra=extra)

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

    # ---------------------------------------------------------------- context
    def _static_prompt(self) -> str:
        cfg = self.s.config
        p = self.s.personality
        key = (p.character_id, p.preset, self.s.llm.model, cfg.tools.enabled, cfg.emotion.enabled, cfg.emotion.mode)
        if self._static_cache[0] != key:
            text = p.static_prompt(describe_model(cfg.llm.provider, self.s.llm.model),
                                   emotion_instructions=self.emotion.instructions(),
                                   tools_note=TOOLS_NOTE if cfg.tools.enabled else "")
            self._static_cache = (key, text)
        return self._static_cache[1]

    def _mood(self) -> Optional[Mood]:
        m = self.s.memory
        if not (self.s.config.emotion.mood_enabled and m):
            return None
        row = m.store.get_mood(m.character, profile=m.profile)
        return self.s.mood_model.decayed(Mood(*row) if row else None, time.time())

    def _mood_public(self) -> Optional[dict[str, Any]]:
        mood = self._mood()
        return MoodModel.to_dict(mood) if mood else None

    async def _build_messages(self, st: _TurnState, text: str, image_note: Optional[dict[str, Any]],
                              extra: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.s.config
        mem = self.memory
        now_dt = local_now(cfg.proactive.timezone)
        data: dict[str, Any] = {"now": now_dt.strftime("%A %d %B %Y, %H:%M") + f" ({now_dt.tzname()})",
                                "familiarity": STAGE_LABELS.get(self.stage)}
        note = self.s.personality.stage_note(self.stage)
        if note:
            data["tone_for_this_stage"] = note
        if self.user_turns == 0 and st.source in USER_SOURCES + ("greeting",):
            data["last_talked"] = humanize_gap(self.last_seen_gap_s) or "never before"
        mood = self._mood()
        if mood:
            data["your_mood"] = MoodModel.describe(mood)
        if st.source in USER_SOURCES:
            affect = classify_user_affect(text)
            if affect.confidence >= 0.45:
                data["user_seems"] = f"{affect.label} (a guess from their wording; may be wrong)"
                if affect.label in ("sad", "frustrated", "anxious"):
                    data["tone"] = "go easy on teasing; be direct and kind"
            if st.source == "voice":
                data["reply_mode"] = "spoken: keep it to one or two short sentences unless asked for detail"
        history: list[dict[str, Any]]
        if mem:
            found = []
            if st.source in USER_SOURCES:
                try:
                    found = await asyncio.wait_for(mem.retrieve(text), cfg.memory.retrieval_timeout_s)
                except Exception as e:  # noqa: BLE001 - includes TimeoutError; memory is best-effort
                    log.warning("memory retrieval skipped", extra={"error": str(e)[:200] or e.__class__.__name__})
            if found:
                await self._emit(Ev.MEMORY_RETRIEVED, turn_id=st.turn_id,
                                 items=[{**s.memory.public(), "score": s.score} for s in found])
            label = lambda m: m.content + (" (uncertain)" if m.confidence < 0.6 or m.status == Status.DISPUTED  # noqa: E731
                                           else "")
            core = mem.core_memories()
            core_ids = {m.id for m in core}
            data["core_facts"] = [label(m) for m in core]
            data["relevant_memories"] = [label(s.memory) for s in found if s.memory.id not in core_ids]
            data["current_goals"] = [label(m) for m in mem.working_memories()]
            data["earlier_conversation_summary"] = mem.summary()
            history = mem.context_messages()
        else:
            history = self._history[-cfg.memory.short_term_messages:]
        if image_note:
            data["image_from_user"] = image_note
        if extra:
            data.update(extra)
        static = {"role": "system", "content": self._static_prompt()}
        if st.source in USER_SOURCES:
            final = {"role": "user", "content": text}
        else:  # app event: request-only, clearly labelled, never stored as something the user said
            final = {"role": "user", "content": f"(Automatic app event, not typed by the user.) {text}"}
        budget = cfg.llm.num_ctx - cfg.llm.max_tokens - cfg.llm.safety_margin_tokens
        ctx = context_message(data)
        try:
            history, est, dropped = fit_history(static, history, ctx, final, budget)
        except ContextOverflow:
            # Optional context goes first; the current request is never silently truncated.
            for k in ("relevant_memories", "earlier_conversation_summary", "current_goals", "image_from_user"):
                data.pop(k, None)
            ctx = context_message(data)
            history, est, dropped = fit_history(static, [], ctx, final, budget)
        st.metrics["prompt_tokens_est"] = est
        if dropped:
            st.metrics["history_dropped"] = dropped
        return [static, *history, ctx, final]

    def _remember_message(self, role: str, content: str, turn_id: int, interrupted: bool = False,
                          source: str = "user") -> Optional[int]:
        if self.memory:
            return self.memory.add_message(role, content, turn_id, interrupted, source=source)
        # memory off/absent: keep a session-local window only, nothing is persisted
        self._history.append({"role": role, "content": content + (" -- (interrupted)" if interrupted else "")})
        return None

    # ---------------------------------------------------------------- vision
    async def _describe_image(self, image_id: str, question: str) -> Optional[dict[str, Any]]:
        url = self.s.images.get(image_id)
        if url is None:
            await self._emit(Ev.ERROR, component="vision", message="That image expired; please send it again.",
                             recoverable=True)
            return {"note": "the user sent an image, but it could not be loaded"}
        self.last_image_id = image_id
        vision = self.s.router.get("vision")
        if vision is None:
            await self._emit(Ev.ERROR, component="vision", recoverable=True,
                             message="No vision model is configured (models.vision), so I can't see images.")
            return {"note": "the user sent an image, but no vision model is configured, so you cannot see it"}
        t = time.perf_counter()
        try:
            desc = await asyncio.wait_for(vision.complete(
                [{"role": "user", "content": VISION_PROMPT.format(q=question[:500]), "images": [url]}],
                max_tokens=1200), self.s.config.llm.overall_timeout_s)
        except (LLMError, asyncio.TimeoutError) as e:
            await self._emit(Ev.ERROR, component="vision", message=f"image analysis failed: {e}", recoverable=True)
            return {"note": "the user sent an image, but image analysis failed"}
        desc = strip_tags(_THINK_BLOCK.sub("", desc)).strip()[:4000]
        await self._emit(Ev.IMAGE_ANALYZED, image_id=image_id, model=vision.model,
                         ms=round((time.perf_counter() - t) * 1000))
        return {"described_by": f"vision model {vision.model}", "description": desc or "(no description)"}

    async def _ask_image(self, question: str) -> str:
        if not self.last_image_id:
            return "no image has been shared in this conversation"
        note = await self._describe_image(self.last_image_id, question)
        return (note or {}).get("description") or (note or {}).get("note", "")

    # ---------------------------------------------------------------- the turn
    async def _run_turn(self, turn_id: int, text: str, source: str, asr_ms: Optional[int],
                        image_id: Optional[str], extra: Optional[dict[str, Any]]) -> None:
        st = _TurnState(turn_id, time.perf_counter(), source)
        if asr_ms is not None:
            st.metrics["asr_finalize_ms"] = st.metrics["asr_ms"] = asr_ms
        reason = "completed"
        raw: list[str] = []
        queue: asyncio.Queue[Optional[_Sentence]] = asyncio.Queue()
        tts_task: Optional[asyncio.Task[None]] = None
        user_msg_id: Optional[int] = None
        started_at = time.time()
        try:
            if source in USER_SOURCES:
                await self._emit(Ev.TRANSCRIPTION, text=text, final=True, source=source, turn_id=turn_id,
                                 image_id=image_id)
            await self._set_state("thinking")
            image_note = await self._describe_image(image_id, text) if image_id else None
            messages = await self._build_messages(st, text, image_note, extra)
            if source in USER_SOURCES:
                stored = text + (" [shared an image]" if image_id else "")
                user_msg_id = self._remember_message("user", stored, turn_id, source="user")
            st.metrics["context_ms"] = st.ms()
            await self._emit(Ev.LLM_STARTED, turn_id=turn_id, model=self.s.llm.model, source=source)
            tts_task = asyncio.create_task(self._tts_worker(st, queue))
            async with self.s.scheduler.foreground():
                st.metrics["foreground_queue_ms"] = st.ms() - st.metrics["context_ms"]
                try:
                    await self._generate(st, messages, queue, raw)
                except (TimeoutError, asyncio.TimeoutError) as e:
                    st.metrics["fallback"] = cause = str(e) or "timeout"
                    log.warning("generation deadline hit", extra={"turn": turn_id, "cause": cause})
                    if not st.displayed:
                        await self._emit(Ev.ERROR, component="llm", recoverable=True, turn_id=turn_id,
                                         message=f"The model didn't answer in time ({cause}).")
                        await self._dispatch(st, FALLBACKS["timeout"], queue)
            if not st.displayed:
                if source == "greeting" and self.s.personality.current.greeting:
                    st.metrics["fallback"] = "greeting_local"
                    await self._dispatch(st, self.s.personality.current.greeting, queue)
                else:
                    st.metrics["fallback"] = f"empty_answer:{st.finish_reason or 'unknown'}"
                    await self._emit(Ev.ERROR, component="llm", recoverable=True, turn_id=turn_id,
                                     message="The model returned no answer"
                                             + (" (it ran out of tokens while reasoning)."
                                                if st.finish_reason == "length" else "."))
                    await self._dispatch(st, FALLBACKS["empty"], queue)
            await self._emit(Ev.LLM_FINISHED, turn_id=turn_id,
                             text=strip_tags(self._strip_think("".join(raw))) or " ".join(st.displayed),
                             tools=[t["name"] for t in st.tools])
            await queue.put(None)
            await tts_task
        except asyncio.CancelledError:
            reason = self._cancel_reason or "interrupted"
            raise
        except LLMError as e:
            reason = "error"
            log.error("LLM failed", extra={"error": str(e), "turn": turn_id})
            await self._emit(Ev.ERROR, component="llm", message=str(e), recoverable=True, turn_id=turn_id)
            if source == "greeting" and self.s.personality.current.greeting and not st.displayed:
                st.metrics["fallback"] = "greeting_local"
                await self._emit(Ev.LLM_SENTENCE, turn_id=turn_id, index=0, text=self.s.personality.current.greeting)
                st.displayed.append(self.s.personality.current.greeting)
        except Exception as e:  # noqa: BLE001
            reason = "error"
            log.exception("turn failed")
            await self._emit(Ev.ERROR, component="pipeline", message=f"{e.__class__.__name__}: {e}", turn_id=turn_id)
        finally:
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            full = self._strip_think("".join(raw)).strip()
            delivered = (full or " ".join(st.displayed)) if reason == "completed" else " ".join(st.spoken).strip()
            asst_id = None
            if delivered:
                asst_id = self._remember_message("assistant", delivered, turn_id, interrupted=reason != "completed",
                                                 source="reply" if source in USER_SOURCES else source)
            st.metrics["turn_complete_ms"] = st.metrics["total_ms"] = st.ms()
            if st.usage:
                st.metrics["usage"] = st.usage
            if st.tools:
                st.metrics["tools"] = [t["name"] for t in st.tools]
            log.info("turn finished", extra={"turn": turn_id, "reason": reason, "source": source,
                                             **{k: v for k, v in st.metrics.items() if k != "usage"}})
            await self._emit(Ev.ASSISTANT_STOPPED, turn_id=turn_id, reason=reason)
            await self._emit(Ev.METRICS, turn_id=turn_id, **st.metrics)
            if not self._closed:
                self.state = "speaking" if self.client_playing else "listening"
                await self._emit(Ev.STATE, state=self.state)
            self.note_activity()
            if source in USER_SOURCES:
                self._after_user_turn(st, text, delivered, reason, user_msg_id, asst_id, started_at)

    async def _generate(self, st: _TurnState, messages: list[dict[str, Any]],
                        queue: "asyncio.Queue[Optional[_Sentence]]", raw: list[str]) -> None:
        cfg = self.s.config
        think, chunker = ThinkFilter(), SentenceChunker()
        tool_ctx = ToolContext(cfg, self.memory, self._ask_image if self.last_image_id else None, st.turn_id,
                               self.s.http)
        use_tools = cfg.tools.enabled and st.source in USER_SOURCES and bool(self.s.tools.active(tool_ctx))
        overall = st.t0 + cfg.llm.overall_timeout_s
        first_content = st.t0 + st.metrics.get("context_ms", 0) / 1000 + cfg.llm.first_content_timeout_s
        step = retries = 0
        seen_calls: set[tuple[str, str]] = set()
        while step <= cfg.tools.max_steps:
            llm = self.s.router.get("fast" if step == 0 else "smart") or self.s.llm
            tools = self.s.tools.schemas(tool_ctx) if use_tools and step < cfg.tools.max_steps else None
            calls = []
            step_text: list[str] = []
            if "provider_dispatch_ms" not in st.metrics:
                st.metrics["provider_dispatch_ms"] = st.ms()
            try:
                async for ev in self._bounded(llm.stream_events(messages, tools=tools), st, overall, first_content):
                    if ev.kind == "thinking":
                        st.metrics.setdefault("llm_first_thinking_ms", st.ms())
                        if not st.thinking_announced:
                            st.thinking_announced = True
                            await self._emit(Ev.ASSISTANT_THINKING, turn_id=st.turn_id)
                    elif ev.kind == "content":
                        st.progressed = True
                        if "llm_first_content_ms" not in st.metrics:
                            st.metrics["llm_first_content_ms"] = st.metrics["llm_first_token_ms"] = st.ms()
                        raw.append(ev.text)
                        step_text.append(ev.text)
                        visible = think.feed(ev.text)
                        if not visible:
                            continue
                        if cfg.frontend.show_token_stream:
                            await self._emit(Ev.LLM_TOKEN, turn_id=st.turn_id, text=visible)
                        for chunk in chunker.feed(visible):
                            await self._dispatch(st, chunk, queue)
                    elif ev.kind == "tool_calls":
                        st.progressed = True
                        calls.extend(ev.tool_calls)
                    elif ev.kind == "done":
                        st.finish_reason = ev.finish_reason
                        for k, v in ev.usage.items():
                            if isinstance(v, (int, float)):
                                st.usage[k] = st.usage.get(k, 0) + v
                        if step == 0 and ev.usage.get("prompt_tokens") is not None:
                            st.metrics["prompt_tokens"] = ev.usage["prompt_tokens"]
                            st.metrics["cached_tokens"] = ev.usage.get("cached_tokens")
            except LLMError as e:
                # One bounded retry, only before anything was delivered and only within the deadline.
                if e.retryable and not st.displayed and not calls and retries == 0 and time.perf_counter() < overall - 2:
                    retries += 1
                    st.metrics["retries"] = retries
                    await asyncio.sleep(0.3)
                    continue
                raise
            if tools is None:
                calls = []  # tools weren't offered this round: never execute calls the model emits anyway
            fresh = [c for c in calls if (c.name, json.dumps(c.arguments, sort_keys=True)) not in seen_calls]
            if not fresh or (step > 0 and "".join(step_text).strip()):
                # Nothing new to run, or the model already started answering after a tool result:
                # running more tools here only duplicates the answer. Finish with what was said.
                if calls and not fresh:
                    st.metrics["duplicate_tool_calls"] = st.metrics.get("duplicate_tool_calls", 0) + len(calls)
                break
            calls = fresh
            seen_calls.update((c.name, json.dumps(c.arguments, sort_keys=True)) for c in calls)
            for chunk in chunker.feed(think.flush()) + chunker.flush():  # speak "let me check" now
                await self._dispatch(st, chunk, queue)
            messages.append({"role": "assistant", "content": "".join(step_text),
                             "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]})
            for call in calls[:6]:  # bound per step
                await self._emit(Ev.TOOL_STARTED, turn_id=st.turn_id, tool=call.name, call_id=call.id)
                res = await self.s.tools.execute(call, tool_ctx)
                st.tools.append({"name": res.name, "ok": res.ok, "ms": res.ms})
                await self._emit(Ev.TOOL_FINISHED, turn_id=st.turn_id, tool=res.name, call_id=call.id, ok=res.ok,
                                 summary=res.summary, ms=res.ms)
                if res.ok and res.name == "remember" and self.memory and res.data and res.data.get("new"):
                    m = self.memory.store.get(res.data["id"])
                    if m:
                        await self._emit(Ev.MEMORY_CREATED, **m.public())
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": res.content})
            step += 1
        for chunk in chunker.feed(think.flush()) + chunker.flush():
            await self._dispatch(st, chunk, queue)

    async def _bounded(self, agen: AsyncIterator[LLMEvent], st: _TurnState, overall: float,
                       first_content: float) -> AsyncIterator[LLMEvent]:
        """Iterate provider events under the first-content and overall deadlines (I07)."""
        try:
            while True:
                now = time.perf_counter()
                deadline = overall if st.progressed else min(overall, first_content)
                try:
                    async with asyncio.timeout(max(0.01, deadline - now)):
                        ev = await agen.__anext__()
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise TimeoutError("overall_deadline" if deadline == overall else "first_content_deadline")
                yield ev
        finally:
            await agen.aclose()  # closes the HTTP stream -> the server stops generating

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_BLOCK.sub("", text)

    async def _dispatch(self, st: _TurnState, chunk: str, queue: "asyncio.Queue[Optional[_Sentence]]") -> None:
        for m in _ACT.finditer(chunk):  # body actions for the avatar: never spoken, never shown
            if m.group(1).lower() in _ACTIONS:
                await self._emit(Ev.AVATAR_ACTION, turn_id=st.turn_id, action=m.group(1).lower())
        chunk = _ACT.sub("", chunk)
        display, state = self.emotion.analyze(chunk)
        changed = self.emotion.update(state)
        if state:
            st.emotions.append(state)
        idx = len(st.displayed)
        if changed:
            await self._emit(Ev.EMOTION_CHANGED, turn_id=st.turn_id, sentence_index=idx, **changed.to_dict())
        if not display:
            return
        key = re.sub(r"\W+", "", display.lower())
        if key and key in st.seen_sentences:
            # the model repeated itself across a tool round ("Mia." ... tool ... "Mia."): show/say it once
            st.metrics["repeated_sentences_dropped"] = st.metrics.get("repeated_sentences_dropped", 0) + 1
            return
        st.seen_sentences.add(key)
        st.displayed.append(display)
        st.metrics.setdefault("llm_first_deliverable_ms", st.ms())
        await self._emit(Ev.LLM_SENTENCE, turn_id=st.turn_id, index=idx, text=display)
        speech = clean_for_tts(display, self.s.config.tts.strip_actions)
        tcfg = self.s.config.tts
        if st.source == "voice" and speech and not st.speech_capped:
            words = len(speech.split())
            over_words = tcfg.max_spoken_words and st.spoken_words + words > tcfg.max_spoken_words and st.spoken_sentences
            over_sent = tcfg.max_spoken_sentences and st.spoken_sentences >= tcfg.max_spoken_sentences
            if over_words or over_sent:
                st.speech_capped = True  # the rest is shown on screen, not spoken
                st.metrics["speech_capped_at"] = st.spoken_sentences
            else:
                st.spoken_words += words
                st.spoken_sentences += 1
        if st.speech_capped:
            speech = ""
            st.spoken.append(chunk)  # displayed (text mode delivery for this part)
            return
        await queue.put(_Sentence(idx, chunk, display, speech, self.emotion.current))

    async def _announce_speaking(self, st: _TurnState) -> None:
        if not st.speaking_announced:
            st.speaking_announced = True
            st.metrics["first_audio_ms"] = st.metrics["first_answer_audio_server_ms"] = st.ms()
            await self._emit(Ev.ASSISTANT_SPEAKING, turn_id=st.turn_id)
            await self._set_state("speaking")

    async def _tts_down(self, err: Optional[str] = None) -> None:
        """Mark the TTS server down for 30 s (shared by all turns/tabs) and tell the user once, calmly."""
        self.s.status["tts_down_until"] = time.monotonic() + 30.0
        self.s.status["tts_down_notified"] = True
        if self._tts_notice_sent:
            return
        self._tts_notice_sent = True
        browser = self.s.config.tts.fallback == "browser"
        log.warning("TTS server unreachable; using fallback", extra={"fallback": self.s.config.tts.fallback,
                                                                      "error": (err or "")[:200]})
        await self._emit(Ev.ERROR, component="tts", level="info", recoverable=True,
                         message=("Riko's own voice (GPT-SoVITS) isn't running, so she's using your browser's voice "
                                  "for now. She switches back automatically once it starts."
                                  if browser else "Riko's voice server isn't running; replies are text only."))

    async def _tts_worker(self, st: _TurnState, queue: "asyncio.Queue[Optional[_Sentence]]") -> None:
        tts = self.s.tts
        # A dead TTS server is skipped for 30 s across turns, then retried; meanwhile the fallback voice speaks.
        tts_down = self.s.status.get("tts_down_until", 0.0) > time.monotonic()
        browser_fallback = self.s.config.tts.fallback == "browser"
        if tts_down and not self._tts_notice_sent:
            await self._tts_down()  # detected at startup: still tell the user once
        mood = self._mood()
        mood_speed = MoodModel.speed_factor(mood) if mood else 1.0
        while True:
            job = await queue.get()
            if job is None:
                return
            if job.speech and st.turn_id != self.turn_id:
                return  # stale turn: never revive playback (I02)
            if not job.speech or self.muted or tts.name == "none" or (tts_down and not browser_fallback):
                st.spoken.append(job.raw)
                continue
            if not tts.produces_audio or tts_down:  # browser speechSynthesis (configured, or as fallback)
                st.spoken.append(job.raw)
                await self._announce_speaking(st)
                await self._emit(Ev.TTS_SPEAK, turn_id=st.turn_id, index=job.index, text=job.speech,
                                 emotion=job.emotion.emotion, speed=round(job.emotion.speed_factor * mood_speed, 3))
                continue
            await self._emit(Ev.TTS_STARTED, turn_id=st.turn_id, index=job.index)
            req = SpeechRequest(job.speech, self.s.config.tts.voices[self.voice_name], job.emotion.emotion,
                                job.emotion.intensity, round(job.emotion.speed_factor * mood_speed, 3))
            first = True
            try:
                async for chunk in tts.synthesize(req):
                    if first:
                        first = False
                        st.spoken.append(job.raw)
                        await self._announce_speaking(st)
                    await self.sink.send_audio(st.turn_id, job.index, chunk.sample_rate, chunk.pcm)
            except TTSError as e:
                if e.unavailable:  # server not running: fall back for this and later sentences, no scary error
                    tts_down = True
                    await self._tts_down(str(e))
                    if first and browser_fallback:
                        st.spoken.append(job.raw)
                        await self._announce_speaking(st)
                        await self._emit(Ev.TTS_SPEAK, turn_id=st.turn_id, index=job.index, text=job.speech,
                                         emotion=job.emotion.emotion,
                                         speed=round(job.emotion.speed_factor * mood_speed, 3))
                        first = False
                else:
                    log.error("TTS failed", extra={"error": str(e), "sentence": job.index})
                    await self._emit(Ev.ERROR, component="tts", message=str(e), recoverable=True, turn_id=st.turn_id)
                if first:
                    st.spoken.append(job.raw)  # the text was still shown to the user
            else:
                if not first and self.s.status.pop("tts_down_notified", None):
                    log.info("TTS server is back")
            await self._emit(Ev.TTS_FINISHED, turn_id=st.turn_id, index=job.index)

    # ---------------------------------------------------------------- after a user turn
    def _after_user_turn(self, st: _TurnState, user_text: str, delivered: str, reason: str,
                         user_msg_id: Optional[int], asst_id: Optional[int], started_at: float) -> None:
        mem = self.memory
        if reason == "completed":
            self.user_turns += 1
            self.proactive.user_turn_completed()
            if mem:
                mem.store.record_user_turn(mem.character, self.logical_key, time.time(),
                                           self.s.config.memory.familiarity.session_gap_minutes * 60,
                                           profile=mem.profile)
        if self.s.config.emotion.mood_enabled and self.s.memory and reason in ("completed", "barge_in"):
            m = self.s.memory
            new = self.s.mood_model.update(self._mood(), time.time(), st.emotions, classify_user_affect(user_text))
            m.store.set_mood(m.character, new.valence, new.arousal, new.updated_at, profile=m.profile)
            self._spawn(self._emit(Ev.MOOD, **MoodModel.to_dict(new)))
        if mem and user_msg_id is not None:  # the user's words are final even if the reply was interrupted
            ids = [i for i in (user_msg_id, asst_id) if i is not None]
            self.s.scheduler.submit(f"extract:{mem.profile}:{mem.character}:{user_msg_id}",
                                    lambda: self._extract_job(user_text, delivered, started_at, st.turn_id, ids))

    async def _extract_job(self, user_text: str, delivered: str, source_ts: float, turn_id: int,
                           msg_ids: list[int]) -> None:
        mem = self.memory
        if mem is None:
            return
        created = await mem.extract_from_turn(user_text, delivered, source_ts=source_ts, turn_id=turn_id)
        mem.store.mark_extracted(msg_ids)
        for m in created:
            await self._emit(Ev.MEMORY_CREATED, **m.public())
        await mem.maybe_summarize()

    # ================================================================ client state / settings
    async def set_playback(self, playing: bool) -> None:
        self.client_playing = playing
        if not playing:
            self.note_activity()
            if not self.turn_running:
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
                self.s.reset_mood_model()
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
            if kw.get("memory_enabled") is not None and self.s.memory:
                self.s.memory.set_enabled(bool(kw["memory_enabled"]))
            if kw.get("proactive_level") in ("off", "minimal", "normal"):
                self.proactive_level = kw["proactive_level"]
        except PersonalityError as e:
            await self._emit(Ev.ERROR, component="settings", message=str(e), recoverable=True)
        await self._emit(Ev.SETTINGS, settings=self.settings())
