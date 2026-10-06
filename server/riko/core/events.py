"""Server -> client events and the binary audio frame format.

JSON events: ``{"type": <legacy name>, "event": <dotted name>, "ts": <unix seconds>, ...fields}``

``type`` keeps the original wire names so existing clients keep working; ``event`` carries the dotted
name (``assistant.token``, ``tool.started`` ...). New clients switch on ``event``. Unknown events are
ignored by both, so adding events never breaks a client.

Binary audio frames (server -> client), little endian:
    bytes 0-3   magic  b"RKA1"
    bytes 4-7   uint32 turn_id         (client drops audio from stale turns)
    bytes 8-11  uint32 sentence_index  (client syncs subtitles to playback)
    bytes 12-15 uint32 sample_rate
    bytes 16-   int16 mono PCM

Binary audio frames (client -> server): raw int16 mono PCM at 16 kHz.
"""
from __future__ import annotations

import struct
import time
from typing import Any

AUDIO_MAGIC = b"RKA1"
AUDIO_HEADER = struct.Struct("<4sIII")
PROTOCOL_VERSION = 2


class Ev:
    HELLO = "hello"
    STATE = "state"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_STOPPED = "user_speech_stopped"
    TRANSCRIPTION = "transcription"
    LLM_STARTED = "llm_started"
    ASSISTANT_THINKING = "assistant_thinking"   # reasoning in progress - never carries reasoning text (I01)
    LLM_TOKEN = "llm_token"
    LLM_SENTENCE = "llm_sentence"
    LLM_FINISHED = "llm_finished"
    TTS_STARTED = "tts_started"
    TTS_FINISHED = "tts_finished"
    TTS_SPEAK = "tts_speak"  # browser-side TTS fallback
    ASSISTANT_SPEAKING = "assistant_speaking"
    ASSISTANT_STOPPED = "assistant_stopped"
    ASSISTANT_DUCK = "assistant_duck"
    ASSISTANT_UNDUCK = "assistant_unduck"
    EMOTION_CHANGED = "emotion_changed"
    MOOD = "mood"
    AVATAR_ACTION = "avatar_action"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    IMAGE_ANALYZED = "image_analyzed"
    MEMORY_CREATED = "memory_created"
    MEMORY_RETRIEVED = "memory_retrieved"
    SETTINGS = "settings"
    METRICS = "metrics"
    ERROR = "error"
    PONG = "pong"


DOTTED = {
    Ev.HELLO: "conversation.started", Ev.STATE: "state.changed",
    Ev.USER_SPEECH_STARTED: "user.speaking", Ev.USER_SPEECH_STOPPED: "user.stopped",
    Ev.TRANSCRIPTION: "transcription.final", Ev.LLM_STARTED: "assistant.started",
    Ev.ASSISTANT_THINKING: "assistant.thinking", Ev.LLM_TOKEN: "assistant.token",
    Ev.LLM_SENTENCE: "assistant.sentence", Ev.LLM_FINISHED: "assistant.text_done",
    Ev.TTS_STARTED: "tts.started", Ev.TTS_FINISHED: "tts.finished", Ev.TTS_SPEAK: "tts.speak_in_browser",
    Ev.ASSISTANT_SPEAKING: "assistant.speaking", Ev.ASSISTANT_STOPPED: "assistant.finished",
    Ev.ASSISTANT_DUCK: "assistant.duck", Ev.ASSISTANT_UNDUCK: "assistant.unduck",
    Ev.EMOTION_CHANGED: "assistant.emotion", Ev.MOOD: "assistant.mood", Ev.AVATAR_ACTION: "avatar.action",
    Ev.TOOL_STARTED: "tool.started", Ev.TOOL_FINISHED: "tool.finished", Ev.IMAGE_ANALYZED: "image.analyzed",
    Ev.MEMORY_CREATED: "memory.created", Ev.MEMORY_RETRIEVED: "memory.retrieved",
    Ev.SETTINGS: "settings.changed", Ev.METRICS: "metrics", Ev.ERROR: "error", Ev.PONG: "pong",
}


def make_event(type_: str, **fields: Any) -> dict[str, Any]:
    return {"type": type_, "event": DOTTED.get(type_, type_), "ts": round(time.time(), 3), **fields}


def encode_audio_frame(turn_id: int, sentence_index: int, sample_rate: int, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(AUDIO_MAGIC, turn_id, sentence_index, sample_rate) + pcm


def decode_audio_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    magic, turn_id, idx, sr = AUDIO_HEADER.unpack_from(frame, 0)
    if magic != AUDIO_MAGIC:
        raise ValueError("not a Riko audio frame")
    return turn_id, idx, sr, frame[AUDIO_HEADER.size:]
