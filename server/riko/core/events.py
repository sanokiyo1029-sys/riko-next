"""Server -> client events and the binary audio frame format.

JSON events: ``{"type": <name>, "ts": <unix seconds>, ...fields}``

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


class Ev:
    HELLO = "hello"
    STATE = "state"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_STOPPED = "user_speech_stopped"
    TRANSCRIPTION = "transcription"
    LLM_STARTED = "llm_started"
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
    MEMORY_CREATED = "memory_created"
    MEMORY_RETRIEVED = "memory_retrieved"
    SETTINGS = "settings"
    METRICS = "metrics"
    ERROR = "error"
    PONG = "pong"


def make_event(type_: str, **fields: Any) -> dict[str, Any]:
    return {"type": type_, "ts": round(time.time(), 3), **fields}


def encode_audio_frame(turn_id: int, sentence_index: int, sample_rate: int, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(AUDIO_MAGIC, turn_id, sentence_index, sample_rate) + pcm


def decode_audio_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    magic, turn_id, idx, sr = AUDIO_HEADER.unpack_from(frame, 0)
    if magic != AUDIO_MAGIC:
        raise ValueError("not a Riko audio frame")
    return turn_id, idx, sr, frame[AUDIO_HEADER.size:]
